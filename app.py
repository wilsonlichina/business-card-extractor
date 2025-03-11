import os
import json
import logging
import base64
from io import BytesIO
from PIL import Image
import gradio as gr
import boto3
from dotenv import load_dotenv
import time
from random import uniform

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS Credentials and Configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-west-2')  # Default to us-west-2 if not set

# Initialize Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID, 
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Model IDs
MODEL_IDS = {
    "Nova Lite": "us.amazon.nova-lite-v1:0",
    "Claude 3.5 Sonnet V2": "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
}

def clean_json_string(s):
    """
    Clean JSON string by removing control characters and fixing common issues.
    
    Args:
        s: Input JSON string
        
    Returns:
        str: Cleaned JSON string
        
    Raises:
        ValueError: If no valid JSON object is found
    """
    # Remove control characters
    clean = ''.join(char for char in s if ord(char) >= 32 or char in '\n\r\t')
    
    # Find the first { and last }
    start = clean.find('{')
    end = clean.rfind('}') + 1
    if start == -1 or end == 0:
        raise ValueError("No valid JSON object found in string")
    
    json_str = clean[start:end]
    
    # Fix common JSON formatting issues
    json_str = json_str.replace('\n', ' ')
    json_str = json_str.replace('\r', ' ')
    json_str = ' '.join(json_str.split())  # Normalize whitespace
    
    return json_str

def image_to_bytes(image):
    """Convert PIL Image to bytes and format for Bedrock API"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return "jpeg", buffered.getvalue()

def extract_business_card_info(image, model_name="Nova Lite", max_retries=3):
    """
    Extract business card information (email) using Amazon Bedrock Converse API
    
    Args:
        image: PIL Image object
        model_name: Name of the model to use
        max_retries: Maximum number of retry attempts for API calls
    
    Returns:
        dict: Result with extracted information and status
    """
    try:
        # Log which model is being used
        logger.info(f"Using model: {model_name}")
        
        # Resize image if needed
        max_size = 1600  # Reasonable size for API calls
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            image = image.resize((int(image.width * ratio), int(image.height * ratio)))
            logger.info(f"Image resized to {image.size}")
        
        # Convert image to bytes format
        file_type, image_bytes = image_to_bytes(image)
        
        # Get model ID
        model_id = MODEL_IDS.get(model_name)
        if not model_id:
            logger.error(f"Unknown model: {model_name}")
            return {"status": "error", "message": f"Unknown model: {model_name}"}
        
        # System message
        system_message = [{
            "text": "You are an expert at extracting information from business cards."
        }]
        
        # User message with image
        user_message = {
            "role": "user",
            "content": [
                {
                    "image": {
                        "format": file_type,
                        "source": {
                            "bytes": image_bytes
                        }
                    }
                },
                {
                    "text": """This is an image of a business card. 
                    Extract the following information:
                    1. Email address

                    Respond in valid JSON format with fields "email". 
                    If any information is not found, use an empty string for that field.
                    Example: {"email": "john@example.com"}"""
                }
            ]
        }

        # Inference config
        inference_config = {
            "maxTokens": 1000,
            "temperature": 0.1
        }
        
        # API call with retries
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Add random delay between retries to avoid rate limiting
                if retry_count > 0:
                    delay = min(2 ** retry_count + uniform(0, 1), 10)  # Max 10 second delay
                    logger.info(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                
                logger.info(f"Calling Amazon Bedrock Converse API with {model_id}")
                response = bedrock_runtime.converse(
                    modelId=model_id,
                    messages=[user_message],
                    system=system_message,
                    inferenceConfig=inference_config
                )
                
                output_text = response["output"]["message"]["content"][0]["text"]
                logger.info(f"Raw model response: {output_text}")
                
                # Clean and parse JSON
                try:
                    json_str = clean_json_string(output_text)
                    result = json.loads(json_str)
                    
                    # Ensure all required fields exist
                    for field in ["email"]:
                        if field not in result:
                            result[field] = ""
                    
                    return {
                        "status": "success",
                        "data": result,
                        "raw_response": output_text
                    }
                except (ValueError, json.JSONDecodeError) as e:
                    logger.warning(f"Failed to parse JSON: {str(e)}")
                    
                    # If we've already retried enough, return error
                    if retry_count == max_retries - 1:
                        return {
                            "status": "error",
                            "message": f"Failed to parse response as JSON: {str(e)}",
                            "raw_response": output_text
                        }
                    retry_count += 1
                    continue
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"API error: {error_msg}")
                
                if 'ThrottlingException' in error_msg and retry_count < max_retries - 1:
                    retry_count += 1
                    continue
                else:
                    return {
                        "status": "error",
                        "message": f"API error: {error_msg}"
                    }
        
        # This should not be reached with the logic above, but just in case
        return {
            "status": "error",
            "message": f"Failed after {max_retries} retries"
        }
        
    except Exception as e:
        logger.exception(f"Error in business card extraction: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }


def process_image(image, model_name):
    """Process the image and extract business card information"""
    if image is None:
        return "Please upload an image", "No image provided", {}
    
    # Convert to PIL Image if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Setup log capture
    log_messages = []
    log_handler = logging.StreamHandler(BytesIO())
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_handler.setFormatter(formatter)
    logger.addHandler(log_handler)
    
    # Process the image
    result = extract_business_card_info(image, model_name)
    
    # Get logs
    log_handler.flush()
    log_stream = log_handler.stream
    log_stream.seek(0)
    logs = log_stream.read().decode('utf-8')
    logger.removeHandler(log_handler)
    
    # Format result for display
    if result["status"] == "success":
        data = result["data"]
        
        # Build message for display
        message_parts = []
        if data["email"]:
            message_parts.append(f"Email: {data['email']}")
            
        if message_parts:
            message = "\n".join(message_parts)
        else:
            message = "No information found in the business card image."
        
        return message, logs, data
    else:
        return f"Error: {result['message']}", logs, {}


# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Business Card Information Extractor") as demo:
        gr.Markdown("# Business Card Information Extractor")
        gr.Markdown("Upload a business card image to extract email address")
        
        with gr.Row():
            with gr.Column(scale=2):
                image_input = gr.Image(
                    type="pil", 
                    label="Business Card Image"
                )
                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_IDS.keys()), 
                    value="Nova Lite", 
                    label="AI Model"
                )
                process_btn = gr.Button("Extract Information", variant="primary")
            
            with gr.Column(scale=3):
                result_text = gr.Textbox(label="Extracted Information", lines=3)
                logs_text = gr.Textbox(label="System Logs", lines=10)
                json_output = gr.JSON(label="JSON Output")
        
        process_btn.click(
            fn=process_image,
            inputs=[image_input, model_dropdown],
            outputs=[result_text, logs_text, json_output]
        )
        
        gr.Markdown("### Instructions:")
        gr.Markdown("""
        1. Upload a clear image of a business card
        2. Select the AI model to use (Nova Lite is faster, Claude 3.5 Sonnet V2 may be more accurate)
        3. Click "Extract Information"
        4. View the extracted information in the output panel and JSON format
        """)
        
        gr.Markdown("### Notes:")
        gr.Markdown("""
        - Requires valid AWS credentials with Amazon Bedrock access
        - For best results, ensure the business card image is well-lit and clearly visible
        - This system extracts email address from business cards
        """)
        
    return demo

# Main execution
if __name__ == "__main__":
    # Check for credentials
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("Warning: AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
    
    # Launch Gradio app
    demo = create_interface()
    demo.launch(share=False)
