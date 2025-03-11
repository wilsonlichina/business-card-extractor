# Business Card Information Extractor

A Python application that extracts contact information (name, email, phone) from business card images using Amazon Bedrock's Converse API.

## Features

- 📷 Upload business card images
- 👤 Extract name, email, and phone number using AI
- 🔄 Switch between AI models (Nova Lite or Claude 3.5 Sonnet V2)
- 📊 View results in both text and JSON format
- 📜 Display system processing logs

## Requirements

- Python 3.8 or higher
- AWS account with Amazon Bedrock access
- API access to the following models:
  - Amazon Nova Lite
  - Claude 3.5 Sonnet V2 (optional)

## Installation

1. Clone this repository or download the files

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file based on the `.env.example` template:

```bash
cp .env.example .env
```

4. Edit the `.env` file with your AWS credentials:

```
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_REGION=us-east-1  # Change as needed
```

## AWS Setup

To use this application, you need:

1. An AWS account with Amazon Bedrock access
2. IAM permissions for Amazon Bedrock
3. Model access enabled for:
   - Amazon Nova Lite (us.amazon.nova-lite-v1:0)
   - Claude 3.5 Sonnet V2 (us.anthropic.claude-3-5-sonnet-20241022-v2:0)

### Obtaining AWS Credentials

1. Go to your AWS IAM console
2. Create a new IAM user or use an existing one
3. Attach the `AmazonBedrockFullAccess` policy (or create a custom policy with more restricted permissions)
4. Generate access keys and add them to your `.env` file

## Usage

1. Run the application:

```bash
python app.py
```

2. Open your web browser and navigate to: `http://127.0.0.1:7860`

3. Use the interface to:
   - Upload a business card image
   - Select your preferred AI model
   - Click "Extract Information"
   - View the extracted name, email, and phone in text and JSON format
   - Check system logs for processing details

## How It Works

The application uses the Amazon Bedrock Converse API to analyze business card images:

1. User uploads an image through the Gradio interface
2. The selected AI model (Nova Lite or Claude 3.5 Sonnet) processes the image
3. The system extracts the contact information using computer vision and text recognition
4. Results are parsed, formatted, and displayed to the user

## Best Practices

- Use clear, high-resolution images of business cards
- Ensure good lighting and minimal glare on the card
- Position the card so text is clearly visible and not cut off
- For better accuracy with difficult-to-read cards, try the Claude 3.5 Sonnet V2 model

## Limitations

- Performance depends on image quality and text clarity
- Results may vary depending on the selected model and business card format
- Some non-standard or creative business card designs may be challenging to process

## License

This project is available under the MIT license.
