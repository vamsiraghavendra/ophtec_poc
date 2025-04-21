# Ophtec Knowledge Bot

A Streamlit-based chatbot for ophthalmology knowledge, specializing in Ophtec products.

## Features

- Interactive chat interface
- Role-based responses (Doctor/Sales)
- Category-specific knowledge (IOLs/CTR/General)
- Context-aware responses
- Professional medical information delivery

## Setup

1. Clone the repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key in Streamlit secrets or `.env` file
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Environment Variables

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key

You can set these either in a `.env` file locally or in Streamlit's secrets management when deploying.

## Deployment

This app is designed to be deployed on Streamlit Cloud:

1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Set up your environment variables in Streamlit's secrets management
4. Deploy!

## Usage

1. Enter your name and role (Doctor/Sales Rep)
2. Select a mode (General/IOLs/CTR)
3. Start asking questions!

## License

MIT License 