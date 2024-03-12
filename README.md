# chat_web
Welcome to the Chat_web repository! ChatWeb is a project that combines the power of LangChain, a language model, with the simplicity of Streamlit to create an interactive application. The application is designed to fetch relevant information from a specified website, making it a versatile tool for users who want quick access to data without navigating through the entire site.

## Features
* Web Url Upload: Easily upload web url to extract information.
* Select LLM: You can easily select models given in the selectbox (OpenAI and Llama2)
* Contextual Queries: Query the system to get answers based on the context provided by your web documents.
* General Conversation: Use ChatPDF for casual and general conversations with the integrated chatbot.
* Powered by LangChain: Interaction with Large Language Models (LLM) is facilitated through LangChain.
* Interactive Web App: Utilizes Streamlit, a Python library for creating interactive web applications.

## Getting Started
To run this project locally, follow these simple steps:
1. Clone the repository to your local machine:
    ```bash
    git clone https://github.com/AashishChapain/chat_web.git
    ```
2. Navigate to the project directory:
    ```bash
    cd chat_web
    ```
3. Install the required dependencies. It's recommended to use a virtual environment
    ```bash
    pip install -r requirements.txt
    ```
4. Load your OpenAI API Key:  
    create your .env file and provide your OpenAI API key:  
        -OPENAI_API_KEY = "your openai key"

5. Install Ollama:  
    github repo: https://github.com/ollama/ollama  
    langchain documentation: https://python.langchain.com/docs/integrations/llms/ollama

6. Run the streamlit application using the following command:
    ```bash
    streamlit run src/app.py
    ```

**Note:** Please select your model and provide url of the website.