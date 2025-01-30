# Bitcoin Chatbot

## Overview
Bitcoin Chatbot is an AI-powered chatbot built using Streamlit, LangChain, and Ollama. It allows users to ask questions related to Bitcoin and retrieves relevant information using a combination of vector embeddings, BM25 ranking, and an LLM-based response system.

## Features
- Loads and processes a Bitcoin-related PDF document
- Uses Google Generative AI embeddings for vector-based retrieval
- Stores and retrieves data using Chroma vector database
- Implements a BM25 reranker to improve retrieval quality
- Utilizes Ollama as the LLM for generating responses
- Preprocesses user input with NLP techniques (spell correction, entity recognition, sentiment analysis)
- Maintains chat history for context-aware conversations
- Handles greetings and common conversational inputs

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Pip
- Ollama (running on `localhost:11434`)
- Google Generative AI API key
- Required Python packages

### Install Dependencies
```sh
pip install streamlit langchain langchain_community langchain_google_genai rank_bm25 nltk spacy textblob chromadb
```

### Download SpaCy Model
```sh
python -m spacy download en_core_web_sm
```

## Setup
1. Place your `bitcoin.pdf` file in the project directory.
2. Store your Google API Key in Streamlit secrets:
   ```sh
   mkdir -p ~/.streamlit
   echo "[secrets]" > ~/.streamlit/secrets.toml
   echo "GOOGLE_API_KEY='your_api_key_here'" >> ~/.streamlit/secrets.toml
   ```
3. Run the chatbot:
   ```sh
   streamlit run app.py
   ```

## Usage
- Open the Streamlit UI and ask any question related to Bitcoin.
- The chatbot will retrieve relevant information and provide a detailed response.
- It preprocesses queries for better understanding and relevance.
- It maintains context across multiple user interactions.

## Project Structure
```
📂 Bitcoin Chatbot
├── app.py               # Main Streamlit app
├── bitcoin.pdf          # Source document for chatbot knowledge
├── chroma_db/           # Chroma vector database storage
├── requirements.txt     # List of dependencies
└── README.md            # Project documentation
```

## Future Improvements
- Add support for multiple documents
- Improve query intent detection
- Enhance LLM response quality with fine-tuned models
- Expand NLP preprocessing with advanced topic modeling

## License
This project is licensed under the MIT License.

