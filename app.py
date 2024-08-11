import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferWindowMemory
from rank_bm25 import BM25Okapi
import nltk
from textblob import TextBlob
import spacy

# Ensure you have NLTK and SpaCy resources
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

# Load and split PDF
loader = PyPDFLoader("bitcoin.pdf")
pages = loader.load_and_split()

text_splitter = CharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(pages)

# Set API key
api_key = st.secrets["GOOGLE_API_KEY"]
os.environ['GOOGLE_API_KEY'] = api_key

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Create Chroma vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# Initialize LLM
llm = Ollama(base_url="http://localhost:11434", temperature=0.7)

# Define preprocessing functions
def preprocess_prompt(prompt):
    # Correct spelling and grammar
    blob = TextBlob(prompt)
    corrected_prompt = str(blob.correct())
    
    # Normalize text
    normalized_prompt = corrected_prompt.lower().strip()
    
    # Tokenize text
    tokens = nltk.word_tokenize(normalized_prompt)
    
    # Perform Named Entity Recognition
    doc = nlp(normalized_prompt)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Perform sentiment analysis
    sentiment = blob.sentiment

    # Detect user intent (basic implementation, customize as needed)
    intent = "query"
    
    # Incorporate context if available (simplified)
    context = " ".join([msg["content"] for msg in st.session_state.messages])
    
    # Language detection and translation (basic implementation)
    language = "en"
    translated_prompt = corrected_prompt
    
    # Reduce noise from prompt
    clean_prompt = " ".join([word for word in tokens if word.isalnum()])
    
    # Extract keywords (basic implementation)
    keywords = list(set(tokens))
    
    # Perform topic modeling (simplified)
    topics = ["bitcoin"]
    
    # Personalize query if applicable (simplified)
    personalized_prompt = clean_prompt
    
    return personalized_prompt, context, intent, sentiment, keywords, topics

# Introduce BM25 Reranker
class CustomBM25Reranker:
    def __init__(self, documents):
        self.bm25 = BM25Okapi([doc.split() for doc in documents])
    
    def rerank(self, query, documents):
        scores = self.bm25.get_scores(query.split())
        return scores[:len(documents)]

# Define the custom RetrievalQA class
class SimpleRetrievalQA:
    def __init__(self, llm, vectorstore, reranker, memory):
        self.llm = llm
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.memory = memory

    def _retrieve(self, query):
        try:
            # Ensure query is a string
            if not isinstance(query, str):
                raise ValueError(f"Query must be a string, got {type(query)}.")
            
            # Perform similarity search using Chroma vector store
            results = self.vectorstore.similarity_search(query, k=5)  # Use string query here
            
            # Extract document texts from results
            documents = [result.page_content for result in results]  # Accessing the page_content attribute directly

            # Debug output
            print(f"Retrieved documents: {documents}")
            
            if not documents:
                return []

            # Apply reranker
            scores = self.reranker.rerank(query, documents)
            
            # Debug output
            print(f"Scores: {scores}")

            if len(scores) != len(documents):
                raise ValueError("Scores and documents length mismatch.")

            # Sort documents based on scores
            ranked_docs = [documents[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)]
            
            return ranked_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

    def _run(self, query):
        documents = self._retrieve(query)
        
        # Debug output
        print(f"Documents for LLM: {documents}")
        
        if not documents:
            return "No relevant documents found."

        # Manually combine documents
        combined_document = " ".join(documents)
        
        # Pass the combined document to the LLM
        response = self.llm(combined_document)
        
        return response

# Initialize custom QA chain
qa_chain = SimpleRetrievalQA(
    llm=llm,
    vectorstore=vectorstore,
    reranker=CustomBM25Reranker([chunk.page_content for chunk in chunks]),
    memory=ConversationBufferWindowMemory(k=5)
)

# Streamlit UI
st.title("Bitcoin Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask me anything about Bitcoin!"):
    # Handle greetings
    greetings = ["hi", "hello", "hey"]
    if prompt.lower().strip() in greetings:
        response = "Hello! How can I assist you with Bitcoin today?"
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
    else:
        # Preprocess prompt
        processed_prompt, context, intent, sentiment, keywords, topics = preprocess_prompt(prompt)
        st.session_state.messages.append({"role": "user", "content": processed_prompt})
        with st.chat_message("user"):
            st.markdown(processed_prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = qa_chain._run(processed_prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
