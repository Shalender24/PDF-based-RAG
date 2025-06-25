# RAG-Powered PDF Chatbot using LangChain + Mistral

A smart, context-aware chatbot that allows you to **ask questions from your own PDF documents** using:

- Semantic search with FAISS + Sentence Transformers
- Local inference with Mistral 7B model
- Chat interface powered by Streamlit
- Safe answers: avoids hallucination and responds gracefully to unknowns

---

## Features

- PDF ingestion and chunking
- Fast semantic search using FAISS vector store
- Embedding with Sentence Transformers (`all-MiniLM-L6-v2`)
- Contextual Q&A using a local Mistral-7B-Instruct-v0.3 model
- Fully interactive chat UI built with Streamlit
- Returns source documents for every answer
- Graceful handling of unknown or unanswerable queries

---


---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
2. Create and Activate a Virtual Environment

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
3. Install Dependencies

pip install -r requirements.txt
4. Add Your PDF Files
Place all your .pdf documents in the data/ directory.

 How It Works
Step 1: Preprocessing
PDFs are loaded and chunked into smaller segments.

Each chunk is embedded using sentence-transformers/all-MiniLM-L6-v2.

The embeddings are stored in a FAISS vector store.

Step 2: Retrieval-Augmented Generation (RAG)
On each user query:

Relevant chunks are retrieved from the vector DB.

Chunks are passed to the local Mistral 7B model along with the question.

A structured, trustworthy answer is generated and shown in the UI.

 Prompt Template (Used for Q&A)

You are a domain-specific assistant. You are provided with context from trusted documents.
Only answer based on the context. If unsure, say: "I'm sorry, I couldn't find an exact answer to your question in the available documents."

Context:
{context}

Question:
{question}

Answer (well-structured, use bullets/headings when helpful):
 Run the Chatbot

streamlit run app.py
Then open http://localhost:8501 in your browser.

 Example Queries
"What is the dosage of Paracetamol?"
"List side effects of Ibuprofen"
"When was this policy updated?"

 If the answer is not found in your documents, the bot will respond:

"I'm sorry, I couldn't find an exact answer to your question in the available documents."

 Example requirements.txt
Generated with pip freeze > requirements.txt:

streamlit
langchain
langchain-community
langchain-huggingface
transformers
torch
faiss-cpu
sentence-transformers
 Handling Unseen Queries
This project is configured to avoid hallucinations by:

Strictly using the given context chunks

Responding with a fallback message when no relevant answer is found

 Future Improvements
 Add file uploader to the Streamlit UI

 Stream token output in real-time

 Integrate reranking (e.g., Cohere Reranker or BGE)

 Support for multiple LLM backends (OpenAI, Gemini, Claude)

 Author
Shalender
```
