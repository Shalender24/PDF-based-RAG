import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class Retriever:
    def __init__(self, data_path="data/", db_path="vectordb/db_faiss"):
        self.data_path = data_path
        self.db_path = db_path
        self.embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    def load_documents(self):
        loader = DirectoryLoader(self.data_path, glob="*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        return documents

    def create_chunks(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        text_chunks = text_splitter.split_documents(documents)

        os.makedirs("chunks", exist_ok=True)
        for i, chunk in enumerate(text_chunks):
            with open(f"chunks/chunk{i}.txt", 'w', encoding='utf-8') as f:
                f.write(str(chunk))
        return text_chunks

    def create_vector_db(self, text_chunks):
        db = FAISS.from_documents(text_chunks, self.embedding_model)
        db.save_local(self.db_path)

    def load_vector_db(self):
        return FAISS.load_local(self.db_path, self.embedding_model, allow_dangerous_deserialization=True)
