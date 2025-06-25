from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from src.retriever import Retriever
from src.generator import Generator


class QAPipeline:
    def __init__(self):
        self.retriever = Retriever()
        self.generator = Generator()
        self.vector_db = self.retriever.load_vector_db()
        self.llm = self.generator.load_local_model()
        self.prompt_template = self.set_prompt()


    def get_chain(self):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.prompt_template}
        )
    def set_prompt(self):
        template = """
        You are a domain-specific assistant. You are provided with some extracted context from trusted documents.

        Follow these rules:
        1. Only answer based on the context provided. DO NOT use outside knowledge.
        2. If the answer cannot be found in the context, say:
        "I'm sorry, I couldn't find an exact answer to your question in the available documents."
        3. Always format your answers with headings, bullet points, or numbered lists when appropriate.

        ---

        Context:
        {context}

        ---

        User Question:
        {question}

        ---

        Your Structured Answer:
        """
        return PromptTemplate(template=template, input_variables=["context", "question"])
