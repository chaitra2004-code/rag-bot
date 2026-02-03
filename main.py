import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

load_dotenv()

def build_rag(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0),
        retriever=db.as_retriever()
    )

    return qa


if __name__ == "__main__":
    qa = build_rag("sample.pdf")

    while True:
        query = input("Ask question: ")
        answer = qa.run(query)
        print("Answer:", answer)
