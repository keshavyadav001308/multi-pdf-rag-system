import os
from langchain_community.document_loaders import PyPDFLoader

def load_pdfs_from_folder(folder_path: str):
    documents = []

    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file)
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            documents.extend(docs)

    return documents
