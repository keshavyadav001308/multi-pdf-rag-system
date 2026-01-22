import os
from langchain_community.vectorstores import FAISS

def create_or_load_faiss(chunks, embeddings, path="vectorstore"):
    if os.path.exists(path):
        vectorstore = FAISS.load_local(
            path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(path)

    return vectorstore
