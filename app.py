import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Multi-PDF RAG with Gemini", layout="wide")
st.title("📄 Multi-PDF RAG System (Gemini AI)")
st.write("Upload multiple PDFs and ask questions based on their content.")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

# -------------------------
# Process PDFs
# -------------------------
if uploaded_files:
    with st.spinner("Processing PDFs..."):

        documents = []

        for uploaded_file in uploaded_files:
            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            documents.extend(loader.load())

            os.remove(tmp_path)

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(documents)

        # Embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector store (in-memory)
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3
        )

        # Prompt
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question strictly using the context below.
            If the answer is not present, say "Not found in the uploaded documents."

            Context:
            {context}

            Question:
            {question}
            """
        )

        # LCEL RAG Chain
        rag_chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )

        st.success("PDFs processed successfully! You can now ask questions.")

        # -------------------------
        # Question Input
        # -------------------------
        query = st.text_input("Ask a question about the uploaded PDFs:")

        if query:
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(query)
                st.markdown("### ✅ Answer")
                st.write(response.content)

else:
    st.info("Please upload one or more PDF files to begin.")
