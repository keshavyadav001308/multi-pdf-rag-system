from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

def create_rag_chain(retriever):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful AI assistant.
        Answer the question strictly using the context below.

        Context:
        {context}

        Question:
        {question}
        """
    )

    # LCEL RAG pipeline
    rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain

