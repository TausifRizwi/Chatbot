import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Load and split documents
@st.cache_resource
def load_docs(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(pages)

# Create vectorstore from documents
@st.cache_resource
def create_vectorstore(_docs):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(_docs, embedding_model)

def search_answer(vectorstore, query, top_k=3):
    docs = vectorstore.similarity_search(query, k=top_k)
    return "\n\n".join(doc.page_content for doc in docs)

# Streamlit App
def main():
    st.title("📘 Constitution QA Bot ")
    st.markdown("Ask a question, and get accurate, long answers from the actual PDF !")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    pdf_path = "Indian Constitution At Work.pdf"
    docs = load_docs(pdf_path)
    vectorstore = create_vectorstore(docs)

    user_query = st.text_input(" Ask a question about the Constitution:")

    if user_query:
        with st.spinner("Searching the book..."):
            answer = search_answer(vectorstore, user_query)
            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("Bot", answer))

    # Display chat history
    for speaker, message in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {message}")

if __name__ == "__main__":
    main()
