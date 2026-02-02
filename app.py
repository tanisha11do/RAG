import streamlit as st
from rag import load_pdfs, create_vector_store, get_answer, load_wikipedia

st.set_page_config(page_title="TeaRAG App")
st.title("RAG Model Benefit Demonstration")

# User selects the source
source = st.selectbox("Select Knowledge Source:", ("PDFs", "Wikipedia", "PDFs + Wikipedia"))

# Upload PDFs only if PDFs option is selected
uploaded_files = None
if source in ("PDFs", "PDFs + Wikipedia"):
    uploaded_files = st.file_uploader(
        "Upload your PDF documents", type="pdf", accept_multiple_files=True
    )

# User query
query = st.text_input("Enter your question:")

if query:
    pdf_chunks = None
    wiki_docs = None

    # Load PDFs if uploaded
    if uploaded_files:
        pdf_chunks = load_pdfs(uploaded_files)

    # Load Wikipedia docs if chosen
    if source in ("Wikipedia", "PDFs + Wikipedia"):
        wiki_docs = load_wikipedia(query)

    # Create vector store from PDFs + Wikipedia
    vector_store = create_vector_store(pdf_chunks, wiki_docs)

    # Get answer
    answer = get_answer(vector_store, query)

    st.write("**Answer:**", answer)
