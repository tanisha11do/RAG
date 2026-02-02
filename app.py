import streamlit as st
from rag import load_pdfs, create_vector_store, get_answer, load_wikipedia

# For Wikipedia
import wikipedia

st.set_page_config(page_title="TeaRAG App")

st.title("RAG Model Benefit Demonstration")

# User selects the source
options = st.selectbox("Select Query Option:", ("Wikipedia", "Research Paper"))

# ----------------- Wikipedia Option -----------------
if options == "Wikipedia":
    query = st.text_input("Enter your Wikipedia query:")

    if query:
        try:
            # Search Wikipedia for the query
            search_results = load_wikipedia.search(query)

            if not search_results:
                st.write("No Wikipedia results found.")
            else:
                # Get the first result's summary
                summary = load_wikipedia.summary(search_results[0], sentences=5)
                st.write("**Answer:**")
                st.write(summary)

        except Exception as e:
            st.write("Error fetching Wikipedia data:", e)

# ----------------- Research Paper Option -----------------
elif options == "Research Paper":
    st.write("Upload PDFs and ask questions!")

    uploaded_files = st.file_uploader(
        "Upload your PDF documents", type="pdf", accept_multiple_files=True
    )

    if uploaded_files:
        # Load and split PDFs
        docs_split = load_pdfs(uploaded_files)
        st.success(f"{len(docs_split)} text chunks created!")

        # Create vector store
        vector_store = create_vector_store(docs_split)
        st.success("Vector store created!")

        # Query input
        query = st.text_input("Ask a question about your documents:")

        if query:
            answer = get_answer(vector_store, query)
            st.write("**Answer:**", answer)
