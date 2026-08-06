import hashlib
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

st.set_page_config(page_title="PDF Assistant", page_icon="📄", layout="wide")


def reset_document_state() -> None:
    """Clear responses when the user selects a different document."""
    st.session_state.vectorstore = None
    st.session_state.document_name = None
    st.session_state.document_hash = None
    st.session_state.messages = []


for key, value in {
    "vectorstore": None,
    "document_name": None,
    "document_hash": None,
    "messages": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


st.title("📄 PDF Assistant")
st.caption("Upload a PDF, prepare it once, and ask questions based on its contents.")
st.divider()

with st.sidebar:
    st.header("PDF Assistant")
    st.markdown("1. Upload a PDF  \n2. Prepare the document  \n3. Ask questions")
    st.divider()
    st.caption("Your uploaded document is used for this browser session only.")
    if st.session_state.vectorstore is not None:
        if st.button("Clear current document", use_container_width=True):
            reset_document_state()
            st.rerun()

left_column, right_column = st.columns([1.05, 1.95], gap="large")

with left_column:
    st.subheader("1. Upload a PDF")
    st.caption("Select the document you want to search.")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="PDF files only. You can replace the current document at any time.",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        if st.session_state.document_hash not in (None, file_hash):
            reset_document_state()

        st.success(f"Ready to prepare: {uploaded_file.name}")
        file_size_mb = len(file_bytes) / (1024 * 1024)
        metric_one, metric_two = st.columns(2)
        metric_one.metric("File size", f"{file_size_mb:.1f} MB")
        metric_two.metric("Format", "PDF")

        prepare_document = st.button(
            "Prepare document",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.document_hash == file_hash,
        )

        if prepare_document:
            temporary_path = None
            try:
                with st.status("Preparing your document…", expanded=True) as status:
                    st.write("Reading PDF pages…")
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                        temp_file.write(file_bytes)
                        temporary_path = temp_file.name

                    documents = PyPDFLoader(temporary_path).load()
                    if not documents:
                        raise ValueError("No readable text was found in this PDF.")

                    st.write("Breaking content into searchable sections…")
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = splitter.split_documents(documents)

                    st.write("Creating the search index…")
                    embeddings = MistralAIEmbeddings(model='mistral-embed')
                    st.session_state.vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                    )
                    st.session_state.document_name = uploaded_file.name
                    st.session_state.document_hash = file_hash
                    st.session_state.messages = []
                    status.update(label="Document is ready!", state="complete", expanded=False)
                st.rerun()
            except Exception as error:
                st.session_state.vectorstore = None
                st.error(f"We couldn't prepare this PDF: {error}")
            finally:
                if temporary_path and os.path.exists(temporary_path):
                    os.remove(temporary_path)
    else:
        st.info("Upload a PDF to begin.")

with right_column:
    st.subheader("2. Ask questions")

    if st.session_state.vectorstore is None:
        st.info(
            "Upload a PDF and select **Prepare document**. Your answers will appear here."
        )
    else:
        st.success(f"Chatting with **{st.session_state.document_name}**")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "assistant" and message.get("sources"):
                    st.caption(f"Sources: {message['sources']}")

        query = st.chat_input("Ask a question about this PDF…")
        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Searching the document…"):
                    try:
                        retriever = st.session_state.vectorstore.as_retriever(
                            search_type="mmr",
                            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
                        )
                        source_documents = retriever.invoke(query)
                        context = "\n\n".join(doc.page_content for doc in source_documents)

                        prompt = ChatPromptTemplate.from_messages(
                            [
                                (
                                    "system",
                                    """You are a helpful AI assistant. Use only the provided context to answer.
If the answer is not present in the context, say exactly: I could not find the answer in the document.""",
                                ),
                                ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
                            ]
                        )
                        response = ChatMistralAI(model="mistral-small-2506").invoke(
                            prompt.invoke({"context": context, "question": query})
                        )
                        answer = response.content
                        page_numbers = sorted(
                            {
                                str(document.metadata["page"] + 1)
                                for document in source_documents
                                if "page" in document.metadata
                            },
                            key=int,
                        )
                        sources = f"Pages {', '.join(page_numbers)}" if page_numbers else "Relevant sections"
                        st.markdown(answer)
                        st.caption(f"Sources: {sources}")
                        st.session_state.messages.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )
                    except Exception as error:
                        st.error(f"We couldn't answer that question: {error}")
