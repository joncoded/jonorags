# jonorags by @joncoded (aka @jonchius)
# app.py
# requirements:
# - input PDF documents and get a variable-sized summary
# - using vector store + retriever + LLM with streamlit UI
# - ability to ask questions and get answers based on PDF contents
# goals:
# - document summary of single document
# - document summary of multiple documents
# enhancements:
# - adjustable summary length
# - sentiment analysis
# - find named entities

# === API KEYS (stored in environment variables)
from dotenv import load_dotenv
import os

load_dotenv()
key_model = os.getenv("GROQ_API_KEY").strip()
key_vecdb = os.getenv("PINECONE_API_KEY").strip()

# === VECTOR DB CONNECTION
from pinecone import Pinecone
pinecone = Pinecone(api_key=key_vecdb, environment="us-west1-gcp")
index_name = "jonsrags"
host = "https://jonsrags-su5cgy0.svc.aped-4627-b74a.pinecone.io"
index = pinecone.Index(index_name=index_name, host=host)

# === LLM CONNECTION 
from openai import OpenAI
client = OpenAI(
    api_key=key_model,
    base_url="https://api.groq.com/openai/v1"
)
use_model = "groq/compound-mini"

# === STREAMLIT UI SETUP
import streamlit as st
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        *, h1, h2 { font-family: 'Barlow Semi Condensed' !important; }
        .stExpander > details > summary > span > span:first-child { display: none; }
        /* expander button fix due to streamlit glitch */
        .stExpander > details > summary > span > div::before { 
            content: "⬇️"; display: inline-block; margin-right: 8px; 
        }          
        .stExpander > details[open] > summary > span > div::before { 
            content: "⬆️"; display: inline-block; margin-right: 8px; 
        }
    </style>
    """,
    unsafe_allow_html=True
)
st.set_page_config(page_title="jonorags (by @joncoded)", page_icon="📄")
st.title("JONORAGS")
st.header("📄 PDF summarizer and inquiry chatbot")

with st.expander("Settings", expanded=True, icon="⚙️"):

    st.text("Summarize the following PDFs in this number of...")

    col1, col2 = st.columns([1,1], vertical_alignment="bottom")
    
    with col1:
        summary_sentences = st.number_input(
            "Sentences:",
            min_value = 1,
            max_value = 10,
            value = 3,
            step = 1
        )

    with col2:

        summary_bullets = st.number_input(
            "Bullets:",
            min_value = 1,
            max_value = 5,
            value = 2,
            step = 1
        )

    summary_sentiment = st.radio(
        "Include sentiment analysis?",
        options = ["No", "Yes"],
        index = 0,
        horizontal = True
    )

    summary_ner = st.radio(
        "Include list of persons' names found in document (named entity recognition)?",
        options = ["No", "Yes"],
        index = 0,
        horizontal = True
    )

uploaded_files = st.file_uploader("Upload 1+ PDF files to summarize", type=["pdf"], accept_multiple_files=True)

# ===== HANDLE UPLOAD(S)

if uploaded_files:

    # ===== FILE PROCESSOR

    # text loading into tempfile(s)
    st.write(f"📁 Uploaded {len(uploaded_files)} files: processing...")
    from langchain_community.document_loaders import PyPDFLoader
    import tempfile

    # documents will refer to parts of the uploaded file(s)
    documents = []
    files_loaded = 0

    # load each uploaded PDF file
    for uploaded_file in uploaded_files:
    
        # write uploaded file bytes to a temporary file so PyPDFLoader can read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp.flush()
            tmp_path = tmp.name

        # load the PDF file into langchain Document objects
        loader = PyPDFLoader(tmp_path)

        files_loaded += 1
        
        st.write(f"⏳ Processed file {files_loaded}/{len(uploaded_files)}: {uploaded_file.name} ...")
        
        # returns langchain Document objects (have page_content)
        file_docs = loader.load()

        # set source metadata and attach a combined page_content back to the UploadedFile
        for file_doc in file_docs:
            file_doc.metadata['source'] = uploaded_file.name
        
        uploaded_file.page_content = "\n\n".join([d.page_content for d in file_docs])

        documents.extend(file_docs)

    st.write("📜 Pages processed:", len(documents))

    # text splittings

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 150
    )

    docs = text_splitter.split_documents(documents)
    st.write("⚙️ Splitting document into chunks...")

    # text embeddings

    from langchain_community.embeddings import SentenceTransformerEmbeddings

    embeddings = SentenceTransformerEmbeddings(
        model_name = "intfloat/e5-base"
    )
    st.write("⚙️ Loading chunks to database... this may take a while so thanks for your patience! 🙂")

    # batch embed document chunks (use embed_documents for document embeddings)

    texts = [doc.page_content for doc in docs]
    embeddings_list = embeddings.embed_documents(texts)

    vectors = []
    import uuid
    for i, (doc, vec) in enumerate(zip(docs, embeddings_list)):
        vectors.append({
            # yes this part was vibe coded tbh
            "id": f"{doc.metadata.get('source','doc')}-{i}-{uuid.uuid4().hex[:8]}",
            "values": vec,
            "metadata": {
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", -1),
                "text": doc.page_content
            }
        })
    
    namespace = '__default__'

    # clear the namespace of the vector database to start a new session
    # (didn't want to have previous documents interfere with new ones)
    try:        
        index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        st.warning(f"Could not clear namespace: {e}")
        if "not found" not in str(e).lower():
            st.warning(f"Could not clear namespace: {e}")

    # insert the vectors to the vector database
    upsert_resp = index.upsert(vectors=vectors, namespace=namespace)   

    # === DOCUMENT SUMMARIZER 

    st.header("📄 Summary of document(s)")
    
    # summarize each uploaded file according to user settings
    for uploaded_file in uploaded_files:

        with st.expander(f"Summary of {uploaded_file.name}", expanded=False):

            sentiment_subprompt = ""  
            summary_prompt = f"""
                <document>
                {uploaded_file.page_content}
                </document>
                
                <request>
                Summarize this document in {summary_sentences} sentences, with {summary_bullets} bullet points, each no more than 15 words, about its main theme.
                </request>

                
            """

            # subprompt for sentiment analysis
            if summary_sentiment == "Yes":

                sentiment_subprompt += f"""

                <additional_request>
                    Additionally, provide a brief sentiment analysis of the document in a short sentence, focusing on the overall tone and emotion of the main topic.
                </additional_request>   

                <append_to_format>
                **Sentiment Analysis**:
                A brief sentence about the overall tone and emotion of the document.
                </append_to_format>

                """

            # subpromopt for named entity recognition 
            if summary_ner == "Yes":

                ner_subprompt = f"""

                <additional_request>
                    Additionally, provide a list of named entities found in the document, categorized by type (e.g., persons, organizations, locations).
                </additional_request>

                <append_to_format>
                **Named Entities**:
                - Persons: list of persons, do not include this line if no persons were found
                - Organizations: list of organizations, do not include this line if no organizations were found
                - Locations: list of locations, do not include this line if no locations were found
                - If no named entities were found, state "No named entities found!"
                </addition_to_format>
                """

            # combine subprompts into main prompt
            summary_prompt += f"""

                <format>
                Summary of {summary_sentences} sentence outlining the main themes of the document, with a clause that flows smoothly into the bullet points:
                * Bullet points with themes related to the summary sentence

                {sentiment_subprompt if summary_sentiment == "Yes" else ""}

                {ner_subprompt if summary_ner == "Yes" else ""}
                
                </format>
            """
            
            # get final response!
            summary_response = client.chat.completions.create(
                model=use_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant designed to summarize PDF documents."},
                    {"role": "user", "content": summary_prompt}
                ]
            )
            
            summary_answer = summary_response.choices[0].message.content
            
            st.write(summary_answer)
                
    
    st.write("✅ Document(s) loaded ... if you want, we can discuss about them!")

    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []

    if 'user_question' not in st.session_state:
        st.session_state.user_question = ""

    # ===== DOCUMENT DISCUSSION WITH CHATBOT

    def process_question():
        
        q = st.session_state.get("user_question", "").strip()
        if not q:
            return

        # retrieve relevant documents (same logic as before)
        query_embedding = embeddings.embed_query(q)
        results = index.query(vector=query_embedding, top_k=10, namespace=namespace, include_metadata=True, include_values=False)

        raw_matches = results.get("matches", []) if isinstance(results, dict) else getattr(results, "matches", [])
        normalized = []
        for m in raw_matches:
            if isinstance(m, dict):
                meta = m.get("metadata", {}) or {}
            else:
                meta = getattr(m, "metadata", {}) or {}
            normalized.append(meta)

        if not normalized:
            answer = "No relevant document chunks found. Make sure you processed and upserted documents."
        else:
            context = " ".join([md.get("text", "") for md in normalized]).strip()
            if not context:
                answer = "Found matches but no stored text in metadata. Ensure you upserted 'text' in vector metadata and queried with include_metadata=True."
            else:
                prompt = f"Context: {context}\n\nQuestion: {q}\n\nAnswer:"
                response = client.chat.completions.create(
                    model=use_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant designed to summarize PDF documents and compare them if there are more than one."},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.choices[0].message.content

        st.session_state.qa_history.append({"q": q, "a": answer})
        st.session_state.user_question = ""  # clear the widget (allowed inside callback)    

    if len(st.session_state.qa_history) > 0:
        st.write("### Conversation history:")

    if 'qa_history' not in st.session_state:
        st.session_state.qa_history = []

    for pair in st.session_state.qa_history:
        st.write("Q:", pair["q"])
        st.write("A:", pair["a"])
        st.write("---")

    with st.form(key="question_form"):
        col1, col2 = st.columns([4,1], vertical_alignment="bottom")
        with col1:
            st.text_input("Ask a question about the uploaded documents:", key="user_question")
        with col2:
            st.form_submit_button("Send", on_click=process_question)