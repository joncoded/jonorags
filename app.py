# jonorags by @joncoded (aka @jonchius)
# document summarizer and discussion chatbot
# app.py
#
# requirements:
# - input PDF documents and get a variable-sized summary
# - using vector store + retriever + LLM with streamlit UI
# - ability to ask questions and get answers based on PDF contents
#
# enhancements:
# - adjustable summary length and tokens
# - sentiment analysis
# - find named entities

from dotenv import load_dotenv
import os, tempfile, uuid, hashlib
import streamlit as st

from local import *
from pinecone import Pinecone
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

# =========================================================
# DEVELOPER CONFIGURATION
# =========================================================

# language constants
text = l["en"]

# pinecone
use_index = "jonsrags"
use_phost = "https://jonsrags-su5cgy0.svc.aped-4627-b74a.pinecone.io"

# llm
use_myllm = "openai/gpt-oss-20b"
max_tokens = 1000

# ui
app_title = "JONORAGS 📜"
app_tagline = "pdf summarizer and discusser"

# =========================================================
# API KEYS
# =========================================================

key_vecdb = os.getenv("PINECONE_API_KEY", "").strip()
key_myllm = os.getenv("LLM_API_KEY", "").strip()

if not key_myllm:
    st.error("❌ Missing API key (LLM_API_KEY)")
    st.stop()

if not key_vecdb:
    st.error("❌ Missing Pinecone key (PINECONE_API_KEY)")
    st.stop()

# =========================================================
# CONNECTIONS TO VECTOR DB
# =========================================================

pinecone = Pinecone(api_key = key_vecdb, environment="us-west1-gcp")
index = pinecone.Index(
    index_name = use_index,
    host = use_phost
)
namespace = "session"

# =========================================================
# CONNECTIONS TO LLM
# =========================================================

client = OpenAI(
    api_key = key_myllm,
    base_url = "https://api.groq.com/openai/v1"
)

# =========================================================
# STREAMLIT UI HEADER
# =========================================================

st.set_page_config(page_title = app_title, page_icon="📄")

st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        *, h1, h2 { font-family: 'Barlow Semi Condensed' !important; }
        .stExpander > details > summary > span > span:first-child { display: none; }
        /* expander button fixes due to streamlit glitch */
        .stExpander > details > summary > span > div::before { 
            content: "▼"; display: inline-block; margin-right: 8px; 
        }          
        .stExpander > details[open] > summary > span > div::before { 
            content: "▲"; display: inline-block; margin-right: 8px; 
        }
    </style>
    """,
    unsafe_allow_html=True
)

# sticky header hack
def header(content):
    st.markdown(f"""
                <div style="position:fixed; top:60px; left:0; width:100%; background-color:#222; color:#fff; padding:5px; z-index:9999">
                    <div style="display:flex; justify-content:center; align-items:center;">
                        {content}
                    </div>
                </div>""", unsafe_allow_html=True)
header(f"<h1 style=\"font-size:24px\">{app_title}</h1> <div style=\"font-size:12px\">{app_tagline}</div>")
# padding hack
st.write("<br><br>", unsafe_allow_html=True)

# =========================================================
# SESSION STATES
# =========================================================

if "summaries" not in st.session_state:
    st.session_state.summaries = {}

if "doc_hashes" not in st.session_state:
    st.session_state.doc_hashes = set()

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "doc_contents" not in st.session_state:
    st.session_state.doc_contents = {}

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "processing_started" not in st.session_state:
    st.session_state.processing_started = False

# =========================================================
# USER CONFIGURATION
# =========================================================

uploaded_files = st.file_uploader(
    text["upload_instructions"],
    type=["pdf"],
    accept_multiple_files=True,
    key=f"pdf_uploader_{st.session_state.uploader_key}"
)

with st.expander("⚙️ Settings", expanded=True):

    st.text(text["settings_question"])

    col1, col2 = st.columns(2, vertical_alignment="bottom")

    with col1:
        summary_sentences = st.number_input(text["sentences"], min_value=1, max_value=10, value=3)
    with col2:
        summary_bullets = st.number_input(text["bullets"], min_value=1, max_value=5, value=2)

    col1, col2 = st.columns(2, vertical_alignment="top")
    
    with col1:
        label_sentiment = f"""{text["sentiment_analysis"]} \n\n {text["sentiment_analysis_ex"]}"""
        summary_sentiment = st.radio(
            label_sentiment, ["No", "Yes"], horizontal=True
        )
    with col2:
        label_ner = f"""{text["ner"]} \n\n {text["ner_ex"]}"""
        summary_ner = st.radio(
            label_ner, ["No", "Yes"], horizontal=True
        )

# track uploaded files to detect new uploads
if "last_uploaded_files" not in st.session_state:
    st.session_state.last_uploaded_files = []

# check if new files were uploaded
if uploaded_files and uploaded_files != st.session_state.last_uploaded_files:
    st.session_state.summaries = {}
    st.session_state.doc_hashes = set()
    st.session_state.embeddings = None
    st.session_state.processing_started = False
    st.session_state.last_uploaded_files = uploaded_files

# show process button only if files are uploaded and not yet processed
if uploaded_files:
    st.info(f"""📁 {text["files_uploaded"]} : {len(uploaded_files)} ... {text["process_click_when_ready"]}""")
    if st.button(f"🚀 {text["process_docs"]}", type="secondary"):
        st.session_state.processing_started = True
        st.rerun()

# =========================================================
# HELPERS
# =========================================================

# file hash for caching content
def file_hash(uploaded_file):
    uploaded_file.seek(0)
    h = hashlib.sha256(uploaded_file.read()).hexdigest()
    uploaded_file.seek(0)
    return h

# process the files and ingest into vector DB
def ingest_files(uploaded_files):
    documents = []
    files_loaded = 0 

    for uploaded_file in uploaded_files:

        # load PDF file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp.flush()
            loader = PyPDFLoader(tmp.name)
            file_docs = loader.load()

        files_loaded += 1
        st.write(f"⏳ {text["processed_file"]}: {files_loaded}/{len(uploaded_files)}: {uploaded_file.name} ...")            

        for d in file_docs:
            d.metadata["source"] = uploaded_file.name

        # store the combined content with the file hash
        combined_content = "\n\n".join(d.page_content for d in file_docs)
        h = file_hash(uploaded_file)
        st.session_state.doc_contents[h] = combined_content
        
        documents.extend(file_docs)

    st.write(f"📜 {text["pages_processed"]}:", len(documents))

    # text splitting
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)    
    st.write(f"⚙️ {text["splitting_documents"]}")
    docs = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-base")
    
    # chunk loading
    st.write(f"⚙️ {text["loading_chunks"]}")      
    texts = [d.page_content for d in docs]
    vectors_emb = embeddings.embed_documents(texts)

    # vector upsert preparation
    vectors = []
    for i, (doc, vec) in enumerate(zip(docs, vectors_emb)):
        vectors.append({
            "id": f"{doc.metadata.get('source','doc')}-{i}-{uuid.uuid4().hex[:8]}",
            "values": vec,
            "metadata": {
                "source": doc.metadata.get("source", ""),
                "page": doc.metadata.get("page", -1),
                "text": doc.page_content,
            },
        })

    # ensure index is deleted before upserting
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        if "not found" not in str(e).lower():
            st.warning(f"Could not clear namespace: {e}")
    
    # take the vectors and upsert into pinecone
    index.upsert(vectors=vectors, namespace=namespace)

    return embeddings

def summarize_document(content):    
    
    if not content or len(content.strip()) == 0:
        return "⚠️ " + text["no_content_to_summarize"]
    
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_text(content)

        partials = []
        for chunk in chunks:
            try:
                r = client.chat.completions.create(
                    model=use_myllm,      
                    max_tokens=max_tokens,      
                    messages=[
                        {"role": "system", "content": "Summarize this document chunk concisely."},
                        {"role": "user", "content": chunk}
                    ]
                )
                partials.append(r.choices[0].message.content)
            except Exception as e:
                st.warning(f"⚠️ Error summarizing a chunk: {str(e)}")
                continue

        if not partials:
            return "⚠️ " + text["no_content_to_summarize"]

        combined = "\n".join(partials)

        prompt = f"""
        <document>{combined}</document>
        <request>
        Summarize in {summary_sentences} sentences with {summary_bullets} bullet points.
        </request>
        """

        if summary_sentiment == "Yes":
            prompt += "\nProvide a brief sentiment analysis."

        if summary_ner == "Yes":
            prompt += "\nList named entities grouped by type."

        summary_response = client.chat.completions.create(
            model=use_myllm,       
            max_tokens=max_tokens,     
            messages=[
                {"role": "system", "content": "You summarize PDF documents clearly and concisely."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return summary_response.choices[0].message.content
        
    except Exception as e:
        error_msg = f"❌ Error generating summary: {str(e)}"
        st.error(error_msg)
        return error_msg

# =========================================================
# INGEST + SUMMARIZE (RUNS ONCE PER FILE)
# =========================================================

if uploaded_files and st.session_state.processing_started:
    if st.session_state.embeddings is None:
        st.write(f"🔥 {text["processing_docs"]}")
        st.session_state.embeddings = ingest_files(uploaded_files)

    st.header(f"📄 {text["summary_of_documents"]}")

    files_shown = 0

    for uploaded_file in uploaded_files:
        h = file_hash(uploaded_file)

        if h not in st.session_state.doc_hashes:
            
            # get content from session state            
            content = st.session_state.doc_contents.get(h, "")
            
            if content:                
                summary = summarize_document(content)
                st.session_state.summaries[h] = {
                    "name": uploaded_file.name,
                    "summary": summary,
                }
                st.session_state.doc_hashes.add(h)
            else:
                st.warning(f"⚠️ {text["no_content_found"]} ({uploaded_file.name})")
                st.session_state.summaries[h] = {
                    "name": uploaded_file.name,
                    "summary": "⚠️ " + text["no_content_found"],
                }

        # Display the summary
        expanded = True if files_shown == 0 else False
        with st.expander(f"{text["summary"]} ({uploaded_file.name})", expanded=expanded):
            summary_data = st.session_state.summaries.get(h)
            if summary_data and summary_data.get("summary"):
                st.write(summary_data["summary"])
            else:
                st.write(f"⚠️ {text["no_summary_could_be_made"]}")
        files_shown += 1
    

# =========================================================
# CHAT
# =========================================================

# custom CSS for chat messages
st.html(
    """
    <style>
        .stChatMessage {
            border: 1px solid #ccc;
            border-radius: 5px;
        }    
    </style>
    """
)

if st.session_state.embeddings:

    def process_question():
        q = st.session_state.user_question.strip()
        if not q:
            return

        q_emb = st.session_state.embeddings.embed_query(q)
        results = index.query(
            vector=q_emb,
            top_k=10,
            namespace=namespace,
            include_metadata=True,
            include_values=False,
        )

        context = " ".join(
            m["metadata"].get("text", "") for m in results.get("matches", [])
        )

        prompt = f"Context: {context}\n\nQ: {q}\n\n A:"

        r = client.chat.completions.create(
            model=use_myllm,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": "Answer using the provided context."},
                {"role": "user", "content": prompt}
            ]
        )

        st.session_state.qa_history.append({
            "q": q,
            "a": r.choices[0].message.content
        })
        st.session_state.user_question = ""

    if st.session_state.qa_history:       
        st.write(f"### {text["conversation_history"]}")
        for pair in st.session_state.qa_history:
            with st.chat_message("q"):                
                st.write(f"**{pair['q']}**")
            with st.chat_message("a"):
                st.write(pair["a"], unsafe_allow_html=True)                     

    with st.form("chat_form"):
        col1, col2 = st.columns([4,1], vertical_alignment="bottom")
        with col1:
            st.text_input(f"{text["ask_question"]}", key="user_question")
        with col2:
            st.form_submit_button(text["submit_message"], on_click=process_question)

# =========================================================
# CHAT RESET CONTROLS
# =========================================================

st.caption(f"{text["start_over"]}: ")

if st.button(f"🆕 {text["reset_everything"]}"):
    # full reset
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception as e:
        if "not found" not in str(e).lower():
            st.warning(f"{text["could_not_clear_namespace"]}: {e}")
    
    # Increment uploader_key to clear file uploader
    uploader_key = st.session_state.get("uploader_key", 0)
    st.session_state.clear()
    st.session_state.uploader_key = uploader_key + 1
    st.session_state.processing_started = False
    st.rerun()        

if st.button(f"💬 {text["reset_just_the_chat"]}"):
    # keep documents
    for key in [
        "qa_history",
        "user_question",
    ]:
        st.session_state.pop(key, None)

    st.rerun()