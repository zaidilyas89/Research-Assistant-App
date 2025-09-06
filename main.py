import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from fpdf import FPDF
import io
# DEFAULT_API_KEY = None
DEFAULT_API_KEY = None
if "api_keys" in st.secrets and "GOOGLE_API_KEY" in st.secrets["api_keys"]:
    DEFAULT_API_KEY = st.secrets["api_keys"]["GOOGLE_API_KEY"]


st.set_page_config(page_title="AI Research Paper Assistant", page_icon="📄", layout="wide")






    # use_default = st.checkbox("📂 Use Example Papers", value=False)

# ------------------ HEADER ------------------
# Outer container for the card
with st.container():
    st.markdown(
        """
        <div style="background-color:#eef2fa; padding:30px; border-radius:15px; border:1px solid #cbd6e2">
        <h1 style="text-align:center; color:#2a3f5f;">
        📄 AI-powered Research Paper Assistant.
        </h1>
        <h2 style="text-align:center; color:#2a3f5f;">
        Summarize, find research gaps, and ask questions with RAG + LLMs.
        </h2>
        </div>
        """,
        unsafe_allow_html=True
    )




st.divider()
# Optional info box (with clean multiline formatting)
st.info(
    """
    An interactive AI tool that helps researchers work smarter:

    - **Summarize PDFs** & generate structured literature reviews (Map-Reduce approach).  
    - **Identify research gaps** & answer domain-specific questions with citation-backed responses.  
    - Powered by **Retrieval-Augmented Generation (RAG)**, **FAISS vector search**, and **Google Gemini LLMs**.  
    - Built with **Python, Streamlit, FAISS, and Google Gemini** → showcasing expertise in **LLM integration, RAG pipelines, and applied AI app development**.  
    """
)
st.divider()


st.markdown("<h1 style='font-size:30px;'>Step 1: Select Google Gemini Model</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <p style='font-size:18px;'>
    First, select the Gemini model you want to use for this app.  
    If you wish to start over, simply click the <b>Reset</b> button in the menu below to refresh the app.
    </p>
    """,
    unsafe_allow_html=True
)

with st.expander("📖 Select Model"):
    col1, col2 = st.columns([3, 1])  # Left column wider for model selection

    with col1:
        GEMINI_MODEL = st.selectbox(
            "⚙️ Select Gemini Model:",
            ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-2.5-pro"],
            index=0
        )

    with col2:
        if st.button("🔄 Reset App"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
# ------------------ SIDEBAR ------------------
with st.sidebar:
    st.sidebar.markdown("## 🛠️ Developer's Info")

    st.sidebar.markdown(
        """
        <p style='font-size:20px; font-weight:bold; margin-bottom:5px;'>Zaid Ilyas</p>
        <p style='font-size:16px; margin:0;'>PhD<br>Edith Cowan University</p>
        <p style='font-size:16px; margin:0;'>Perth, Australia</p>
        <p style='font-size:16px; margin-top:5px;'><b>📧 Email:</b> <a href="mailto:z.ilyas@ecu.edu.au">z.ilyas@ecu.edu.au</a></p>
        <p style='font-size:16px; margin:0;'><b>🔗 LinkedIn:</b> <a href="https://www.linkedin.com/in/zaid-ilyas-6389b417" target="_blank">zaid-ilyas</a></p>
        """,
        unsafe_allow_html=True
    )
    EMBEDDING_MODEL = "models/text-embedding-004"
    VECTOR_DB = "FAISS (IndexFlatL2)"
    st.markdown("---")
    st.markdown(f"**🧠 Model:** `{GEMINI_MODEL}`")
    st.markdown(f"**🔤 Embedding:** `Google Embedding Model: {EMBEDDING_MODEL}`")
    st.markdown(f"**📦 Vector DB:** `{VECTOR_DB}`")

# Select default or manual key
st.markdown("<h1 style='font-size:30px;'>Step 2: API Selection</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <p style='font-size:18px;'>
    This app can use a default Google API key. If it doesn’t work, you can quickly obtain a new one via 
    <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color:blue; text-decoration:underline;">
    Google API Settings
    </a>.
    </p>
    """,
    unsafe_allow_html=True
)
key_option = st.selectbox("API Key option:", ["None", "Default Key", "Manual Key"],
        index=0)

api_key = None

if key_option == "Manual Key":
    api_key_input = st.text_input("Enter your API Key:")
    if api_key_input:
        api_key = api_key_input
        genai.configure(api_key=api_key)
        st.success("Manual API Key registered ✅")
elif key_option == "Default Key":
    api_key = DEFAULT_API_KEY
    genai.configure(api_key=api_key)
    st.success("Default API Key registered ✅")


def load_pdfs_with_metadata(files, is_uploaded=False):
    docs = []
    for f in files:
        if is_uploaded:
            # Case 1: Streamlit uploaded file (in-memory object)
            reader = PdfReader(f)
            paper_name = f.name
        else:
            # Case 2: File path from directory
            reader = PdfReader(open(f, "rb"))
            paper_name = os.path.basename(f)

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            docs.append({
                "paper": paper_name,
                "page": page_num + 1,
                "text": text.strip()
            })
    return docs


def load_multiple_pdfs_from_paths(paths):
    return [(os.path.basename(p), " ".join([page.extract_text() or "" for page in PdfReader(p).pages])) for p in paths]

# ------------------ STEP 2: CHUNKING ------------------
def chunk_per_page(doc, chunk_size=1200, overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(doc["text"])
    return [
        {
            "paper": doc["paper"],
            "page": doc["page"],
            "chunk_id": i,
            "chunk_text": chunk,
            "source": f"{doc['paper']} (Page {doc['page']})"
        }
        for i, chunk in enumerate(chunks)
    ]

def chunk_with_page_overlap(docs, chunk_size=5000, overlap=500):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    combined_chunks = []
    for i, doc in enumerate(docs):
        text = doc["text"]
        if i + 1 < len(docs):  # add prefix from next page
            text += "\n\n" + docs[i + 1]["text"][:500]
        sub_chunks = splitter.split_text(text)
        for j, chunk in enumerate(sub_chunks):
            combined_chunks.append({
                "paper": doc["paper"],
                "page": f"{doc['page']}-{doc['page']+1}" if i+1 < len(docs) else str(doc['page']),
                "chunk_id": j,
                "chunk_text": chunk,
                "source": f"{doc['paper']} (Pages {doc['page']}–{doc['page']+1})" if i+1 < len(docs) else f"{doc['paper']} (Page {doc['page']})"
            })
    return combined_chunks


def prepare_chunks(docs, mode="per_page"):
    all_chunks = []
    if mode == "per_page":
        for d in docs:
            all_chunks.extend(chunk_per_page(d))
    elif mode == "page_overlap":
        all_chunks = chunk_with_page_overlap(docs)
    return all_chunks


# ------------------ STEP 3: EMBEDDING & FAISS ------------------
def embed_texts(texts):
    return np.array([
        genai.embed_content(model=EMBEDDING_MODEL, content=t)["embedding"] for t in texts
    ], dtype=np.float32)

def create_faiss_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve_top_chunks(query, chunks, index, top_k=5):
    q_emb = np.array([genai.embed_content(model=EMBEDDING_MODEL, content=query)["embedding"]], dtype=np.float32)
    distances, indices = index.search(q_emb, top_k)
    return [chunks[i] for i in indices[0]]

# ------------------ STEP 4: SUMMARIZATION / QA ------------------

def summarize_theme(theme_chunks, mode="full_review", query=None):
    if not theme_chunks:
        return "No chunks to summarize."

    # Handle both dict chunks and (text, label) tuples from map-reduce
    if isinstance(theme_chunks[0], dict):
        context = "\n\n".join(
            [f"Source: {c['source']}\n{c['chunk_text']}" for c in theme_chunks]
        )
    elif isinstance(theme_chunks[0], (tuple, list)):
        context = "\n\n".join(
            [f"Source: {label}\n{text}" for text, label in theme_chunks]
        )
    else:
        raise ValueError("Unsupported theme_chunks format.")

    if mode == "full_review":
        prompt = (
            "You are an expert researcher. Based on the following excerpts from academic papers:\n"
            f"{context}\n\n"
            "Generate a detailed structured literature review with proposed method and its limitations "
            "with the following sections keeping date of publication of each method in consideration:\n"
            "1. Introduction\n2. Related Work\n3. Results\n4. Research Gaps\n5. Citations\n"
            "Write concisely but include specific findings and citations if present. "
            "Re-number the citations (remove duplicates based on title of paper) from all papers and "
            "include corresponding numbering in the generated text.\n"
            "Format each citation on its own line.\n"
            "If you don't know something, state that you don't know."
        )
    
    else:
        return "Invalid mode."

    model = genai.GenerativeModel(GEMINI_MODEL)

    
    response = model.generate_content(prompt)
    return extract_gemini_text(response)


def map_per_paper(chunks):
    """
    Summarize all chunks of each paper at once (no batching).
    Returns a dict: {paper_name: summary_text}
    """
    papers = {}
    # Group chunks by paper
    paper_groups = {}
    for c in chunks:
        paper_groups.setdefault(c["paper"], []).append(c)

    for paper, paper_chunks in paper_groups.items():
        # Combine all chunks for this paper into (text, label) tuples
        batch_for_summary = [(c["chunk_text"], f"Page {c['page']}") for c in paper_chunks]
        
        # Single summarize call per paper
        paper_summary = summarize_theme(batch_for_summary, mode="full_review")
        papers[paper] = paper_summary
        st.success(f"Done summarizing paper: {paper}")

    return papers



def reduce_across_papers(per_paper_summaries):
    """
    Generate global literature review combining per-paper summaries.
    """
    combined = [(summary, paper) for paper, summary in per_paper_summaries.items()]
    global_summary = summarize_theme(combined, mode="full_review")
    return global_summary

def extract_gemini_text(response):
    """Safely extract text from a Gemini response object."""
    try:
        # Easiest case
        if getattr(response, "text", None):
            return response.text.strip()

        # Check candidates
        if hasattr(response, "candidates") and response.candidates:
            cand = response.candidates[0]

            # Blocked response
            if hasattr(response, "prompt_feedback") and getattr(response.prompt_feedback, "block_reason", None):
                return f"⚠️ Blocked: {response.prompt_feedback.block_reason}"

            # Try parts
            if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                texts = [p.text for p in cand.content.parts if hasattr(p, "text")]
                if texts:
                    return "\n".join(texts).strip()

        return "⚠️ No valid text returned (empty or blocked)."

    except Exception as e:
        return f"⚠️ Error extracting Gemini response: {e}"


class ResearchChatbot:
    def __init__(self, chunks, index, literature_summary=None, model=GEMINI_MODEL):
        """
        chunks: list of dicts [{source, chunk_text}]
        index: FAISS index
        literature_summary: dict {paper_name: summary} or string summarizing all papers
        model: Gemini model name
        """
        self.chunks = chunks
        self.index = index
        self.literature_summary = literature_summary or "No summary available."
        self.model = genai.GenerativeModel(model)
        self.history = []  # stores [{q, a, sources}]

    def retrieve(self, query, top_k=5, bias=None):
        """Retrieve top_k most relevant chunks using vector search.
        Optionally bias toward chunks with section keywords (methodology, dataset, etc.)."""
        q_emb = np.array(
            [genai.embed_content(model=EMBEDDING_MODEL, content=query)["embedding"]],
            dtype=np.float32,
        )
        distances, indices = self.index.search(q_emb, top_k * 2)  # retrieve more initially
        candidates = [self.chunks[i] for i in indices[0]]

        if bias:
            filtered = [
                c for c in candidates
                if any(b.lower() in c["chunk_text"].lower() for b in bias)
            ]
            if filtered:
                return filtered[:top_k]
        return candidates[:top_k]

    def analyze_query(self, query):
        """Detect if the question targets methodology, datasets, research gaps, etc."""
        query_l = query.lower()
        if "methodology" in query_l or "methods" in query_l or "approach" in query_l:
            return ["methodology", "methods", "approach"]
        elif "dataset" in query_l or "data set" in query_l or "cohort" in query_l:
            return ["dataset", "data", "cohort"]
        elif "results" in query_l or "findings" in query_l or "evaluation" in query_l:
            return ["results", "experiments", "evaluation"]
        elif "conclusion" in query_l or "discussion" in query_l:
            return ["conclusion", "summary", "discussion"]
        elif "research gap" in query_l or "future work" in query_l or "limitation" in query_l:
            return ["gap", "limitation", "future"]
        return None  # no bias

    def ask(self, query):
        """Answer a query using retrieved context + literature review summary + history (RAG)."""
        # Step 1: detect if query is section-specific
        bias = self.analyze_query(query)

        # Step 2: retrieve relevant chunks
        retrieved = self.retrieve(query, top_k=3, bias=bias)

        # Step 3: prepare retrieved context text
        context_text = "\n\n".join(
            [f"Source: {c['source']}\n{c['chunk_text']}" for c in retrieved]
        )

        # Step 4: prepare literature review summary text
        if isinstance(self.literature_summary, dict):
            # Combine all paper summaries into one block
            lit_summary_text = "\n\n".join(
                [f"Source: {paper} (Literature Review Summary)\n{summary}"
                 for paper, summary in self.literature_summary.items()]
            )
        else:
            lit_summary_text = f"Source: Literature Review Summary\n{self.literature_summary}"

        # Step 5: build conversation history (last 3 turns)
        history_text = ""
        for turn in self.history[-3:]:
            history_text += f"Q: {turn['q']}\nA: {turn['a']}\n\n"

        # Step 6: construct prompt
        prompt = (
            "You are an expert research assistant. "
            "Use BOTH the retrieved context and the literature review summary to answer. "
            "If the question is high-level (datasets, methodology trends, research gaps), "
            "combine details from multiple papers. "
            "Be precise, concise, and always cite the sources explicitly.\n\n"
            f"{history_text}"
            f"Literature Review Summary:\n{lit_summary_text}\n\n"
            f"Retrieved Context:\n{context_text}\n\n"
            f"Question: {query}\nAnswer:"
        )

        # Step 7: query Gemini
        try:
            response = self.model.generate_content(prompt)
            answer = extract_gemini_text(response)
        except Exception as e:
            answer = f"Error from Gemini: {e}"

        # Step 8: deduplicate sources for consistent bottom listing
        sources = []
        seen = set()
        for c in retrieved:
            src = c["source"]
            if src not in seen:
                sources.append(src)
                seen.add(src)
        if isinstance(self.literature_summary, dict):
            for paper in self.literature_summary.keys():
                if paper not in seen:
                    sources.append(f"{paper} (Literature Review Summary)")
                    seen.add(paper)
        else:
            if "Literature Review Summary" not in seen:
                sources.append("Literature Review Summary")

        # Step 9: save turn in history
        self.history.append({"q": query, "a": answer, "sources": sources})

        return answer, sources


# ------------------ FILE INPUT ------------------
st.divider()
# Select default or manual key
st.markdown("<h1 style='font-size:30px;'>Step 3: Papers Selection</h1>", unsafe_allow_html=True)
st.markdown(
    """
    <p style='font-size:18px;'>
    <b>Note:</b> You can either select from the example papers included with this app or upload your own. 
    This app is for demo purposes, so avoid using large documents as it may quickly exhaust the API quota.
    </p>
    """,
    unsafe_allow_html=True
)
option = st.selectbox(
    "Select PDF source:",
    ("None", "Use Example Papers", "Upload Your Own Papers"),
    index=0  # 0 = "None", 1 = "Use Example Papers", 2 = "Upload Your Own Papers"
)

pdfs = []

if option == 'Use Example Papers':
    default_folder = "default_papers"
    default_files = [os.path.join(default_folder, f) for f in os.listdir(default_folder) if f.endswith(".pdf")]
    pdfs = load_pdfs_with_metadata(default_files, is_uploaded=False)
    
elif option == "Upload Your Own Papers":
    uploaded_files = st.file_uploader("Upload PDF papers", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("📑 Uploading PDFs..."):
            pdfs = load_pdfs_with_metadata(uploaded_files, is_uploaded=True)
    else:
        pdfs = []
paper_titles = []
for paper in pdfs:
    paper_titles.append(paper['paper'])

paper_lst = list(set(paper_titles))
for p in paper_lst:
    st.write(f"- {p}")

# ------------------ MAIN WORKFLOW ------------------

if pdfs:   # ✅ only process if PDFs exist
    if "qa_chunks" not in st.session_state:
        with st.spinner("📑 Creating Chunks..."):
            lr_chunks = prepare_chunks(pdfs, mode="page_overlap")  # literature review
            qa_chunks = prepare_chunks(pdfs, mode="per_page")      # Q&A
            st.success("Chunks Created Successfully!")
        with st.spinner("📑 Generating Embeddings..."):
            qa_embeddings = embed_texts([c["chunk_text"] for c in qa_chunks])
            st.success("Embeddings Created Successfully!")
        with st.spinner("📑 Generating VectorDB Indices..."):
            qa_index = create_faiss_index(qa_embeddings)
            st.success('FAISS indices created Successfully!')

        with st.spinner("📑 Performing 'Map' Process to summarize papers individually. Please wait..."):
            per_paper_summaries = map_per_paper(lr_chunks)
            st.success('Individual Paper Summarization Done Successfully!!')
        with st.spinner("📑 Performing 'Reduce' Process to generate final summary. Please wait..."):
            literature_review = reduce_across_papers(per_paper_summaries)
            st.success("✅ Completed!")

            # store in session state
            st.session_state.qa_chunks = qa_chunks
            st.session_state.qa_index = qa_index
            st.session_state.literature_review = literature_review

    # ✅ reuse stored objects instead of recomputing
    qa_chunks = st.session_state.qa_chunks
    qa_index = st.session_state.qa_index
    literature_review = st.session_state.literature_review

    with st.expander("📑 Literature Review Summary"):
        st.write(literature_review or "⚠️ No valid text returned.")
        # if literature_review:
        #     # Create PDF in memory
        #     pdf_buffer = io.BytesIO()
        #     pdf = FPDF()
        #     pdf.add_page()
        #     pdf.set_auto_page_break(auto=True, margin=15)
        #     pdf.set_font("Arial", size=12)
        #     pdf.multi_cell(0, 10, literature_review)
        #     pdf.output(pdf_buffer)
        #     pdf_buffer.seek(0)  # Reset pointer

        #     # Provide download button
        #     st.download_button(
        #         label="📥 Download PDF",
        #         data=pdf_buffer,
        #         file_name="literature_review.pdf",
        #         mime="application/pdf"
        #     )
# Step 3 Heading
    st.markdown(
        "<h2 style='font-size:32px; text-align:left;'>Step 4: Ask Questions About the Selected Papers</h2>", 
        unsafe_allow_html=True
    )

    # Step 3 description with example questions
    st.markdown(
        """
        <div style='font-size:18px; line-height:1.6;'>
            <p><b>Some example questions may include:</b></p>
            <ul>
                <li>What datasets have been used in these studies? Please summarize in bullet points.</li>
                <li>Which paper proposed the hybrid transformer and CNN-based approach to tackle the problem?</li>
                <li>What are the possible future directions of work in this field?</li>
            </ul>
            <p style='margin-top:10px;'><b>Note:</b> This conversation-based QA section has a memory feature, so it keeps track of previous responses. 
            Additionally, answers include the approximate page number of the document where specific information is located, e.g., details of the datasets.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "bot" not in st.session_state:
            st.session_state.bot = ResearchChatbot(qa_chunks, qa_index, literature_review)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask a research question..."):
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Get bot response
        answer, sources = st.session_state.bot.ask(user_input)
        response_text = f"{answer}\n\n🔎 **Sources:**\n" + "\n".join([f"- {s}" for s in sources])

        st.session_state.messages.append({"role": "assistant", "content": response_text})
        with st.chat_message("assistant"):
            st.markdown(response_text)

else:
    st.warning("👆 Please select **Use Example Papers** or upload PDFs to continue.")


