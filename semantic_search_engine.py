import os
import chromadb
import streamlit as st

from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Semantic Search Engine",
    layout="wide",
)
st.header("Multi-Doc :green[Semantic Search] Engine", text_alignment="center")
st.markdown(
    "A Multi-Doc :green[Semantic Search] Engine understands that both phrases describe the same concept and retrieves the correct document. This is the foundation of modern AI search and the first step toward building RAG systems.",
    text_alignment="center",
)

uploaded_files = st.file_uploader(
    "UPLOAD ALL DOCUMENTS (.txt format)", accept_multiple_files=True, type=["txt"]
)

st.divider()

if "total_docs" not in st.session_state:
    st.session_state.total_docs = 0


if "docs_ready" not in st.session_state:
    st.session_state.docs_ready = False

if "database_ready" not in st.session_state:
    st.session_state.database_ready = False

ids = []
metadata = []
documents = []
select_box_topics = ["All"]

for doc in os.listdir("docs"):
    if os.path.isfile(f"docs/{doc}"):
        with open(f"docs/{doc}", "r") as f:
            with st.spinner("Loading documents.."):
                documents.append(f.read())
                ids.append(doc.title().replace(".Txt", ""))
                select_box_topics.append(doc.title().replace(".Txt", ""))
                metadata.append({"topic": doc.title().replace(".Txt", "")})


def load_docs():
    if uploaded_files:
        os.makedirs("docs", exist_ok=True)

        for file in uploaded_files:
            with open(f"docs/{file.name}", "wb") as f:
                f.write(file.getbuffer())
        st.success("Documents Successfully Uploaded")
        st.session_state.docs_ready = True


@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def show_status():
    col1, col2, col3 = st.columns(3)
    with col1:
        for docs in os.listdir("docs"):
            st.session_state.total_docs = len(os.listdir("docs"))
        st.metric("Documents Loaded", st.session_state.total_docs)
    with col2:
        if st.session_state.docs_ready:
            st.metric("Collection Status:", "🟢 Ready")
        else:
            st.metric("Collection Status:", "🔴 Not Indexed")
    with col3:
        st.metric("Model Loaded:", "all-MiniLM-L6-v2")
    return True


def generate_embeddings(em_model):
    embeddings = em_model.encode(documents)
    return embeddings


def create_database(ids, documents, embeddings, metadata):
    client = chromadb.PersistentClient(path="./embeddings")

    try:
        client.delete_collection("embeddings")
    except Exception:
        pass

    database = client.get_or_create_collection(name="embeddings")

    database.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadata,
    )
    st.session_state.database_ready = True
    return database


def search_query(query, database):
    if topic == "All":
        results = database.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "distances"],
        )
    else:
        results = database.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            include=["documents", "distances"],
            where={"topic": topic},
        )
    if results:
        st.success("Document SuccessfullY Retrieved")
        for id, distance, doc in zip(
            results["ids"][0], results["distances"][0], results["documents"][0]
        ):
            st.subheader(
                f":green[Topic:] {id.upper()} \n :yellow[Distance Vector =] {distance:.4f}"
            )
            st.write(doc)
            st.divider()


if st.button("UPLOAD DOCS TO SYSTEM"):
    with st.spinner("Uploading Docs to system..."):
        if uploaded_files:
            load_docs()
        else:
            st.error("Upload your files first")
            st.session_state.docs_ready = False

if st.session_state.docs_ready:
    show_status()
    em_model = load_model()
    embeddings = generate_embeddings(em_model=em_model)
    database = create_database(ids, documents, embeddings, metadata)
    col1, col2 = st.columns(2)
    with col1:
        topic = st.selectbox("Select your topic:", tuple(select_box_topics))
    with col2:
        n_results = st.slider(
            "Top n document: ",
            min_value=1,
            max_value=st.session_state.total_docs,
            value=1,
        )
    user_query = st.text_area("Enter the test query:")
    if user_query:
        query_embedding = em_model.encode(user_query)
    else:
        st.warning("Enter your query first")
    if st.button("Search"):
        search_query(user_query, database)
else:
    st.warning("Database is not ready yet")
