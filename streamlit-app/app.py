import os
import json
import streamlit as st
from openai import OpenAI
import openai
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

load_dotenv()

# Use your raw JSON file path here.
raw_json = r"C:\Users\Dee\root\Projects\dev\unpatternedAi\unstructureparser\data\raw\who_extracted_pdf.json"

# -- Configuration --
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or "your-openai-api-key"
openai.api_key = OPENAI_API_KEY

# Use your provided OpenAI client initialization.
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# -- ChromaDB Setup --
embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-ada-002")
client = chromadb.PersistentClient(path="./vector_db", settings=Settings())
collection_name = "who_publications"
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_function
)


# -- Utility Functions --

@st.cache_resource(show_spinner=False)
def load_extracted_json(json_path):
    """Loads the JSON file containing extracted PDF text data."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def store_documents(documents):
    """
    Given a list of document dicts (with keys 'source', 'page', and 'text'),
    store them in ChromaDB, including both text and embeddings.
    """
    for item in documents:
        doc_id = f"{item['source']}_page_{item['page']}"
        # Add each document to the collection.
        collection.add(
            ids=[doc_id],
            documents=[item["text"]],
            metadatas=[{"page": item["page"], "source": item["source"]}],
        )
    return f"Stored {len(documents)} documents in ChromaDB."


def retrieve_similar_documents(query, top_k=5):
    """
    Retrieve similar documents based on the query from ChromaDB.
    """
    results = collection.query(query_texts=[query], n_results=top_k)

    if not results or "documents" not in results or not results["documents"]:
        return []

    matched_texts = results["documents"][0] if results["documents"][0] else []
    return [str(doc) for doc in matched_texts]


def generate_answer(query, retrieved_context):
    """
    Combine the query and the retrieved context to generate an answer using OpenAI's Chat API.
    The generated answer should focus on global health from a Monitoring, Learning and Evaluation (MLE)
    perspective. It should discuss the impact, costs, mortality, and challenges of global health trends.
    """
    context = "\n\n".join(retrieved_context)
    prompt = (
        "You are an expert in global health, specializing in Monitoring, Learning, "
        "and Evaluation (MLE). Given the following context extracted from WHO reports:\n"
        f"{context}\n\n"
        "Answer the following question in a detailed and analytical manner. "
        "Add include number and statistics from WHO publication documents"
        "Your answer should assess global health trends, discussing the current impact, costs, demographic, mortality, "
        "and ongoing challenges. "
        f"Question: {query}\n"
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # or use "gpt-4" if available
            messages=[
                {"role": "system", "content": "You are an assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Sorry, I couldn't generate an answer."


# -- Streamlit App Interface --

st.set_page_config(page_title="Global Health Trends (MLE)", layout="wide")
st.title("Global Health Trends Analysis (MLE)")

# Optionally, load and store documents (only do once)
# Adjust the json_path to where your JSON file is stored.
#json_path = os.path.join(os.getcwd(), "data", "raw", "who_extracted_pdf.json")
if st.button("Load and Store Documents"):
    try:
        parsed_json = load_extracted_json(raw_json)
        result = store_documents(parsed_json)
        st.success(result)
    except Exception as ex:
        st.error(f"Error: {ex}")

# Text input for the user's query
query = st.text_input("Enter your question about global health:", value="What are the global trends in health?")

if st.button("Get Analysis"):
    if query:
        with st.spinner("Retrieving relevant documents and generating answer..."):
            retrieved_docs = retrieve_similar_documents(query, top_k=5)
            answer = generate_answer(query, retrieved_docs)
        st.subheader("-> Analysis <-")
        st.write(answer)
    else:
        st.warning("Please enter a query.")
