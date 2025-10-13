from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Load environment variables (for GROQ API key)
load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "placement_bot"

llm = None
vector_store = None
retriever = None


def initialize_components():
    """Initialize LLM and Vector Store"""
    global llm, vector_store, retriever

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})


def process_folder(folder_path: str, reset: bool = False):
    """
    Loads all .txt files from a folder and stores them in vector DB
    Only adds chunks if vector store is empty or reset=True
    """
    yield "🔄 Initializing Components..."
    initialize_components()

    # Reset vector store only if requested
    if reset:
        yield "🗑️ Resetting vector store..."
        vector_store.reset_collection()

    # Check if vector store is already populated
    if len(vector_store.get()) > 0:
        yield f"⚡ Vector store already has {len(vector_store.get())} chunks. Skipping adding new chunks."
        return

    yield f"📂 Loading text files from {folder_path}..."
    loader = DirectoryLoader(folder_path, glob="*.txt", show_progress=True)
    data = loader.load()

    yield "✂️ Splitting text into chunks..."
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(data)
    yield f"📊 Total Chunks: {len(docs)}"

    yield "📝 Adding chunks to vector database..."
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield f"✅ Placement data added to vector database successfully! ({len(docs)} docs)"



def generate_answer(query: str):
    """Query the placement bot using updated LangChain API"""
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")

    # Custom prompt for QA
    prompt = PromptTemplate(
    template="""You are a helpful assistant for answering questions about placements. 
Use the retrieved context to answer the question clearly and professionally (pointwise is preferred). 
Do not mention the word 'context' or copy text directly. 
If the answer is not found, say you don't know.

Question: {input}

Relevant Information:
{context}

Final Answer (natural and conversational):""",
    input_variables=["context", "input"],
)


    # Step 1: Stuff documents into LLM
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # Step 2: Retrieval chain
    chain = create_retrieval_chain(retriever, question_answer_chain)

    # Run chain
    result = chain.invoke({"input": query})
    answer = result["answer"]
    sources = result.get("context", [])

    return answer, sources


if __name__ == "__main__":
    # Folder containing placement .txt files
    folder = "resources/placement_texts"

    # Step 1: Load and process files
    for msg in process_folder(folder):
        print(msg)

    # Step 2: Ask questions
    while True:
        query = input("\nAsk Placement Bot: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting Placement Bot.")
            break

        answer, sources = generate_answer(query)
        print("\n🤖 Answer:", answer)

        if sources:
            print("📌 Sources:")
            for s in sources:
                print("-", s.page_content[:200], "...")
        else:
            print("📌 Sources: Not available")
