import os
import glob
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv(override=True)

BASE_DIR = Path(__file__).parent.parent

KNOWLEDGE_BASE = BASE_DIR / "knowledge-base"
DB_NAME = BASE_DIR / "vector_db"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL
)


def fetch_documents():

    folders = glob.glob(str(KNOWLEDGE_BASE / "*"))

    documents = []

    for folder in folders:

        if not os.path.isdir(folder):
            continue

        doc_type = os.path.basename(folder)

        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={
                "encoding": "utf-8"
            }
        )

        folder_docs = loader.load()

        for doc in folder_docs:

            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    print(f"Loaded {len(documents)} documents")

    return documents


def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks")

    return chunks


def create_vectorstore(chunks):

    if DB_NAME.exists():

        Chroma(
            persist_directory=str(DB_NAME),
            embedding_function=embeddings
        ).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_NAME),
        collection_name="baseline"
    )

    print(
        f"Stored {len(chunks)} vectors in Chroma"
    )

    return vectorstore


if __name__ == "__main__":

    documents = fetch_documents()

    chunks = create_chunks(documents)

    create_vectorstore(chunks)

    print("\nIngestion complete.")