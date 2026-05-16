from src.config.set import settings
from src.core.preprocess import preprocess_documents
from src.core.vector_store import create_vectorstore


def ingest_data():

    print("Loading and preprocessing documents...")

    documents = preprocess_documents(
        settings.RAW_DATA_PATH
    )

    print(f"Total documents processed: {len(documents)}")


    print("Creating Chroma vector database...")

    create_vectorstore(documents)

    print("Vector database created successfully.")


if __name__ == "__main__":

    ingest_data()