import os
import yaml
from langchain.vectorstores import FAISS, Chroma, Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

def load_config(path="config.yaml"):
    """
    Load configuration from a YAML file (works in script or notebook).

    Args:
        path (str): Path to the YAML config file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    try:
        # Check if __file__ exists (script) or fallback to current working dir (notebook)
        base_dir = os.path.dirname(__file__) if '__file__' in globals() else os.getcwd()
        abs_path = os.path.join(base_dir, path)

        with open(abs_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {abs_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file: {e}")

def load_vectorstore(config, embedding_model=None):
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vector_config = config["vectorstore"]
    provider = vector_config["provider"]

    if provider == "pinecone":
        pinecone_cfg = vector_config["pinecone"]
        pc = Pinecone(api_key=pinecone_cfg["api_key"])
        index_name = vector_config["index_name"]

        # Delete if exists, then recreate (optional — adjust for production!)
        if index_name in pc.list_indexes().names():
            pc.delete_index(index_name)

        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=vector_config["dimension"],
                metric=vector_config["metric"],
                spec=ServerlessSpec(
                    cloud=pinecone_cfg["cloud"],
                    region=pinecone_cfg["region"]
                )
            )

        index = pc.Index(index_name)
        return LangchainPinecone(index, embedding_model, index_name=index_name)

    elif provider == "chroma":
        persist_dir = vector_config["chroma"]["persist_directory"]
        return Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )

    elif provider == "faiss":
        persist_path = vector_config["faiss"]["persist_path"]
        if os.path.exists(persist_path):
            return FAISS.load_local(persist_path, embedding_model)
        else:
            return None  # You'll need to create and save it after adding docs

    else:
        raise ValueError(f"Unsupported vectorstore provider: {provider}")