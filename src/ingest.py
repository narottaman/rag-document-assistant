from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import wandb
import yaml

def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)

def ingest_documents(pdf_dir: str) -> list:
    config = load_config()
    chunk_size = config["embedding"]["chunk_size"]
    chunk_overlap = config["embedding"]["chunk_overlap"]

    # W&B: log ingestion params
    wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        job_type="ingestion",
        config={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []
    import os
    for fname in os.listdir(pdf_dir):
        if fname.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, fname))
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

    # W&B: log how many chunks were created
    wandb.log({
        "total_documents": len(os.listdir(pdf_dir)),
        "total_chunks": len(all_chunks),
        "avg_chunk_length": sum(len(c.page_content) 
                                for c in all_chunks) / len(all_chunks)
    })
    wandb.finish()
    return all_chunks