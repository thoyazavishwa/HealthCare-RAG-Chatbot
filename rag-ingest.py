import os
import pdfplumber
from dotenv import load_dotenv
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import chromadb
from langchain_community.vectorstores import Chroma
from chromadb.config import Settings

# -------------------------------------------------
# Load Environment Variables
# -------------------------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in environment variables")

PDF_DIRS = ["data/Clinical_Guidelines", "data/Drug_Interactions"]
VECTORSTORE_DIR = "vectorstore/chroma"

# -------------------------------------------------
# PDF Extraction (Text + Tables)
# -------------------------------------------------
def extract_pdf_content(pdf_path: str) -> List[Document]:
    documents = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            # -------- Text --------
            text = page.extract_text()
            if text and text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_num + 1,
                            "content_type": "text"
                        }
                    )
                )

            # -------- Tables --------
            tables = page.extract_tables()
            for table in tables:
                table_text = "\n".join(
                    [" | ".join(cell or "" for cell in row) for row in table]
                )

                documents.append(
                    Document(
                        page_content=f"TABLE:\n{table_text}",
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_num + 1,
                            "content_type": "table"
                        }
                    )
                )

    return documents

# -------------------------------------------------
# Load All PDFs
# -------------------------------------------------
def load_all_pdfs(pdf_dirs: list) -> List[Document]:
    all_docs = []
    for pdf_dir in pdf_dirs:
        if not os.path.exists(pdf_dir):
            print(f"❌ PDF directory not found: {pdf_dir}")
            continue
        for file in os.listdir(pdf_dir):
            if file.lower().endswith(".pdf"):
                pdf_path = os.path.join(pdf_dir, file)
                print(f"📄 Processing: {file} in {pdf_dir}")
                all_docs.extend(extract_pdf_content(pdf_path))
    return all_docs

# -------------------------------------------------
# Chunking (Clinical Optimized)
# -------------------------------------------------
def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(documents)

# -------------------------------------------------
# Ingestion Pipeline
# -------------------------------------------------

def ingest():
    print("🚀 Starting RAG ingestion (OpenAI API)...")

    # Ensure vectorstore directory exists
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    # 1. Load PDFs from all directories
    raw_docs = load_all_pdfs(PDF_DIRS)
    print(f"✅ Extracted {len(raw_docs)} raw sections")
    if not raw_docs:
        print("❌ No documents found in PDF directories. Exiting.")
        return

    # 2. Chunk
    chunked_docs = chunk_documents(raw_docs)
    print(f"🧩 Created {len(chunked_docs)} chunks")
    if not chunked_docs:
        print("❌ No chunks created. Exiting.")
        return

    # 3. Embeddings (OpenAI)
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
        )
    except Exception as e:
        print(f"❌ Error initializing OpenAIEmbeddings: {e}")
        return

    # 4. Vector Store
    try:
        client = chromadb.PersistentClient(    # PersistentClient for disk storage
            path=VECTORSTORE_DIR,  
            settings=Settings(
                anonymized_telemetry=False
            )
        )

        vectordb = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            client=client,
            collection_name="medical_documents"
        )

        print("🎉 Ingestion completed successfully!")
        print(f"📁 Stored files: {os.listdir(VECTORSTORE_DIR)}")
    except Exception as e:
        print(f"❌ Error creating or persisting Chroma vectorstore: {e}")
        return

# Entry Point
# -------------------------------------------------
if __name__ == "__main__":
    ingest()
