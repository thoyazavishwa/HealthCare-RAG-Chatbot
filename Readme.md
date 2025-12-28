# HealthCare RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot designed for healthcare applications. It leverages domain-specific data to provide accurate and context-aware responses to user queries about clinical guidelines and drug interactions.

## Features
- **RAG-based Chatbot**: Combines retrieval of relevant documents with generative AI for informed answers.
- **Healthcare Focus**: Uses clinical guidelines and drug interaction data.
- **Local Vector Store**: Utilizes ChromaDB for efficient document retrieval.

## Project Structure
```
app.py                # Main application (chatbot server)
rag-ingest.py         # Script to ingest and index documents
requirements.txt      # Python dependencies
Readme.md             # Project documentation
data/                 # Source documents
  Clinical_Guidelines/
  Drug_Interactions/
vectorstore/          # ChromaDB vector store
  chroma/
    chroma.sqlite3
    ...
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation
1. Clone the repository or download the source code.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Data Ingestion
1. Place your clinical guidelines and drug interaction documents in the respective folders under `data/`.
2. Run the ingestion script to index documents:
   ```bash
   python rag-ingest.py
   ```

### Running the Chatbot
Start the chatbot server:
```bash
python app.py
```

## Usage
- Interact with the chatbot via the provided interface (see `app.py` for details).
- The chatbot will retrieve relevant information from the ingested healthcare documents and generate responses.

## Folder Details
- `data/Clinical_Guidelines/`: Place clinical guideline documents here.
- `data/Drug_Interactions/`: Place drug interaction documents here.
- `vectorstore/`: Stores the ChromaDB vector database files.

## License
This project is for educational and research purposes. Please ensure compliance with healthcare data privacy regulations when using real data.

## Acknowledgements
- [ChromaDB](https://www.trychroma.com/)
- OpenAI and other LLM providers
