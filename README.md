# support-triage-agent

A SaaS support-ticket triage and RAG auto-reply agent in Python.

## Project Goal
Automate SaaS customer support triage by classifying incoming ticket intents and categories, retrieving relevant knowledge base context via vector embeddings (pgvector), generating grounded auto-replies, and intelligently escalating complex or sensitive issues to human agents.

## Data & License
This project utilizes the **Bitext Customer Support LLM Chatbot Training Dataset** (`Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv`).
- **License**: Community Data License Agreement – Sharing, Version 1.0 (CDLA-Sharing-1.0)
- **Copyright**: (c) Bitext Innovations
- **Data Policy**: Dataset files are downloaded at runtime into `data/` and are not committed to source control.

## Ticket embeddings

Open `notebooks/embed_tickets_colab.ipynb` in Google Colab and select a GPU
runtime. The notebook downloads the Bitext CSV, applies the same intent-stratified
80/20 split as `pipeline/ingest.py`, embeds each training instruction with
`BAAI/bge-small-en-v1.5`, and downloads `embeddings.parquet`. Copy that file to
`data/embeddings.parquet`, apply `sql/migrations/002_ticket_embeddings.sql`, then
load it into Postgres:

```sh
make load-embeddings
```

Only training tickets are embedded. Keeping test tickets out of the retrieval
index prevents their text from leaking into evaluation results.
