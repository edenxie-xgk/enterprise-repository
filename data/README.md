# Data Directory

This folder is reserved for local datasets, benchmark exports, and document seed files used during development.

Large runtime artifacts are intentionally not committed:

- `benchmarks/*.json` stores local benchmark outputs.
- `chinese_documents_seed/*.pdf` stores optional source documents for ingestion and LoRA-data preparation.
- The tracked `chinese_documents_seed/manifest.csv` lists the original public document URLs so the PDFs can be downloaded again when needed.

The project supports `pdf`, `doc/docx`, `md/markdown`, `txt`, `xls/xlsx`, `csv`, `pptx`, `json`, and common image formats in the ingestion path.
