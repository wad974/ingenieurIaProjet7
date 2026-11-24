"""Modules utilitaires du projet.

- ``config`` / ``logging_config`` : configuration et journalisation
- ``data_loader`` / ``markdown_converter`` : chargement des sources
- ``document_ingestion`` / ``document_store`` : pipeline RAG documentaire
- ``vector_store`` : embeddings Ollama, découpage, index FAISS
- ``query_classifier`` / ``rag`` : routage RAG vs DIRECT et génération
- ``database`` : feedback utilisateurs (héritage, non utilisé par l'API)
"""
