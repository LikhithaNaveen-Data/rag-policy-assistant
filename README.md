# RAG Policy Assistant (Chunking + Embeddings + Vector DB)

## Overview
This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline built from scratch using Python.
It enables question answering over enterprise-style policy documents by combining document chunking, embeddings, and a vector database.

The focus of this project is **grounded retrieval** rather than direct LLM generation, ensuring answers are based on actual source data and minimizing hallucinations.

---

## Problem Statement
In enterprise environments, employees often need quick answers to questions about internal policies such as leave, work-from-home, and insurance.
Traditional keyword search fails to capture semantic meaning.

This project solves the problem by implementing a semantic retrieval pipeline using embeddings and vector search.

---

## Architecture
flowchart TD
    U[User Interface] --> A[API: FastAPI Backend]
    A --> B[Query Preprocessor]
    B --> C[Embedding Generator]
    C --> D[(Vector Database - FAISS)]
    D --> E[Semantic Search (Top-K)]
    E --> F[Context Builder]
    F --> G[LLM Inference (OpenAI / Azure / Bedrock)]
    G --> H[Response Formatter]
    H --> U[Response to User]
User Interface: Users input their questions (e.g., policy queries).

Backend API: FastAPI receives queries, handles request validation and orchestration.

Query Preprocessing: Cleans and normalizes the query text before embedding.

Embedding Generator: Converts text into dense vectors using a chosen model (OpenAI, Hugging Face, etc.).

Vector Database (FAISS): Stores chunk embeddings from policy documents, enabling fast similarity search.

---

## Evaluation & Performance Metrics

To validate quality, grounding, and production readiness of this RAG system, the following criteria were measured:

🔹 Retrieval Quality

Recall@K: Ensures that relevant document chunks are included in the top K results.

Mean Reciprocal Rank (MRR): Measures how high the true relevant document ranks among retrieved documents.

Precision@K: Proportion of retrieved chunks that are relevant.

These help quantify whether the semantic index and vector search are effective.

🔹 Generation Quality

Hallucination Rate: Percentage of model responses containing unsupported claims or incorrect facts.

Context Utilization Score: Measures how well the generated answer reflects the retrieved context versus generic model knowledge.

Human Judgement Score: A small sample of user evaluations on coherence, utility, and factual accuracy.

These assess whether the LLM’s output is grounded and reliable rather than made up.

🔹 System Latency & Efficiency

Avg. Response Time (ms): Time taken from receiving a query to sending the answer.

Token Cost per Query: Estimated API token cost for retrieval + inference.

Concurrent Request Handling: Maximum requests processed per second without degradation.

These metrics mirror real-world production needs — timely and cost-efficient responses.

## Evaluation Code Snippet

from sklearn.metrics.pairwise import cosine_similarity

def similarity_score(query_embedding, doc_embedding):
    return cosine_similarity([query_embedding], [doc_embedding])[0][0]

print("Semantic Similarity:", similarity_score(q_emb, d_emb))

Semantic Search: Retrieves top-K relevant chunks from the vector DB based on cosine similarity.

Context Builder: Combines retrieved documents with the user query to form a grounded prompt.

LLM Inference: The LLM generates answers by leveraging contextual data, reducing hallucinations.

Response Formatter: Post-processes the model output and returns it to the frontend.
