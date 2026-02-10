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
