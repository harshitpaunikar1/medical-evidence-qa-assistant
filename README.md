# Medical Evidence Q&A Assistant

This repository documents a safety-aware medical question-answering system that retrieves trusted evidence before responding, cites its sources, and refuses unsupported or risky questions.

## Domain
Healthcare / MedTech

## Overview
Designed as an education-focused assistant rather than a diagnosis engine.

## Methodology
1. Defined trusted-source scope around medical PDFs, guidelines, and PubMed-style references so the system answered from controlled evidence only.
2. Built ingestion and parsing flow with PDF extraction, cleanup, chunking, embeddings, and Qdrant indexing for retrieval-ready medical content.
3. Added reranking so the strongest evidence surfaced first before answer generation, improving citation quality and grounding.
4. Constrained Gemini Flash to answer only from retrieved context and designed response patterns for citations, uncertainty, and disclaimers.
5. Treated refusal behaviour as a product feature by warning or declining when evidence was weak or the request moved into unsafe territory.
6. Exposed the pipeline through FastAPI so the assistant could be integrated into a lightweight healthcare education workflow.

## Skills
- RAG
- PyMuPDF
- MiniLM Embeddings
- Qdrant
- Reranking
- Gemini Flash
- Safety Guardrails
- FastAPI

## Source
This README was generated from the portfolio project data used by `/Users/harshitpanikar/Documents/Test_Projs/harshitpaunikar1.github.io/index.html`.
