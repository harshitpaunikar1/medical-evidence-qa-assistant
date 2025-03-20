# Medical Evidence Q&A Assistant Diagrams

Generated on 2026-04-26T04:29:37Z from README narrative plus project blueprint requirements.

## RAG pipeline architecture

```mermaid
flowchart TD
    N1["Step 1\nDefined trusted-source scope around medical PDFs, guidelines, and PubMed-style ref"]
    N2["Step 2\nBuilt ingestion and parsing flow with PDF extraction, cleanup, chunking, embedding"]
    N1 --> N2
    N3["Step 3\nAdded reranking so the strongest evidence surfaced first before answer generation,"]
    N2 --> N3
    N4["Step 4\nConstrained Gemini Flash to answer only from retrieved context and designed respon"]
    N3 --> N4
    N5["Step 5\nTreated refusal behaviour as a product feature by warning or declining when eviden"]
    N4 --> N5
```

## Retrieval → reranking → generation flow

```mermaid
flowchart LR
    N1["Inputs\nMedical PDFs, guidelines, or evidence documents"]
    N2["Decision Layer\nRetrieval → reranking → generation flow"]
    N1 --> N2
    N3["User Surface\nAPI-facing integration surface described in the README"]
    N2 --> N3
    N4["Business Outcome\nOutput quality"]
    N3 --> N4
```

## Evidence Gap Map

```mermaid
flowchart LR
    N1["Present\nREADME, diagrams.md, local SVG assets"]
    N2["Missing\nSource code, screenshots, raw datasets"]
    N1 --> N2
    N3["Next Task\nReplace inferred notes with checked-in artifacts"]
    N2 --> N3
```
