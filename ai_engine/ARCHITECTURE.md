# Enterprise Local AI Architecture

## Overview
This document outlines the architecture for the local, privacy-first AI engine.
The system is designed to run entirely offline (CPU/GPU) using PyTorch and HuggingFace.

## Folder Structure
```
ai_engine/
├── __init__.py           # Package marker
├── server.py             # FastAPI entry point (isolated process)
├── model_config.py       # Phase 1: GPU/Device configuration & Health Checks
├── preprocessing.py      # Phase 2: NLP Cleaning & Tokenization
├── embeddings.py         # Phase 3: Vector generation (SentenceTransformers)
├── vector_store.py       # Phase 3: FAISS Memory Store
├── reasoning.py          # Phase 4: Chain-of-Thought & Reasoning Layer
└── guardrails.py         # Phase 5: Input validation & Output safety
```

## Phase 1: Hardware Verification & Config (`model_config.py`)
- Detects CUDA availability.
- Fallback to MPS (Mac) or CPU.
- Logs device capabilities (VRAM, Cores) at startup.
- **Constraint**: No training loops allowed. Inference only.

## Phase 3: Model Pipeline Design
### Data Ingestion
- **Raw Input**: User text + Optional Images.
- **Preprocessing**: Regex cleaning, normalization.

### Feature Extraction
- **Text**: `all-MiniLM-L6-v2` (Local, ~80MB) for lightweight embeddings.
- **Vision**: Placeholder for ViT functionality.

### Inference Layer
- **Model**: Design allows swapping base models (e.g., Llama-2-7b-chat-quantized).
- **LoRA Support**: Architecture supports loading LoRA adapters for domain specificity.

## Phase 4: Reasoning Layer (`reasoning.py`)
- **Concept**: Split generation into "Thinking" and "Answering".
- **Mechanism**:
  1. Retrieve relevant context (RAG).
  2. Generate internal reasoning trace (hidden or debug-only).
  3. Synthesize final user-facing response.
- **Auditability**: All reasoning steps are logged for enterprise compliance.

## Phase 5: Security (`guardrails.py`)
- **Input Sanitization**: Prevent prompt injection.
- **Output Filtering**: Regex-based redaction of PII (Mock for now).
- **Latency Budget**: Hard timeouts for inference to prevent UI blocking.
