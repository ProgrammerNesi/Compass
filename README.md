self-evaluating-rag/
│
├── app.py
│
├── .env
├── requirements.txt
│
├── data/
│   ├── uploads/
│   └── processed/
│
├── vector_db/
│
├── evaluation/
│   ├── gold_set.jsonl
│   ├── evaluator.py
│   ├── metrics.py
│   └── baseline.py
│
├── ingestion/
│   ├── loader.py
│   ├── chunker.py
│   └── embedder.py
│
├── retrieval/
│   ├── vector.py
│   ├── rewrite.py
│   ├── expansion.py
│   └── reranker.py
│
├── agent/
│   ├── state.py
│   ├── nodes.py
│   ├── tools.py
│   └── graph.py
│
├── generation/
│   └── answer.py
│
└── ui/
    └── components.py