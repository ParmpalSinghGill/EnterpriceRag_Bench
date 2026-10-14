# Enterprise RAG & Agent with LlamaIndex (AWS Deployment Ready)

This package implements the proven **Enterprise Hybrid RAG (88.5%–90.0% accuracy)** directly inside the **LlamaIndex** framework, designed for minimal code footprint and straightforward deployment to **AWS (ECS, App Runner, EC2, or Lambda)**.

---

## 1. Quick Start: Use in Python with 3 Lines of Code

```python
from llamaindex_app import get_query_engine

# 1. Initialize query engine (automatically loads BGE-M3, Postgres HNSW, and SQLite BM25)
query_engine = get_query_engine(top_k=7)

# 2. Ask any enterprise question
response = query_engine.query("What was the latency reduction achieved by the lattice scheduler?")

print(response.response)
```

### Or Use the Autonomous ReAct Agent

```python
from llamaindex_app import get_agent

# Initialize agent equipped with internal enterprise search tools
agent = get_agent(verbose=True)

# Converse or run multi-step reasoning
reply = agent.chat("Check our Linear tickets and tell me why regional failover had jitter last week.")
print(reply)
```

---

## 2. Core Components

1. **`EnterpriseHybridRetriever` ([`llamaindex_app/retriever.py`](retriever.py))**:
   - Subclasses LlamaIndex `BaseRetriever`.
   - Fuses **Dense BGE-M3 (PostgreSQL pgvector HNSW)** and **Sparse BM25 (SQLite FTS5)** with Reciprocal Rank Fusion ($k=20, w_{\text{sparse}}=1.2$).
   - Leverages **Domain Grounding & HyDE expansion** to eliminate entity-masking failures.
   - Emits clean LlamaIndex `NodeWithScore` objects.

2. **`NemotronLLM` ([`llamaindex_app/llm.py`](llm.py))**:
   - Subclasses LlamaIndex `CustomLLM`.
   - Connects to `nvidia/nemotron-3.5-lightning-30b-a3b` (or any OpenAI-compatible / AWS Bedrock endpoint).
   - Enforces strict factual groundings and zero-hallucination guardrails.

3. **`FastAPI Application` ([`llamaindex_app/app.py`](app.py))**:
   - Production REST API exposing:
     - `GET /health` (Target group health checks)
     - `POST /query` (Deterministic Top-7 RAG answering)
     - `POST /chat` (Multi-turn ReAct agent reasoning)

---

## 3. Running Locally / Testing

### Run the FastAPI Server
```bash
uvicorn llamaindex_app.app:app --host 0.0.0.0 --port 8000
```

### Test Query via cURL
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What audit log retention period did Redwood commit to during the 2027 API migration?"}'
```

### Run Benchmark Evaluation
```bash
python llamaindex_app/evaluate.py --samples 5 --top-k 7
```

---

## 4. Deploying to AWS

### A. AWS App Runner (Easiest - Fully Managed)
1. Push the Docker image to **AWS Elastic Container Registry (ECR)**:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
   docker build -t enterprise-rag-llamaindex -f llamaindex_app/Dockerfile .
   docker tag enterprise-rag-llamaindex:latest <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/enterprise-rag:latest
   docker push <YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/enterprise-rag:latest
   ```
2. Create an **App Runner Service**:
   - **Source**: ECR Image `<YOUR_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/enterprise-rag:latest`
   - **Port**: `8000`
   - **Health Check Path**: `/health`
   - **Environment Variables**:
     - `ERAG_DATABASE_URL`: `postgresql://<user>:<pwd>@<rds-endpoint>:5432/<dbname>`
     - `OPENCODE_BASE_URL`: `<nim-endpoint-or-opencode-url>`
     - `OPENCODE_API_KEY`: `<api-key>`

### B. AWS ECS (Fargate or EC2)
- Deploy task definition with container port `8000`.
- Attach Application Load Balancer (ALB) pointing health checks to `HTTP:8000/health`.
- Attach AWS EFS (Elastic File System) volume if mounting shared `indexes/bm25_fts5.db`.
- Point `ERAG_DATABASE_URL` to an **Amazon RDS for PostgreSQL** (with `pgvector` extension enabled).

---

## 5. Configuration (Environment Variables)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `ERAG_DATABASE_URL` | `postgresql://erag:erag@127.0.0.1:5433/erag` | PostgreSQL connection string with pgvector |
| `OPENCODE_BASE_URL` | `https://integrate.api.nvidia.com/v1` | LLM API endpoint base URL |
| `OPENCODE_API_KEY` | *(None)* | API authentication key |
| `ERAG_LLM_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | Target generation and reasoning model |
| `PORT` | `8000` | HTTP service port for Uvicorn |

