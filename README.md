# Deel AI Engineer Challenge – Solution

##  Overview
This repository contains the completed solution for the Deel Python AI Engineer Challenge. The project implements a robust Flask API service for advanced text processing on transaction data, addressing two core challenges:

 Fuzzy Matching: Connecting transaction descriptions to known users.

Semantic Search: Identifying transactions that are semantically similar to a user-provided query.

The solution is developed using Python best practices and is accompanied by a detailed strategy for scaling the proof-of-concept to a production environment.

---

##  Project Structure

| File                 | Description                                                                               |
| -------------------- | ----------------------------------------------------------------------------------------- |
| **app.py**           | Main Flask application, including data loading, pre-computation logic, and API endpoints. |
| **requirements.txt** | Lists all Python dependencies.                                                            |
| **transactions.csv** | Input dataset for transaction details.                                                    |
| **users.csv**        | Input dataset for user details.                                                           |

---

##  Getting Started

This solution requires **Python 3.8+** and assumes that `transactions.csv` and `users.csv` are in the root directory.

### 1️.Setup Environment

```bash
python -m venv venv
# On Windows PowerShell
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2️.Start the Server

```bash
flask run
```

>  Wait for “Embeddings are ready.” before testing endpoints.

---

##  Example API Usage

###  Match Transaction

```bash
curl http://127.0.0.1:5000/match_transaction/caqjJtrI
```

Example Response:

```json
{
  "transaction_id": "caqjJtrI",
  "matched_user": "Matthew West",
  "match_score": 92
}
```

###  Similar Transactions

```bash
Invoke-RestMethod -Uri http://127.0.0.1:5000/similar_transactions -Method Post -ContentType "application/json" -Body '{"description": "transfer from my bank account"}'
```

Example Response:

```json
{
  "query": "transfer from my bank account",
  "similar_transactions": [
    {"transaction_id": "Xyz123", "score": 0.88},
    {"transaction_id": "Abc456", "score": 0.83}
  ]
}
```

---

##  Task 1: Match Transaction to User

**Approach:**

* Extracts user name substring between “From” and “for Deel” using regex.
* Compares with user list using **thefuzz.token_set_ratio** for robust fuzzy matching.

**Limitation:**
Fixed text pattern; breaks if transaction format changes.

**Improvements:**

* NLP-based NER (spaCy/Hugging Face) for name extraction.
* Store user names in **Elasticsearch/OpenSearch** for scalable fuzzy lookup.

---

##  Task 2: Find Semantically Similar Transactions

**Approach:**

* Uses **sentence-transformers (all-MiniLM-L6-v2)** to compute embeddings.
* Computes **cosine similarity** for semantic matching.

**Limitation:**
In-memory storage (NumPy) — not scalable for millions of records.

**Improvements:**

* Use a **Vector Database** (Pinecone, ChromaDB, FAISS).
* Move embedding generation to **asynchronous workers (Celery/Kafka)**.

---

##  Task 3: Production Readiness

| Area                  | Improvement             | Rationale                               |
| --------------------- | ----------------------- | --------------------------------------- |
| **Deployment**     | Docker                  | Consistent runtime across environments. |
| **Infrastructure** | Kubernetes / Serverless | Auto-scaling and fault tolerance.       |
| **API Layer**      | Auth, rate limiting     | Security and controlled access.         |
| **Monitoring**     | ELK / Prometheus        | Track latency and errors.               |
| **CI/CD**          | GitHub Actions          | Automated testing and deployment.       |


