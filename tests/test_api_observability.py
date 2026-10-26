import pytest
from fastapi.testclient import TestClient
from app.api import app, rag_logs, query_metrics, log_lock

client = TestClient(app)

def test_overview_endpoint_structure():
    response = client.get("/api/analytics/overview")
    assert response.status_code == 200
    data = response.json()
    assert "total_docs" in data
    assert "total_chunks" in data
    assert "total_clusters" in data
    assert "total_questions" in data
    assert "evidence_pass_rate_pct" in data
    assert "avg_rag_latency_ms" in data
    assert "avg_rejected_latency_ms" in data

def test_chat_logging_and_capping():
    # Clear logs under lock for test isolation
    with log_lock:
        rag_logs.clear()
        query_metrics["total_questions"] = 0
        query_metrics["successful_queries"] = 0
        query_metrics["insufficient_evidence_queries"] = 0

    # Inject 105 mock log entries to verify deque maxlen=100 capping
    with log_lock:
        for i in range(105):
            query_metrics["total_questions"] += 1
            status = "success" if i % 2 == 0 else "insufficient_evidence"
            rag_logs.appendleft({
                "id": f"log_{query_metrics['total_questions']}",
                "timestamp": "2026-10-26T14:30:00",
                "question": f"Question number {i} regarding machine learning",
                "status": status,
                "citations_count": 2 if status == "success" else 0,
                "similarity_scores": [0.85] if status == "success" else [],
                "latency_ms": 320.5,
                "error": None
            })

    # Verify capped at 100
    assert len(rag_logs) == 100
    # Verify monotonic counter is 105
    assert query_metrics["total_questions"] == 105

    # Test GET /api/analytics/logs default limit
    res = client.get("/api/analytics/logs?limit=50")
    assert res.status_code == 200
    logs = res.json()
    assert len(logs) == 50
    # Most recent log should be log_105
    assert logs[0]["id"] == "log_105"

def test_logs_filtering_by_status():
    res_succ = client.get("/api/analytics/logs?status=success&limit=100")
    assert res_succ.status_code == 200
    logs_succ = res_succ.json()
    assert all(l["status"] == "success" for l in logs_succ)

    res_insuff = client.get("/api/analytics/logs?status=insufficient_evidence&limit=100")
    assert res_insuff.status_code == 200
    logs_insuff = res_insuff.json()
    assert all(l["status"] == "insufficient_evidence" for l in logs_insuff)

def test_logs_search_filter():
    res = client.get("/api/analytics/logs?search=number 42")
    assert res.status_code == 200
    logs = res.json()
    assert len(logs) == 1
    assert "number 42" in logs[0]["question"]
