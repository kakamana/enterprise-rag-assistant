from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_stub():
    r = client.post("/ask", json={"question": "What is the expense limit?", "top_k": 5})
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "sources" in body
    assert 0 <= body["faithfulness_score"] <= 1
    assert "decision_aid_disclaimer" in body


def test_data_generator_deterministic():
    from enterprise_rag.data import build_corpus, build_qa

    a = build_corpus(n_paragraphs=200, seed=7)
    b = build_corpus(n_paragraphs=200, seed=7)
    assert a.equals(b)
    qa_a = build_qa(a, n_qa=20, seed=7)
    qa_b = build_qa(b, n_qa=20, seed=7)
    assert qa_a.equals(qa_b)


def test_features_tokenize():
    from enterprise_rag.features import tokenize

    assert tokenize("Hello, World 2024!") == ["hello", "world", "2024"]
