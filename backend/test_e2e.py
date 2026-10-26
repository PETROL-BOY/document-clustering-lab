
import urllib.request
import json

def test_analyze():
    req_data = json.dumps({'k': 4, 'method': 'gemini', 'algorithm': 'kmeans'}).encode('utf-8')
    req = urllib.request.Request(
        'http://127.0.0.1:8000/api/analyze',
        data=req_data,
        headers={'Content-Type': 'application/json'}
    )
    print("Calling /api/analyze with Gemini dense embeddings...")
    res = urllib.request.urlopen(req, timeout=120)
    data = json.loads(res.read())

    print("\n================== GEMINI CLUSTERING RESULTS ==================")
    print(f"Run ID: {data['run_id']}")
    print(f"Algorithm: {data['algorithm']} | Representation: {data['method']}")
    print(f"Silhouette Score (Cosine): {data['metrics']['silhouette']}")
    print(f"Inertia (WCSS): {data['metrics']['inertia']}")
    if data['metrics']['ari'] is not None:
        print(f"Adjusted Rand Index (ARI): {data['metrics']['ari']}")
        print(f"Normalized Mutual Info (NMI): {data['metrics']['nmi']}")

    print("\n--- Synthesized Clusters ---")
    for c in data['clusters']:
        print(f"\n[Cluster #{c['id']}] Size: {c['size']} docs ({c['percentage']}%)")
        print(f"  Title: {c['llm_label']}")
        print(f"  Summary: {c['llm_summary']}")
        print(f"  Objective Keywords (c-TF-IDF): {', '.join(c['objective_keywords'])}")
        print(f"  LLM Keywords: {', '.join(c['llm_keywords'])}")
        print(f"  Centroid-Nearest Representatives: {len(c['representatives'])} chunks")
        for rep in c['representatives'][:2]:
            print(f"    * {rep['title']} (d={rep['distance_to_centroid']})")

def test_chat():
    question = "How does solar power forecasting predict plant generation?"
    req_data = json.dumps({'question': question, 'top_k': 3}).encode('utf-8')
    req = urllib.request.Request(
        'http://127.0.0.1:8000/api/chat',
        data=req_data,
        headers={'Content-Type': 'application/json'}
    )
    print(f"\n\n================== GROUNDED RAG CHAT QUERY ==================")
    print(f"Question: {question}")
    res = urllib.request.urlopen(req, timeout=60)
    data = json.loads(res.read())

    print(f"Threshold Passed: {data['threshold_passed']}")
    print(f"\nGenerated Answer:\n{data['answer']}")
    print("\nRetrieved Sources:")
    for cit in data['citations']:
        print(f"  * [{cit['title']}] (similarity: {cit['score']})")

if __name__ == "__main__":
    test_analyze()
    test_chat()
