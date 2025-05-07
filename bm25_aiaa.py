import numpy as np
import pyodbc
import torch
import requests
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from rank_bm25 import BM25Okapi
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load AIA - Vietnamese embedding
tokenizer = AutoTokenizer.from_pretrained("AITeamVN/Vietnamese_Embedding")
model = AutoModel.from_pretrained("AITeamVN/Vietnamese_Embedding").to(device)

def get_db_connection():
    try:
        conn = pyodbc.connect(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=;"
            "DATABASE=chatbot;"
            "Trusted_Connection=yes;"
        )
        return conn
    except Exception as e:
        print(f"❌ Lỗi kết nối database: {e}")
        return None

def load_documents():
    conn = get_db_connection()
    if conn is None:
        return []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, content FROM Documents")
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"❌ Lỗi tải dữ liệu Documents: {e}")
        return []

def load_chunks_by_documents(doc_ids):
    if not doc_ids:
        return []
    conn = get_db_connection()
    if conn is None:
        return []
    try:
        cursor = conn.cursor()
        format_strings = ",".join(["?" for _ in doc_ids])
        query = f"SELECT chunk_id, document_id, chunk_text FROM Chunks WHERE document_id IN ({format_strings})"
        cursor.execute(query, doc_ids)
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"❌ Lỗi tải dữ liệu Chunks: {e}")
        return []

def rerank_chunks(query, chunks):
    rerank_tokenizer = AutoTokenizer.from_pretrained("itdainb/PhoRanker")
    rerank_model = AutoModelForSequenceClassification.from_pretrained("itdainb/PhoRanker").to("cuda" if torch.cuda.is_available() else "cpu")
    
    inputs = [query.lower() + " [SEP] " + chunk[2].lower() for chunk in chunks]
    tokenized_inputs = rerank_tokenizer(inputs, padding=True, truncation=True, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        scores = rerank_model(**tokenized_inputs).logits.squeeze(-1).cpu().numpy()
    
    ranked_indices = np.argsort(scores)[::-1][:1000]  # Chọn chunk có điểm cao nhất
    top_chunks = [chunks[i] for i in ranked_indices]
    return top_chunks

def call_gpt_api(prompt):
    api_key = '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'
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500 
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

app = Flask(__name__)
CORS(app)

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return '', 200

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON received"}), 400

    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400
      
    query_lower = query.lower()    
    documents = load_documents()

    document_texts = [doc.content.lower() for doc in documents]
    tokenized_docs = [text.split() for text in document_texts]
    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = query_lower.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_25_idx = np.argsort(bm25_scores)[::-1][:30]

    relevant_docs = [documents[i] for i in top_25_idx]

    # Nhúng câu hỏi bằng AIA
    tokens = tokenizer(query_lower, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        query_embedding = torch.mean(model(**tokens).last_hidden_state, dim=1).squeeze(0).cpu().numpy()
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Nhúng 10 tài liệu bằng AIA
    doc_embeddings = []
    for doc in relevant_docs:
        tokens = tokenizer(doc.content, padding=True, truncation=True, max_length=256, return_tensors="pt").to(device)
        with torch.no_grad():
            doc_embedding = torch.mean(model(**tokens).last_hidden_state, dim=1).squeeze(0).cpu().numpy()
        doc_embedding = doc_embedding / np.linalg.norm(doc_embedding)
        doc_embeddings.append(doc_embedding)

    doc_embeddings = np.array(doc_embeddings, dtype=np.float32)

    # Tính cosine similarity giữa query và từng tài liệu
    cosine_similarities = np.dot(doc_embeddings, query_embedding)

    # Kết hợp điểm BM25 và AIA
    bm25_scores_selected = bm25_scores[top_25_idx]
    combined_scores = 0.2 * bm25_scores_selected + 0.8 * cosine_similarities

    # Chọn ra 10 tài liệu tốt nhất
    top_10_idx = np.argsort(combined_scores)[::-1][:20]
    top_10_docs = [relevant_docs[i] for i in top_10_idx]
    top_10_doc_ids = [doc.id for doc in top_10_docs]

    # Lấy toàn bộ chunk của 10 tài liệu này
    relevant_chunks = load_chunks_by_documents(top_10_doc_ids)
    reranked_chunks = rerank_chunks(query_lower, relevant_chunks)
    top_chunks_content = [chunk[2] for chunk in reranked_chunks]

    if top_chunks_content:
        response_text = f"Dựa trên các văn bản sau:\n{'. '.join(top_chunks_content)}\nHãy trả lời câu hỏi: {query}."
        response = call_gpt_api(response_text)
        if not response:
            response = "Không thể truy xuất phản hồi từ GPT."
            return jsonify({ "response": response })
        return jsonify({
            "response": response if response else "Không tìm thấy văn bản nào liên quan."
        })
    else:
        return jsonify({
            "response": "Không tìm thấy văn bản nào liên quan."
        }), 200

if __name__ == "__main__":
    app.run(debug=True)