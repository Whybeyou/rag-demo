# -*- coding: utf-8 -*-
"""多模态 RAG 知识库问答 Demo（零第三方依赖）

链路（对应简历描述）：
  离线索引：加载 knowledge/ 下的 .md/.txt 文档 → 切块 → 每块算本地 bge-m3 稠密向量
            + 建立 BM25 稀疏索引（中文用 bigram 分词）→ 建立混合索引。
  在线检索：问题 → 向量检索 + BM25 检索 → RRF 融合排序 → top-k 拼上下文。
  生成问答：把检索到的资料拼进 prompt，交给 DeepSeek 生成答案（只依据资料，不编造）。
  增量缓存：文档内容算 MD5，变了才重新建索引；index.json 缓存向量，二次运行秒开。

运行：python rag_demo.py
"""
import hashlib
import json
import math
import os
import re
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(BASE_DIR, "knowledge")
INDEX_FILE = os.path.join(BASE_DIR, "index.json")

DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek-v4-pro"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embed"
OLLAMA_EMBED_MODEL = "bge-m3"


def load_env():
    path = os.path.join(BASE_DIR, "..", ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k, v.strip())


load_env()
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")


# ---------- 基础 HTTP ----------
def http_post_json(url, payload, headers=None, timeout=120):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers=headers or {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def embed(text):
    data = http_post_json(OLLAMA_EMBED_URL, {"model": OLLAMA_EMBED_MODEL, "input": text})
    return data["embeddings"][0]


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = (sum(x * x for x in a)) ** 0.5
    nb = (sum(x * x for x in b)) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


# ---------- 中文分词（bigram） + BM25 ----------
def tokenize(text):
    """中文无空格，用相邻双字 bigram 做 token，兼顾词组特征。"""
    text = re.sub(r"[\s，。！？、；：（）【】「」\"'.,!?;:()\[\]<>]+", "", text)
    return [text[i:i + 2] for i in range(len(text) - 1)]


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = [tokenize(d) for d in docs]
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in self.docs) / self.N if self.N else 0
        self.k1, self.b = k1, b
        self.df = {}
        for d in self.docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1

    def _idf(self, t):
        n = self.df.get(t, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1)

    def search(self, query, top_k=8):
        qt = tokenize(query)
        ranked = []
        for i, d in enumerate(self.docs):
            tf = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for t in qt:
                if t in tf:
                    score += self._idf(t) * (tf[t] * (self.k1 + 1)) / \
                        (tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            ranked.append((i, score))
        ranked.sort(key=lambda x: -x[1])
        return ranked[:top_k]


def rrf_fusion(a_ranked, b_ranked, k=60):
    """RRF 融合两路排序（BM25 + 向量），结果取并集按融合分重排。"""
    score = {}
    for rank, (idx, _) in enumerate(a_ranked):
        score[idx] = score.get(idx, 0) + 1 / (k + rank + 1)
    for rank, (idx, _) in enumerate(b_ranked):
        score[idx] = score.get(idx, 0) + 1 / (k + rank + 1)
    return sorted(score.items(), key=lambda x: -x[1])


# ---------- 文档加载 + 切块 ----------
def load_documents():
    docs = []
    for fn in sorted(os.listdir(KB_DIR)):
        if fn.endswith((".md", ".txt")):
            text = open(os.path.join(KB_DIR, fn), encoding="utf-8").read()
            docs.append({"file": fn, "text": text})
    return docs


def chunk_text(text, max_chars=400):
    chunks = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            chunks += [para[i:i + max_chars] for i in range(0, len(para), max_chars)]
    return chunks


# ---------- 索引（含 MD5 增量 + 缓存） ----------
def build_index(force=False):
    docs = load_documents()
    if not docs:
        raise SystemExit(f"knowledge/ 目录下没有文档，请先放入 .md/.txt 文件（当前目录：{KB_DIR}）")

    # 所有文档内容的 hash，变了才重建
    digest = hashlib.md5("".join(d["text"] for d in docs).encode("utf-8")).hexdigest()

    if not force and os.path.exists(INDEX_FILE):
        cached = json.load(open(INDEX_FILE, encoding="utf-8"))
        if cached.get("digest") == digest:
            print(f"✅ 命中索引缓存（{len(cached['chunks'])} 块），无需重建")
            return cached

    print("🔨 重建索引：切块 + bge-m3 向量化 + BM25 ...")
    chunks, sources = [], []
    for d in docs:
        for c in chunk_text(d["text"]):
            chunks.append(c)
            sources.append(d["file"])

    vectors = [embed(c) for c in chunks]
    index = {"digest": digest, "chunks": chunks, "sources": sources, "vectors": vectors}
    json.dump(index, open(INDEX_FILE, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"✅ 索引完成：{len(chunks)} 块")
    return index


# ---------- 检索 ----------
def retrieve(query, index, top_k=4):
    v = embed(query)

    # 向量路
    vec_ranked = sorted(
        enumerate(index["vectors"]),
        key=lambda iv: cosine(iv[1], v), reverse=True)[:8]

    # BM25 路
    bm25 = BM25(index["chunks"])
    bm25_ranked = bm25.search(query, top_k=8)

    # RRF 融合
    fused = rrf_fusion(bm25_ranked, vec_ranked)[:top_k]
    hits = []
    for idx, _ in fused:
        hits.append({"chunk": index["chunks"][idx], "source": index["sources"][idx]})
    return hits


# ---------- 生成 ----------
def answer(query, hits):
    context = "\n\n".join(f"[{h['source']}]\n{h['chunk']}" for h in hits)
    messages = [
        {"role": "system", "content":
            "你是个人知识库问答助手。只根据下面提供的资料回答问题，"
            "资料里没有的信息就明确说「资料中没有」，绝不编造。回答末尾标注引用来源。"},
        {"role": "user", "content": f"资料：\n{context}\n\n问题：{query}"},
    ]
    resp = http_post_json(DEEPSEEK_URL, {"model": DEEPSEEK_MODEL, "messages": messages},
                          headers={"Content-Type": "application/json",
                                   "Authorization": f"Bearer {API_KEY}"})
    return resp["choices"][0]["message"]["content"]


# ---------- 交互 ----------
def main():
    if not API_KEY:
        raise SystemExit("请先设置 DEEPSEEK_API_KEY（E:\\agent-learning\\.env）")
    index = build_index()
    print("\n个人知识库问答已就绪。输入问题回车，输入 quit 退出。\n")
    while True:
        q = input("🙋 你问：").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit", "退出"):
            break
        hits = retrieve(q, index)
        print("\n📚 命中的资料片段：")
        for i, h in enumerate(hits, 1):
            print(f"  {i}. [{h['source']}] {h['chunk'][:60]}...")
        print("\n🤖 回答：")
        print(answer(q, hits))
        print("-" * 60)


if __name__ == "__main__":
    main()
