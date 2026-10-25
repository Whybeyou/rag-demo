# 个人知识库 RAG 问答系统

从零实现的 RAG（检索增强生成）问答系统，**纯 Python 标准库**，不依赖 LangChain / Chroma 等框架。
覆盖「文档解析 → 切块 → 混合索引 → 检索融合 → 生成问答」完整链路，支持 Markdown / 文本知识库。

## 核心特性

- **混合索引**：`bge-m3` 稠密向量 + 手写 BM25 稀疏索引（中文用 bigram 分词），兼顾语义匹配与关键词精确匹配。
- **检索融合**：BM25 与向量双路召回，用 RRF（倒数排名融合）重排序取 top-k，比单路检索命中更准。
- **增量缓存**：文档内容算 MD5 指纹，内容不变直接复用索引缓存（`index.json`），二次启动秒级加载，避免重复向量化。
- **引用检索与防编造**：检索到的资料拼入 prompt，约束模型只依据资料作答并标注来源，杜绝幻觉。

## 链路

```
离线索引：文档(.md/.txt) → 切块 → bge-m3 向量化 + BM25 索引 → index.json（MD5 缓存）
在线检索：问题 → 向量检索 + BM25 检索 → RRF 融合 → top-k 片段
生成问答：top-k 片段拼入 prompt → DeepSeek 生成答案（带引用、不编造）
```

## 快速开始

前置条件：

1. Python 3（仅标准库，无第三方依赖）
2. DeepSeek API Key（写入 `E:\agent-learning\.env` 的 `DEEPSEEK_API_KEY`）
3. 本地 Ollama 已运行，并拉取 `bge-m3`：`ollama pull bge-m3`
4. 往 `knowledge/` 目录放几份 `.md` 或 `.txt` 文档（已内置 3 份示例）

运行（交互式问答）：

```bash
python rag_demo.py
```

自动测试（内置 3 个问题）：

```bash
python test_rag.py
```

## 运行效果

```
🔨 重建索引：切块 + bge-m3 向量化 + BM25 ...
✅ 索引完成：33 块

问题：什么是 RAG？
命中片段：
  - rag.md | ## RAG 是什么
  - rag.md | RAG（Retrieval-Augmented Generation...）
回答：
RAG（检索增强生成）是一种把"检索"和"生成"结合起来的方法：先从知识库检索
相关资料，再把资料拼进 prompt 交给 LLM 生成答案，能减少幻觉、让回答有据可依。
来源：[rag.md]
```

## 目录结构

```
rag-demo/
├── rag_demo.py      # 主程序：索引 + 检索 + 问答
├── test_rag.py      # 自动测试（3 个问题）
├── knowledge/       # 知识库文档（.md / .txt）
│   ├── ai-agent.md
│   ├── rag.md
│   └── local-llm.md
└── index.json       # 索引缓存（运行时生成，已 gitignore）
```

## 技术栈

- Python 3（纯标准库：`urllib` / `json` / `math` / `hashlib`）
- 本地 Ollama（`bge-m3` embedding，1024 维）
- DeepSeek API（生成问答）
- 手写 BM25 + RRF 融合
