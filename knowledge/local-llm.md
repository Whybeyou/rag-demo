# 本地大模型部署笔记

## Ollama 部署

Ollama 是一个本地运行大模型的工具。常用命令：
- ollama pull qwen2.5:7b 拉取模型
- ollama run qwen2.5:7b 运行模型
- ollama list 查看已装模型

## Embedding 模型

Embedding 模型把文本转成向量，用于语义相似度计算。本机常用的两个 embedding 模型：
- bge-m3：多语言（含中文），1024 维向量，语义效果好
- nomic-embed-text：轻量，适合快速测试

弱 embedding 模型靠字面重叠判断相似，会误判；强模型（如 bge-m3）学的是语义和别名关系。

## 余弦相似度

两个向量的相似度用余弦相似度衡量，范围 0 到 1，越接近 1 越相似。计算公式：点积除以两个向量长度的乘积。

## DeepSeek API

DeepSeek 提供 OpenAI 兼容接口，可以直接用 urllib 或 requests 调用 chat/completions 接口，支持 Function Calling（工具调用）。模型 ID 有 deepseek-v4-pro 和 deepseek-flash。
