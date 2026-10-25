# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"E:\agent-learning\rag-demo")
import rag_demo

index = rag_demo.build_index()
for q in ["什么是 RAG？", "Agent 的任务爆炸问题怎么解决？", "本机有哪些 embedding 模型？"]:
    hits = rag_demo.retrieve(q, index)
    print("\n" + "=" * 60)
    print("问题：", q)
    print("命中片段：")
    for h in hits:
        print("  -", h["source"], "|", h["chunk"][:50])
    print("回答：")
    print(rag_demo.answer(q, hits))
