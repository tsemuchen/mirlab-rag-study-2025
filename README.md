# Iterative RAG with Citation-Aware Mechanism

**Research project at Mirlab, NTU (2025)**  
*Supervised by Prof. Jyh-Shing Roger Jang*  

---

## Introduction
This repository contains code and experiments for improving **Retrieval-Augmented Generation (RAG) systems** through:  
1. **Iterative RAG** – multi-iteration retrieval where answers from one iteration refine the query for the next.  
2. **Citation-Aware Mechanism** – prompting the LLM to cite document IDs when using retrieved facts, enhancing **traceability** and **accountability**.  
3. **Query Expansion (Appending)** – appending answers from previous iterations to the original query to improve retrieval context.  

The project focuses on **multi-hop question answering** using cleaned and truncated versions of the **MusiQue dataset** (limited to 100 questions per category), evaluating how iteration, citation, and query expansion affect retrieval and reasoning quality.  

---

## Methodology
- **Iteration Framework**:  
  - Iteration 1: Query → Retrieval → Answer  
  - Iteration 2: Append Iteration 1 answer → New Query → Retrieval → Refined Answer  
  - Iteration 3+: Extended iterations for diminishing returns analysis  

- **Citation-Aware Prompting**:  
  - Forces LLM to cite document IDs alongside factual claims.  

- **Query Expansion (Appending)**:  
  - Implements a straightforward strategy by **appending the previous iteration’s answer to the original query** before retrieval.  

- **Datasets**:  
  - MusiQue multi-hop QA (public dataset, further cleaned and truncated to 100 questions per category at Mirlab).  

- **Metrics**:  
  - NDCG  
  - Recall  
  - Filtered NDCG  
  - Retrieval size  
  - Runtime efficiency  

---

## Acknowledgments
- Some files are **modified from [python_rag](https://github.com/telunyang/python_rag)**, developed by [Telun Yang](https://github.com/telunyang).

- Experiments are based on the **[MusiQue dataset](https://arxiv.org/abs/2108.00573)**, truncated to 100 questions per category.  

- Special thanks to **Prof. Jyh-Shing Roger Jang** for supervision and guidance.  
