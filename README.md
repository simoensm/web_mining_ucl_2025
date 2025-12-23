# web_mining_ucl_2025
NLP & Sustainability: Measuring the Semantic Gap between Marketing Claims and ESG Reporting.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-In_Progress-yellow)
![Course](https://img.shields.io/badge/Course-Web_Mining-orange)

## Context and Objectives

This project is carried out as part of the **Web Mining (MLSMM2153)** course. As consultants, our mission is to audit the communication strategy of "committed" online stores.

**Subject 6: Study of the promotion of sustainable products.**

The objective is to confront the marketing vocabulary used on product pages with the reality of the company's commitments (ESG/CSR) using a "Data-Driven" approach.

### Key Questions
1. How do ethical brands semantically value their products?
2. Is there consistency between product categories and sustainable promises (Link Analysis)?
3. Is the marketing discourse aligned with sustainability reports?

---

## Project Architecture

The project follows a strict three-step data pipeline:

### 1. Data Collection (Scraping)
* **Targets:** Sustainable e-commerce sites and institutional websites.
* **Method:** Automated navigation and HTML parsing.
* **Strategy:** Targeted entry points to avoid noise and ensure a representative corpus.
* **Output:** Structured data (JSON/CSV).

### 2. Text Mining (NLP)
Transforming the raw corpus into semantic insights.
* **Preprocessing:** Tokenization, Filtering (Stopwords), Lemmatization/Stemming.
* **Vectorization:** TF-IDF / Doc2Vec.
* **Analyses:**
    * *Descriptive:* Word clouds, n-grams.
    * *Semantic:* Sentiment analysis/tonality.
    * *Clustering:* Unsupervised grouping of product descriptions.

### 3. Link Analysis (Graph Theory)
Modeling relationships between products and categories.
* **Graph Construction:** Nodes (Products/Categories) and Edges (Similarity links or navigation).
* **Metrics:** Degree Centrality, PageRank, Betweenness.
* **Objective:** Identify "bridge" products and the structure of the sustainable offer.
