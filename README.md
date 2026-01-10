# Web Mining UCLouvain FUCaM Mons 2025-2026
### Study of the promotion of sustainable products in committed e-commerce stores

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-yellow)
![Course](https://img.shields.io/badge/Course-Web_Mining-orange)

**Course:** Web Mining (MLSMM2153)  
**Institution:** Louvain School of Management – UCLouvain  
**Academic Year:** 2025–2026  

**Authors:**  
- Clément Feron  
- Romain Nicelli  
- Mathias Simoens  

**Supervisors:**  
- Corentin Vande Kerckhove  
- Sylvain Courtain  

**GitHub repository:**  
https://github.com/simoensm/web_mining_ucl_2025  

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Theme](#project-theme)
- [Research Questions](#research-questions)
- [Data Sources](#data-sources)
- [Methodology](#methodology)
  - [Web Scraping](#web-scraping)
  - [Text Mining](#text-mining)
  - [Link Analysis](#link-analysis)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Scripts Index](#scripts-index)
- [Key Results](#key-results)
- [Roadmap & Future Work](#roadmap--future-work)
- [Limitations](#limitations)
- [Academic Context & Disclaimer](#academic-context--disclaimer)

---

## Project Overview

This repository contains the complete codebase developed for the **Web Mining (MLSMM2153)** course at UCLouvain.  
The project applies automated web mining techniques to analyze **how sustainable fashion brands communicate sustainability through their online product descriptions**.

It relies on the three core methodological pillars taught in the course:

1. **Scraping** – automated extraction of web data  
2. **Text Mining** – NLP, TF-IDF, clustering, semantic similarity  
3. **Link Analysis** – graph construction and network metrics  

---

## Project Theme

The project studies the **linguistic and semantic construction of sustainability discourse** in ethical e-commerce.  
It investigates whether sustainability is framed as:

- a technical performance attribute  
- a moral or ESG-driven narrative  
- or a secondary certification-based argument  

---

## Research Questions

1. Which semantic markers are mobilized by sustainable fashion brands to promote their products?
2. What is the semantic distance between institutional ESG discourse and product-level descriptions?
3. Can semantic similarity between products be modeled as a graph revealing central and bridge products?

---

## Data Sources

The corpus is built from three international sustainable fashion brands:

- **Patagonia** (USA) – technical and environmental activism  
- **Ecoalf** (Spain) – circular economy and recycling  
- **Armedangels** (Germany) – slow fashion and social certifications  

All data was collected from **English-language versions** of the websites.

---

## Methodology

### Web Scraping

Scraping was performed using real browser automation to bypass anti-bot protections.

**Main technologies:**
- `playwright` (browser simulation)
- `asyncio`
- `BeautifulSoup`
- `pandas`

**Collected elements:**
- product URLs
- product names
- breadcrumbs (categories)
- sustainability and technical descriptions
- ESG reports (PDF → text)

---

### Text Mining

The text mining pipeline includes:

- text cleaning and normalization
- stopword filtering (NLTK + custom lists)
- lemmatization
- unigram and bigram tokenization
- TF-IDF vectorization
- k-Means clustering (elbow method & silhouette score)
- PCA for visualization
- cosine similarity
- lexical dictionary analysis (ESG / Materials / Technical)

---

### Link Analysis

Products are modeled as nodes in a similarity graph:

- edges weighted by cosine similarity
- thresholded adjacency matrix
- metrics:
  - degree centrality
  - shortest path
  - betweenness centrality

Graph visualization is performed using **Gephi**.

---

## Repository Structure

```text
web_mining_ucl_2025/
│
├── scraping
│   ├── .patagonia/
│   │   ├── patagonia_links_men.py
│   │   ├── patagonia_links_women.py
│   │   ├── patagonia_scraper.py
│   │   └── patagonia_reports.py
│   │
│   ├── .ecoalf/
│   │   ├── ecoalf_links_men.py
│   │   ├── ecoalf_links_women.py
│   │   └── ecoalf_scraper.py
│   │
│   └── .armedangels/
│       ├── armedangels_links_men.py
│       ├── armedangels_links_women.py
│       └── armedangels_scraper.py
│
├── main
│   ├── text_mining_main.py
│   ├── text_mining_report.py
│   ├── text_mining_voc.py
│   ├── link_analysis_main.py
│   └── others/
│       ├── corpus_cleaning.py
│       ├── excel_compil.py
        ├── volume_scrapping.py
        ├── word_freq_comparison.py
        └── silhouette_calcul.py
rapport_MLSMM2153_group_6.pdf
EADME.md
```

---

## Getting Started

### Installation

```bash
Python 3.9+
pip install playwright pandas numpy nltk scikit-learn matplotlib
playwright install
```

### Typical Workflow

```bash
# Scraping
python scraping/patagonia/patagonia_links_men.py
python scraping/patagonia/patagonia_scraper.py

# Text mining
python text_mining/text_mining_main.py

# Link analysis
python link_analysis/link_analysis_main.py
```

---

## Scripts Index

| Script | Purpose |
|------|--------|
| `*links_*.py` | Collect product URLs |
| `*products_.py` | Extract product descriptions |
| `corpus_cleaning.py` | Text cleaning |
| `text_mining_main.py` | Text Mining on product descriptions |
| `text_mining_voc.py` | Supervised Classification based on product descriptions |
| `text_mining_report.py` | Text Mining on ESG reports |
| `link_analysis_main.py` | Network metrics |
| `excel_compil.py` | Excel Compilation |

---

## Key Results

- Three distinct sustainability communication strategies identified
- ESG-product alignment varies strongly across brands
- Semantic similarity graphs reveal central and bridge products

---

## Limitations

- Lexical and statistical analysis only
- No verification of sustainability claims
- Results sensitive to editorial style

---

## Academic Context & Disclaimer

This repository was developed **strictly for academic purposes** within the Web Mining course at UCLouvain.  
It does not aim to evaluate brands ethically nor validate sustainability claims.
