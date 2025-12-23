# web_mining_ucl_2025
NLP &amp; Durabilité : Mesure de la distance sémantique entre promesse client et reporting corporate.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-Completed-success)
![Course](https://img.shields.io/badge/Course-Web_Mining-orange)

## 📖 Contexte et Objectifs

Ce projet s'inscrit dans le cadre du cours de **Web Mining (MLSMM2153)**. En tant que consultants, notre mission est d'auditer la stratégie de communication des boutiques en ligne "engagées".

**Sujet 6 : Étude de la mise en avant des produits durables.**

L'objectif est de confronter le vocabulaire marketing utilisé sur les fiches produits avec la réalité des engagements (ESG/RSE) via une approche "Data-Driven".

### Questions clés
1. Comment les marques éthiques valorisent-elles sémantiquement leurs produits ?
2. Existe-t-il une cohérence entre les catégories de produits et les promesses durables (Link Analysis) ?
3. Le discours marketing est-il aligné avec les rapports de durabilité ?

---

## ⚙️ Architecture du Projet

Le projet suit un pipeline de données strict en trois étapes :

### 1. Collecte de Données (Scraping)
* **Cibles :** Sites e-commerce durables et institutionnels.
* **Méthode :** Navigation automatisée et parsing HTML.
* **Stratégie :** Points d'entrées ciblés pour éviter le bruit et garantir un corpus représentatif.
* **Output :** Données structurées (JSON/CSV).

### 2. Text Mining (NLP)
Transformation du corpus brut en insights sémantiques.
* **Preprocessing :** Tokenisation, filtrage (Stopwords), Lemmatisation/Stemming.
* **Vectorisation :** TF-IDF / Doc2Vec.
* **Analyses :**
    * *Descriptive :* Nuages de mots, n-grams.
    * *Sémantique :* Analyse de sentiments/tonalité.
    * *Clustering :* Groupement non supervisé des descriptions produits.

### 3. Link Analysis (Graphes)
Modélisation des relations entre produits et catégories.
* **Construction du graphe :** Nœuds (Produits/Catégories) et Arêtes (Liens de similarité ou navigation).
* **Métriques :** Degree Centrality, PageRank, Betweenness.
* **Objectif :** Identifier les produits "ponts" et la structure de l'offre durable.

---

## 📂 Structure du Répertoire

```bash
├── data/
│   ├── raw/            # Données brutes issues du scraping (HTML, JSON brut)
│   └── processed/      # Données nettoyées et prêtes pour l'analyse (CSV, Graphes)
├── notebooks/          # Notebooks Jupyter d'exploration et de visualisation
│   ├── 1_scraping.ipynb
│   ├── 2_text_mining.ipynb
│   └── 3_link_analysis.ipynb
├── src/                # Scripts Python réutilisables
│   ├── scraper.py      # Script de collecte
│   ├── preprocessing.py # Fonctions de nettoyage NLP
│   └── utils.py
├── reports/            # Graphes générés et rapport PDF final
├── requirements.txt    # Dépendances du projet
└── README.md           # Documentation du projet
