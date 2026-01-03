# web_mining_ucl_2025
NLP & Sustainability: Measuring the Semantic Gap between Marketing Claims and ESG Reporting.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-In_Progress-yellow)
![Course](https://img.shields.io/badge/Course-Web_Mining-orange)

## Context and Objectives
This project is carried out as part of the Web Mining (MLSMM2153) course. As consultants, our mission is to audit the communication strategy of "committed" online stores.

## Subject 6: Study of the promotion of sustainable products.

The objective is to confront the marketing vocabulary used on product pages with the reality of the company's commitments (ESG/CSR) using a Data-Driven approach.

## Key Questions
- Semantic Valuation: How do ethical brands semantically value their products?
- Link Analysis: Is there consistency between product categories and sustainable promises?
- Alignment: Is the marketing discourse aligned with sustainability reports?

## Project Architecture
The project follows a strict three-step data pipeline:

1) Data Collection (Scraping): Automated navigation and HTML parsing of sustainable e-commerce sites and institutional websites.
2) Text Mining (NLP): Transforming the raw corpus into semantic insights via TF-IDF, clustering, and sentiment analysis.
3) Link Analysis (Graph Theory): Modeling relationships between products using similarity metrics to identify "bridge" products and offer structure.

## Installation and Requirements
To run the pipeline, ensure you have the following Python libraries installed:

pip install (pandas numpy nltk scikit-learn matplotlib wordcloud pypdf openpyxl)
Note: The scripts utilize NLTK resources (stopwords, punkt, wordnet). The scripts are designed to download these resources automatically if they are not found.

## Usage Pipeline
Follow the steps below to replicate the analysis.

1. Scraping and Data Collection

This stage involves harvesting data from websites and converting manual reports into usable text.
   A. Web Scraping Three specific websites are targeted. Each website has a dedicated folder containing two primary scripts:

   - Link Harvesting (links_site.py): Run this first. It scrapes the target category page to harvest all product URLs.
         Input: Category page URL.
         Output: A text file (e.g., site_men_links.txt).
     
   - Product Scraping (products_site.py): Run this second. It iterates through the text file generated in the previous step to scrape specific product details.
         Input: The links text file.
   - Output: A raw Excel file (e.g., site_men_products.xlsx).
     Note: Ensure you manually update the category variable (e.g., changing from 'men' to 'women') in the script before execution.

   B. PDF Report Conversion Used for processing ESG/CSR reports manually downloaded as PDFs.

   - Script: pdf_to_txt_patagonia.py
   - Functionality: This script reads PDF files, repairs broken text spacing (e.g., converting "T o h e l p" to "To help"), removes non-alphabetic characters, and filters stopwords.
   - Output: A clean .txt file ready for text mining.

C. Data Compilation (Optional)

Script: excel_compil.py

Functionality: If performing cross-site analysis, use this script to merge multiple product Excel files (.xlsx) or CSV files into a single master dataset.

2. Data Cleaning
Before analysis, the raw product data must be standardized.

Script: corpus_cleaning.py

Input: The raw product Excel file (e.g., .patagonia/patagonia_products.xlsx).

Functionality:

Standardizes column names to lowercase.

Removes exact duplicates based on product names.

Deep Cleaning: Removes numbers, special characters, content inside brackets, and converts text to lowercase.

Output: A clean dataset (e.g., patagonia_dataset.xlsx).

3. Text Mining (NLP)
This module extracts semantic insights from the cleaned data.

A. Unsupervised Product Clustering

Script: text_mining_main.py

Input: The cleaned dataset from Step 2.

Functionality:

Preprocessing: Tokenization, Lemmatization (or Stemming), and stopword filtering.

Vectorization: Calculates a TF-IDF matrix manually (TF * IDF).

Analysis: Uses the Elbow method to determine the optimal number of clusters, performs K-Means clustering, and visualizes results using PCA.

Output: An Excel file enriched with a cluster column.

B. Supervised Vocabulary Analysis

Script: text_mining_voc.py

Functionality: Analyzes product descriptions based on pre-defined lexicons:

ESG: (e.g., fair trade, carbon, organic).

Technical: (e.g., waterproof, durable, gore-tex).

Materials: (e.g., polyester, hemp, nylon).

Output: Assigns membership scores to products, allowing for comparison of vocabulary usage across different brands.

C. Report Analysis

Script: text_mining_text_files.py

Input: Text files generated from PDFs (Step 1B).

Functionality: Performs n-gram analysis (unigram, bigram, trigram) and generates word clouds directly from raw text files to analyze corporate reporting discourse.

4. Link Analysis
This step prepares the data for network visualization in Gephi.

Script: gephi_convert.py

Input: The clustered Excel dataset and the TF-IDF matrix from Step 3A.

Functionality:

Nodes: Generates a list of nodes representing products (ID, Label, Cluster).

Edges: Calculates the Cosine Similarity between all products. Edges are created only if the similarity exceeds a defined threshold (default: 0.20).

Output: Two CSV files (nodes.csv and edges.csv) importable into Gephi.
