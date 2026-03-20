# PySpark Big Data & Recommendation Systems 🚀

A comprehensive collection of distributed computing implementations for large-scale data analysis, text mining, and recommendation engines using **Apache Spark (PySpark)**. This repository covers advanced algorithms for similarity search, graph theory, and collaborative filtering.

## 🌟 Key Project Modules

### 1. Recommendation Engines 🤖
* **Content-Based Filtering:** Built a recommender using **TF-IDF** vectorization and **Cosine Similarity** to suggest movies based on genre and metadata.
* **Collaborative Filtering:** Implemented **User-Based Collaborative Filtering** with Pearson Correlation. Includes an optimized "Fast" version to handle memory constraints and improve execution time.
* **Explainability:** Integrated **SHAP** (SHapley Additive exPlanations) to provide transparency into recommendation logic.

### 2. Scalable Similarity Search 🔍
* **MinHash & LSH:** Developed a Locality Sensitive Hashing pipeline to solve the "Near Neighbor" problem in high-dimensional datasets (MovieLens 100K).
* **S-Curve Analysis:** Mathematical modeling and visualization of LSH thresholds to balance False Positives and False Negatives.
* **K-Gram Generators:** Tools for character and word-level k-gram extraction for document similarity.

### 3. Text Mining & NLP 📚
* **Gutenberg Dataset Pipeline:** Automated extraction of metadata (Title, Language, Release Date) from thousands of raw text files.
* **Distributed TF-IDF:** A scalable pipeline for calculating term importance across a massive corpus of books.
* **Data Cleaning:** Regex-based cleaning to strip headers, footers, and boilerplate from Project Gutenberg texts.

### 4. Graph Analytics 🕸️
* **Influence Networks:** Used **GraphFrames** to model relationships between authors.
* **Centrality Measures:** Implemented **PageRank**, In-Degree, and Out-Degree algorithms to identify key influencers within a social or citation network.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Framework:** PySpark (Spark SQL, MLlib, GraphFrames)
* **Math & Data:** NumPy, Pandas, Scipy
* **Visualization:** Matplotlib, NetworkX
* **ML Explainability:** SHAP

## 📊 Evaluation Metrics
The models are rigorously tested using:
* **Precision @ K & Recall:** For ranking quality.
* **RMSE (Root Mean Square Error):** For rating prediction accuracy.
* **Jaccard Similarity:** For set-based overlap analysis.

## 🚀 Getting Started

### Prerequisites
Ensure you have a Spark environment configured.
```bash
pip install pyspark numpy matplotlib pandas graphframes shap
