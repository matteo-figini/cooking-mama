# 📊 Cooking Mama - NLP Project

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)


## 🎯 Project Overview

This repository contains the code for the Natural Language Processing course project held by Prof. Carman, Academic Year 2024/2025, Politecnico di Milano.

The main task of this project was to choose a dataset and apply various NLP techniques to it: in our case, we chose the [RecipeNLG dataset](https://recipenlg.cs.put.poznan.pl/) due to its versatility.


## 🚀 Getting Started

### 🔧 Prerequisites & Installation

No system prerequisites are required.
To use the notebooks, you can clone the repository and run them on Google Colab or Kaggle Platform.

## 📋 Notebook Descriptions

### 🔍 Notebook 1: Preprocessing & Binary Classification
This notebook explores the structure and the content of the dataset with an EDA, analysing the recipes and the entities distribution.
Afterwards, we trained a Word2Vec model, allowing us to capture the semantic relationships between words in ingredients. Word2Vec learns to represent words as dense vectors, where words with similar meanings are positioned closer together in the vector space. This enables us to explore word similarities, identify common patterns, and improve various natural language processing tasks like recipe recommendation and ingredient analysis.
Then, we performed some clustering and relative analysis.
In the end, we trained a binary classifier to recognize between vegan and non-vegan recipes. By exploiting [Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct), we subsampled around 20k recipes and classified as 'vegan' or 'non-vegan', trained different classifiers, such as Logistic Regression, Support Vector Machine and Bidirectional LSTM, and tested around a test-set. Good improvements on accuracy and recall metrics are present when the dataset is balanced or a stronger model is employed.

### 📈 Notebook 2: Using Mistral LLM with RAG & Indexing
In this notebook, we use a large language model called [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3), to define a cooking chatbot and enhancing the capabilites of answering user's questions, based on the RecipeNLG dataset (on a subsampling of the dataset).
First, the model is tested without any additional comment; aside, different system prompts (polite or rude) are tested to demonstrate the importance of a strong, robust and polite prompt design in modern commercial LLMs.

After this, we applied a very simple version of RAG (Retrieval-Augmented Generation) and indexing, using the dataset we have provided. The sequence is the following:
1. First, the user ask a specific recipe
2. The model provide a response without external information sources
3. Then, the model takes the query and performs an indexing on the RecipeNLG sampled dataset, finding the most similar recipes (BM25/TF-IDF/DFRee)
4. The model retries the query, by inlining the top-1 recipe and see changes in the answer, then evaluate it.
The model generates the response combining its own knowledge, with external knowledge coming from the dataset and embedded in the question. Indexing allows to automatically retrieve the most similar recipes matching the request and get the results.

### 🎯 Notebook 3: Further Developments
In this notebook we tested several models working on the dataset:
- A text-to-image model is employed to produce images of recipes extracted from the dataset.
- A multi-agent system generates and refines cooking recipes using a pretrained LLM (GPT-4) and information retrieval techniques, using LangChain and LangGraph.
- In the end, we built a complete chatbot offering also a text-to-speech feature.

## 👥 Authors

- **Matteo Figini** - [GitHub](https://github.com/matteo-figini)
- **Riccardo Figini** - [GitHub](https://github.com/RiccardoFiginiST)
- **Caterina Motti** - [GitHub](https://github.com/mttcrn)
- **Simone Zacchetti** - [GitHub](https://github.com/SimoneZacchetti)
- **Samuele Forner** - [GitHub](https://github.com/samueleforner)

## 📄 License

This project is licensed under the MIT License.

## 📞 Contact

For questions or suggestions, feel free to contact us by opening an issue on GitHub!

---

⭐ If this project was helpful to you, consider giving it a star!