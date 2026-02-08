
# 📰 Text Classification with Multiple Embeddings and Models (AG News)

This repository contains a comparative study of **text classification models** paired with different **word embedding techniques** on the **AG News** dataset. The goal is to analyze how representation choices (TF-IDF, Word2Vec CBOW, Word2Vec Skip-gram) interact with different model architectures and how these combinations affect performance in a **4-class news topic classification** task: **World, Sports, Business, Sci/Tech**.

The project was developed as a **group assignment**, where each member implemented and evaluated one model family:

* **Traditional ML**: Logistic Regression
* **Neural Models**: RNN, LSTM, and GRU

Each model is evaluated using:

* **TF-IDF**
* **Word2Vec CBOW**
* **Word2Vec Skip-gram**

---

## 📊 Dataset

We use the **AG News Topic Classification Dataset**, which contains:

* **120,000 training samples**
* **7,600 test samples**
* **4 balanced classes**: World, Sports, Business, Sci/Tech

Each sample consists of a **title + description**, which are concatenated into a single text input during preprocessing.

Why AG News?

* Balanced and sufficiently large for neural models
* Standard benchmark in text classification
* Fixed train/test split for fair comparison

---

## 🧠 Models Implemented

* **Logistic Regression (Baseline)**
* **Recurrent Neural Network (RNN)**
* **Long Short-Term Memory (LSTM)**
* **Gated Recurrent Unit (GRU)**

Each model is trained and evaluated with:

* **TF-IDF features**
* **Word2Vec CBOW embeddings**
* **Word2Vec Skip-gram embeddings**

Neural models use:

* Embedding dimension: 100
* Sequence length: up to 100 tokens
* Dropout for regularization
* Adam optimizer
* Early stopping to prevent overfitting

---

## 🔧 Preprocessing

A **shared preprocessing pipeline** is used for fair comparison:

* Lowercasing
* Removing special characters
* Normalizing whitespace
* Concatenating title and description
* Tokenization and padding (for neural models)

Notes:

* **Stopwords are not removed** for Word2Vec to preserve contextual learning
* **TF-IDF** uses unigram + bigram features
* For sequence models, TF-IDF vectors are reshaped to fit the input format

---

## 📈 Evaluation Metrics

All models are evaluated on the **test set** using:

* **Accuracy**
* **Macro F1-score**
* **Precision & Recall**
* **Confusion Matrices**

We also include:

* Training curves (accuracy & loss)
* Hyperparameter tuning experiments (especially for GRU)
* Per-class F1-score analysis

---

## 🏆 Key Findings (Summary)

* **Logistic Regression + TF-IDF** performs strongest among traditional models.
* **GRU + Skip-gram** achieves the **best overall performance** among GRU variants.
* **Dense embeddings (CBOW, Skip-gram)** outperform TF-IDF when used with sequence models.
* **TF-IDF + LSTM (with SVD)** performs surprisingly well, showing that strong baselines still matter.
* Most confusion occurs between **Business** and **Sci/Tech** due to overlapping vocabulary.

---

## 📁 Repository Structure (Suggested)

```bash
.
├── notebooks/
│   ├── logistic_regression.ipynb
│   ├── rnn.ipynb
│   ├── lstm.ipynb
│   └── gru.ipynb
├── data/
│   └── ag_news.csv
├── Group_10_Report.pdf
└── README.md
```

---

## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/lemwaizz/Group_10_formative.git
cd Group_10_formative
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open the notebooks:

```bash
jupyter notebook
```

4. Run the notebook for the model you want to reproduce:

* `gru.ipynb`
* `lstm.ipynb`
* `rnn.ipynb`
* `logistic_regression.ipynb`

---

## 👥 Team Contributions

* **Wengelawit Solomon** – Traditional ML (Logistic Regression)
* **Dennis Mwai** – GRU
* **Hirwa Blessing** – LSTM
* **Jean Jabo** – RNN

---

## 📚 Reference

For full methodology, experiments, tables, and discussion, see the full report:
**Group_10_Report.pdf** 


