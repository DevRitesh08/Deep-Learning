# NLP End-to-End Classification Projects

This directory contains end-to-end practical Natural Language Processing projects covering data cleaning, exploratory data analysis, vectorization techniques, leakage-free pipelines, and machine learning classification algorithms.

---

## Projects Overview

| Project | Dataset | Techniques Covered | Models Evaluated |
|---|---|---|---|
| **[01. Spam Ham Classification (BoW & TF-IDF)](./01_spam_ham_classification_bow_and_tfidf.ipynb)** | SMS Spam Collection | EDA, Lemmatization, Train-Test Split (Data Leakage Prevention), N-grams (1, 2), CountVectorizer, TfidfVectorizer | Multinomial Naive Bayes |
| **[02. Spam Ham Classification (Word2Vec)](./02_spam_ham_classification_word2vec.ipynb)** | SMS Spam Collection | Custom Word2Vec (Gensim CBOW), Average Word2Vec vectorization, empty sentence vector alignment | Random Forest Classifier |
| **[03. Kindle Review Sentiment Analysis](./03_kindle_review_sentiment_analysis.ipynb)** | Kindle Store Book Reviews | Rating polarity mapping, review text preprocessing, Lemmatization, CountVectorizer, TfidfVectorizer, Custom Word2Vec & Average Word2Vec | Multinomial Naive Bayes, Logistic Regression |

---

## Key Best Practices Followed

1. **Data Leakage Prevention**: Train-test split is performed *before* vectorization (`fit_transform` on train set only, `transform` on test set).
2. **Dense & Sparse Vector Handling**: Handling empty sentence vectors when filtering stopwords in Word2Vec pipelines to keep features and labels properly aligned.
3. **Comprehensive Evaluation**: Performance is measured with accuracy scores, precision/recall/F1 classification reports, and confusion matrix heatmaps.
