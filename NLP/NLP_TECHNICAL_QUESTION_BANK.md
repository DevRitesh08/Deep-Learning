# NLP Core Concepts & Technical Question Bank

> **Domain:** Natural Language Processing & Text Engineering  
> **Structure:** In-depth technical questions, mathematical derivations, intuitive mental models, and production trade-offs.  
> **Coverage:** Text Preprocessing, Tokenization, Lexical Vectorization (OHE, BoW, N-Grams, TF-IDF), Distributed Representations (Word2Vec, CBOW, Skip-Gram), Document Embeddings (Average Word2Vec), and Applied Pipeline Architectures.

---

## Table of Contents
1. [Pillar 1: Text Preprocessing, Normalization & Tokenization](#1-pillar-1-text-preprocessing-normalization--tokenization)
2. [Pillar 2: Classical Lexical Vectorization (OHE, BoW, N-Grams, TF-IDF)](#2-pillar-2-classical-lexical-vectorization-ohe-bow-n-grams-tf-idf)
3. [Pillar 3: Distributed Word Embeddings & Geometric Semantics](#3-pillar-3-distributed-word-embeddings--geometric-semantics)
4. [Pillar 4: Word2Vec Architectural Mechanics (CBOW vs Skip-Gram)](#4-pillar-4-word2vec-architectural-mechanics-cbow-vs-skip-gram)
5. [Pillar 5: Document Embeddings, FastText, GloVe & The Polysemy Problem](#5-pillar-5-document-embeddings-fasttext-glove--the-polysemy-problem)
6. [Pillar 6: Production Pipeline Architecture, Diagnostics & Evaluation](#6-pillar-6-production-pipeline-architecture-diagnostics--evaluation)

---

## 1. Pillar 1: Text Preprocessing, Normalization & Tokenization

### Q1: What is the fundamental difference between Word Tokenization, Character Tokenization, and Subword Tokenization (BPE/WordPiece)?
**Quick Take:**  
Word tokenization splits on spaces/punctuation (creating huge vocabularies and failing on out-of-vocabulary words). Character tokenization splits every character (eliminating out-of-vocabulary words but exploding sequence lengths and destroying semantic chunks). Subword tokenization (Byte-Pair Encoding, WordPiece) dynamically splits frequent words as single units and rare words into sub-morphemic pieces (`"unhappily"` $\to$ `["un", "happi", "ly"]`), delivering a compact fixed vocabulary that handles any arbitrary word.

**Deep Dive:**
- **Word Tokenization:** Vocabulary $|V| \approx 50,000 - 500,000$. Any unseen word or typo becomes `<UNK>`.
- **Character Tokenization:** $|V| \approx 100 - 256$. Solves `<UNK>` entirely, but an average 50-word sentence becomes 250+ tokens, straining the quadratic $O(L^2)$ attention limits of transformer models and losing token-level semantics.
- **Subword Tokenization (BPE / WordPiece):** Uses statistical frequency to iteratively merge character pairs into common fragments:
  - Frequent words stay intact: `"the"`, `"learning"`, `"vector"`.
  - Rare/technical/misspelled words decompose into roots and affixes: `"electromagnetism"` $\to$ `["electro", "##magnet", "##ism"]`.
  - Vocabulary is strictly bounded (typically $30,000 - 50,000$ tokens), completely eliminating out-of-vocabulary failures while keeping sequence lengths manageable.

---

### Q2: Why is Stemming considered a heuristic rule-based approach while Lemmatization is algorithmic and lexicon-based?
**Quick Take:**  
Stemming applies crude string-chopping rules (like stripping `-ing`, `-es`, `-ly`) without knowledge of language grammar or dictionary existence, often generating invalid words (`"studi"`, `"histori"`). Lemmatization uses morphological analysis and a lexical database (like WordNet) to resolve a word back to its legitimate canonical root (its *lemma*), guaranteed to be a real dictionary word (`"study"`, `"history"`).

**Comparison Matrix:**

| Evaluation Dimension | Stemming (e.g., Porter, Snowball) | Lemmatization (WordNet) |
| :--- | :--- | :--- |
| **Linguistic Mechanism** | Suffix-stripping heuristic state machine | Morphological analysis + Lexicon lookup |
| **Output Validity** | Often non-words (`"univers"`, `"happili"`) | Always valid dictionary words (`"universe"`, `"happily"`) |
| **Computational Cost** | Microsecond speed, tiny memory footprint | $5\times - 10\times$ slower; requires lexical database in memory |
| **Context Dependency** | Zero awareness of context or POS tag | Heavily dependent on the word's Part-of-Speech tag |
| **Irregular Forms** | Fails entirely (`"went"` $\to$ `"went"`) | Resolves irregular inflections (`"went"` $\to$ `"go"`) |

---

### Q3: Why does NLTK's `WordNetLemmatizer` fail on `"running"` unless a POS tag is explicitly passed?
**Quick Take:**  
Because `WordNetLemmatizer.lemmatize(word, pos)` defaults to `pos='n'` (Noun). In English, `"running"` can be a noun (*"Running is good for health"*) or a verb (*"He is running fast"*). Without an explicit verb tag (`pos='v'`), the lemmatizer looks for the noun form of `"running"`, finds it valid in WordNet, and leaves it unchanged.

**Code Reality:**
```python
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("running"))          # Returns 'running' (Assumes Noun!)
print(lemmatizer.lemmatize("running", pos='v'))  # Returns 'run'     (Correct Verb Lemma!)
print(lemmatizer.lemmatize("better", pos='a'))   # Returns 'good'    (Correct Adjective Lemma!)
```
In automated pipelines, you must map Penn Treebank POS tags (`NN`, `VB`, `JJ`, `RB`) to WordNet POS constants (`wordnet.NOUN`, `wordnet.VERB`, `wordnet.ADJ`, `wordnet.ADV`) before calling the lemmatizer.

---

### Q4: When is removing stopwords a critical mistake in an NLP pipeline?
**Quick Take:**  
Stopword removal is only safe when the task depends purely on topical keywords (document classification, spam detection, topic modeling). It is disastrous in:
1. **Sentiment Analysis:** `"not"`, `"neither"`, `"nor"` are standard stopwords. Filtering them transforms `"The food was not good"` into `"food good"`, completely inverting the polarity.
2. **Machine Translation:** Grammatical particles, pronouns, and prepositions dictate case, gender, and syntactic agreement in the target language.
3. **Question Answering (QA):** Question intent is governed entirely by interrogative stopwords (`"Who"`, `"Where"`, `"When"`, `"Why"`).
4. **Modern Transformer Models (BERT, GPT):** Self-attention mechanisms compute bidirectional contextual representations across all tokens; removing words corrupts positional embeddings and attention weights.

---

### Q5: What is the optimal execution order for a robust text pre-processing pipeline, and why does order matter?
**Quick Take:**  
Execution order is dictated by dependency chains:
1. **Regex Cleaning of Non-Text Artifacts:** Strip raw HTML, XML tags, URLs, and noisy scripts before token boundaries are established.
2. **Sentence Tokenization (`sent_tokenize`):** Split raw text into documents/sentences while period characters still demarcate sentence boundaries.
3. **Named Entity Recognition (NER) & POS Tagging:** Run **before** lowercasing. Uppercase letters are critical features for identifying proper nouns (e.g., `"Apple"` the company vs `"apple"` the fruit; `"US"` the nation vs `"us"` the pronoun).
4. **Word Tokenization:** Segment sentences into individual tokens.
5. **Lowercasing:** Compress vocabulary by standardizing sentence-initial capitalization.
6. **Task-Specific Stopword Filtering:** Selectively remove non-informative tokens while preserving negations.
7. **Lemmatization (with POS tags):** Reduce tokens to lemmas.

---

## 2. Pillar 2: Classical Lexical Vectorization (OHE, BoW, N-Grams, TF-IDF)

### Q6: What are the three mathematical failure modes of One-Hot Encoding for text?
**Quick Take:**  
1. **Curse of Dimensionality & Memory Inefficiency:** Vector dimensionality equals $|V|$. In production ($|V| \ge 100,000$), vectors are $99.999\%$ sparse zeros, wasting gigabytes of RAM.
2. **Orthogonality / Zero Semantic Information:** In linear algebra, distinct one-hot vectors are orthogonal:
   $$\mathbf{u} \cdot \mathbf{v} = 0 \implies \text{Cosine Similarity} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = 0$$
   The model cannot detect that `"car"` and `"automobile"` are synonyms.
3. **Variable Matrix Shape:** A 10-word document produces a $(10 \times |V|)$ matrix, whereas a 3-word document produces a $(3 \times |V|)$ matrix. Standard classifiers require fixed-dimension vector inputs $\mathbb{R}^d$.

---

### Q7: How does Bag of Words (BoW) solve the variable matrix shape of OHE, and what does it sacrifice in return?
**Quick Take:**  
BoW aggregates all word occurrences within a document into a single frequency vector of fixed length $|V|$. Regardless of whether a document contains 5 words or 5,000 words, its representation is strictly a single vector $\mathbf{v} \in \mathbb{R}^{|V|}$.  
**What it sacrifices:** It discards word order, syntax, and grammatical dependencies entirely. The sentences `"The dog bit the man"` and `"The man bit the dog"` produce identical Bag of Words representations despite expressing opposite real-world events.

---

### Q8: Derive the mathematical proof showing why Bigrams resolve negation blindness where Unigrams fail.
**Proof:**  
Consider two documents:
- $D_1$: `"food good"` (Cleaned positive sentiment)
- $D_2$: `"food not good"` (Cleaned negative sentiment)

**Under Unigrams ($N=1$):**
- Vocabulary: $V_1 = [\text{"food"}, \text{"good"}, \text{"not"}]$
- $\mathbf{v}(D_1) = [1, 1, 0]^\top, \quad \|\mathbf{v}(D_1)\| = \sqrt{1^2 + 1^2 + 0^2} = \sqrt{2}$
- $\mathbf{v}(D_2) = [1, 1, 1]^\top, \quad \|\mathbf{v}(D_2)\| = \sqrt{1^2 + 1^2 + 1^2} = \sqrt{3}$
- Cosine Similarity:
  $$\text{Sim}_{\text{unigram}} = \frac{1(1) + 1(1) + 0(1)}{\sqrt{2} \cdot \sqrt{3}} = \frac{2}{\sqrt{6}} \approx \mathbf{0.816} \quad (81.6\% \text{ similar!})$$

**Under Bigrams ($N=2$):**
- Bigram Vocabulary: $V_2 = [\text{"food good"}, \text{"food not"}, \text{"not good"}]$
- $\mathbf{v}(D_1) = [1, 0, 0]^\top, \quad \|\mathbf{v}(D_1)\| = 1$
- $\mathbf{v}(D_2) = [0, 1, 1]^\top, \quad \|\mathbf{v}(D_2)\| = \sqrt{2}$
- Cosine Similarity:
  $$\text{Sim}_{\text{bigram}} = \frac{1(0) + 0(1) + 0(1)}{1 \cdot \sqrt{2}} = \frac{0}{\sqrt{2}} = \mathbf{0.000} \quad (\text{Completely separated!})$$

By capturing adjacent token pairs, bigrams turn `"not good"` into a dedicated feature that eliminates the misleading overlap caused by the isolated word `"good"`.

---

### Q9: Walk through the exact formula for TF-IDF and explain why the logarithm is applied to Inverse Document Frequency.
**Mathematical Formulation:**
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$
Where:
$$\text{TF}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}, \quad \text{IDF}(t, D) = \ln\left(\frac{N}{\text{DF}(t)}\right)$$
- $f_{t,d}$: Raw frequency of term $t$ in document $d$.
- $N$: Total number of documents in corpus $D$.
- $\text{DF}(t)$: Number of documents containing term $t$.

**Why the Logarithm?**  
In a corpus of $N = 10,000,000$ documents:
- A rare word appearing in 1 document has raw ratio $\frac{N}{\text{DF}} = 10,000,000$.
- A common word appearing in 10,000 documents has raw ratio $\frac{N}{\text{DF}} = 1,000$.
- Without a logarithm, the rare word's multiplier is $10,000\times$ larger, completely obliterating the Term Frequency signal.
- The natural logarithm dampens this exponential scale into manageable linear steps: $\ln(10^7) \approx 16.1$ vs $\ln(10^3) \approx 6.9$. It maintains proper ordering while preventing rare terms from causing numerical instability.

---

### Q10: How does Scikit-Learn's `TfidfVectorizer` implementation differ from textbook TF-IDF?
**Technical Nuances in Scikit-Learn:**
1. **Smooth IDF (`smooth_idf=True`, default):** Adds `1` to the numerator and denominator to prevent division-by-zero:
   $$\text{IDF}_{\text{sklearn}}(t) = \ln\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1$$
   *(The trailing $+1$ ensures that words with $\text{DF}=N$ receive a non-zero weight instead of being completely eliminated).*
2. **L2 Normalization (`norm='l2'`, default):** Each document vector is normalized to unit Euclidean length:
   $$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\sqrt{\sum v_i^2}}$$
   This guarantees that long documents with many words do not artificially dominate shorter documents.

---

### Q11: What is BM25, and why do modern search engines (like Elasticsearch) prefer it over standard TF-IDF?
**Quick Take:**  
BM25 (Best Matching 25) is an advanced probabilistic evolution of TF-IDF that solves two major limitations of classical TF-IDF:
1. **Term Frequency Saturation:** In standard TF-IDF, if a word appears 20 times in a document, its score is roughly $20\times$ higher than if it appeared once. In BM25, term frequency has diminishing returns: after 3 or 4 occurrences, additional repetitions asymptotically approach an upper bound.
2. **Document Length Normalization:** BM25 explicitly accounts for average document length in the corpus, aggressively penalizing long, rambling documents while boosting concise, highly concentrated matches.

---

## 3. Pillar 3: Distributed Word Embeddings & Geometric Semantics

### Q12: What is the core difference between count-based embeddings (TF-IDF) and prediction-based embeddings (Word2Vec)?
**Quick Take:**  
Count-based methods (TF-IDF, Co-occurrence matrices) compute global statistical frequencies across the corpus, resulting in high-dimensional, sparse vectors where semantic similarity is not directly encoded. Prediction-based methods (Word2Vec) train a neural network to predict context words within local sliding windows; the learned internal weights become dense, continuous vectors that capture fine-grained semantic, syntactic, and analogical relationships.

---

### Q13: Explain the geometry of Word2Vec analogical reasoning: $\mathbf{v}_{\text{King}} - \mathbf{v}_{\text{Man}} + \mathbf{v}_{\text{Woman}} \approx \mathbf{v}_{\text{Queen}}$.
**Vector Space Derivation:**
1. Word2Vec projects words into a continuous latent space where semantic properties correspond to spatial displacement vectors.
2. The vector displacement $\vec{d}_{\text{gender}} = \mathbf{v}_{\text{Woman}} - \mathbf{v}_{\text{Man}}$ encodes the exact directional shift representing the feminine-to-masculine transition.
3. The displacement $\mathbf{v}_{\text{King}} - \mathbf{v}_{\text{Man}}$ removes the male component from royalty, leaving an abstract concept vector: $\vec{v}_{\text{royalty}}$.
4. Adding $\mathbf{v}_{\text{Woman}}$ applies the female direction to the royalty vector:
   $$\mathbf{v}^* = \mathbf{v}_{\text{King}} + (\mathbf{v}_{\text{Woman}} - \mathbf{v}_{\text{Man}})$$
5. Computing the Cosine Similarity between $\mathbf{v}^*$ and every word in the vocabulary identifies $\mathbf{v}_{\text{Queen}}$ as the closest vector by angular proximity.

```
       ▲ Dimension 2 (Royalty)
       │
       │    [King] ● ───────────────► ● [Queen]
       │       ▲                         ▲
       │       │                         │
       │       │ + (Woman - Man)         │ + (Woman - Man)
       │       │                         │
       │    [Man]  ● ───────────────► ● [Woman]
       │
       └────────────────────────────────────────► Dimension 1 (Gender)
```

---

### Q14: Why is Cosine Similarity preferred over Euclidean Distance for comparing word embeddings?
**Quick Take:**  
Euclidean distance is sensitive to vector magnitude (length). In Word2Vec, vector magnitude is often influenced by token frequency (words occurring frequently in the training corpus accumulate larger weight updates). Cosine similarity normalizes vector lengths, measuring purely the **directional angle $\theta$** in latent space:
$$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \cos(\theta)$$
Two words expressing the same concept in different frequency contexts will have vectors pointing in the same direction even if their magnitudes differ.

---

## 4. Pillar 4: Word2Vec Architectural Mechanics (CBOW vs Skip-Gram)

### Q15: Walk through the exact forward pass of the Continuous Bag of Words (CBOW) architecture.
**Detailed Mathematical Forward Pass:**
1. **Input:** Let context window size be $C=2$ (total context words $2C=4$). Context words are represented as one-hot vectors $\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3, \mathbf{x}_4 \in \{0, 1\}^{|V|}$.
2. **Input Weight Matrix ($W \in \mathbb{R}^{|V| \times N}$):** Multiplying $W^\top \mathbf{x}_i$ retrieves row $i$, which is the $N$-dimensional input embedding of word $i$.
3. **Hidden / Projection Layer ($\mathbf{h} \in \mathbb{R}^N$):** Average all context word vectors without non-linear activation:
   $$\mathbf{h} = \frac{1}{2C} \sum_{i=1}^{2C} W^\top \mathbf{x}_i$$
4. **Output Weight Matrix ($W' \in \mathbb{R}^{N \times |V|}$):** Project the hidden vector into the full vocabulary space to produce logit scores $\mathbf{z}$:
   $$\mathbf{z} = {W'}^\top \mathbf{h} \in \mathbb{R}^{|V|}$$
5. **Softmax Output ($\hat{\mathbf{y}} \in \mathbb{R}^{|V|}$):** Convert raw logits into a valid probability distribution:
   $$\hat{y}_j = P(w_j \mid \text{context}) = \frac{\exp(z_j)}{\sum_{k=1}^{|V|} \exp(z_k)}$$
6. **Loss:** Categorical Cross-Entropy against the one-hot target center word $\mathbf{y}$.

---

### Q16: Walk through the forward pass of the Skip-Gram architecture.
**Mathematical Mechanics:**
1. **Input:** A single center target word represented as one-hot vector $\mathbf{x} \in \{0, 1\}^{|V|}$.
2. **Hidden Layer ($\mathbf{h} \in \mathbb{R}^N$):** Extract the embedding of the center word:
   $$\mathbf{h} = W^\top \mathbf{x}$$
3. **Output Layer:** For each of the $2C$ context positions, compute logit scores using the shared output matrix $W' \in \mathbb{R}^{N \times |V|}$:
   $$\mathbf{z} = {W'}^\top \mathbf{h}$$
4. **Softmax Probabilities:** Predict the probability of observing context word $w_{c}$ at position $c$:
   $$P(w_{c} \mid w_{\text{target}}) = \frac{\exp({\mathbf{v}'_{w_c}}^\top \mathbf{v}_{w_{\text{target}}})}{\sum_{k=1}^{|V|} \exp({\mathbf{v}'_{w_k}}^\top \mathbf{v}_{w_{\text{target}}})}$$
5. The loss is the sum of cross-entropy losses across all $2C$ context word predictions.

---

### Q17: Why is standard Softmax computationally intractable in Word2Vec, and how does Negative Sampling solve it?
**The Bottleneck:**  
The denominator of the Softmax function is $\sum_{k=1}^{|V|} \exp({\mathbf{v}'_{w_k}}^\top \mathbf{h})$. In a vocabulary of $|V| = 1,000,000$ words, computing this sum and its gradients across billions of training windows requires trillions of floating-point operations ($O(|V|)$ cost per training sample).

**The Negative Sampling Solution (SGNS):**  
Replaces multi-class classification over $|V|$ classes with $K+1$ independent binary logistic regressions:
1. Maximize the probability that the true context word $w_O$ came from the data:
   $$\ln \sigma({\mathbf{v}'_{w_O}}^\top \mathbf{v}_{w_I})$$
2. Minimize the probability that $K$ randomly selected noise words $w_i$ came from the data:
   $$\sum_{i=1}^K \ln \sigma(-{\mathbf{v}'_{w_i}}^\top \mathbf{v}_{w_I})$$
**Total Objective Function:**
$$\mathcal{L}_{\text{NEG}} = \ln \sigma({\mathbf{v}'_{w_O}}^\top \mathbf{v}_{w_I}) + \sum_{i=1}^K \mathbb{E}_{w_i \sim P_n(w)} \left[\ln \sigma(-{\mathbf{v}'_{w_i}}^\top \mathbf{v}_{w_I})\right]$$
Computational complexity drops from **$O(|V|) \to O(K)$**, where $K \approx 5 - 20$, achieving a $10,000\times$ speedup.

---

### Q18: What is the Unigram 3/4 Distribution, and why is it used in Negative Sampling?
**Mathematical Definition:**
$$P_n(w) = \frac{f(w)^{0.75}}{\sum_{w'} f(w')^{0.75}}$$
Where $f(w)$ is the unigram frequency of word $w$.

**Why the 0.75 Exponent?**  
- If you sample strictly proportional to frequency ($f(w)^1$), frequent stopwords (`"the"`, `"is"`) get sampled as negative examples almost 100% of the time, while rare words are never sampled.
- If you sample uniformly ($f(w)^0$), all words have equal probability, completely ignoring actual language statistics.
- The power of $0.75$ hits the empirical sweet spot:
  - Suppose `"the"` has frequency $f = 1,000,000 \implies (10^6)^{0.75} = 31,622$ (Dampened by $31\times$!).
  - Suppose `"quarks"` has frequency $f = 16 \implies (16)^{0.75} = 8$ (Dampened by only $2\times$!).
  - The relative probability of sampling rare words as negative examples increases substantially.

---

### Q19: Under what conditions should you select CBOW vs Skip-Gram in production?
**Decision Framework:**
- **Select CBOW when:**
  1. Training speed is a primary constraint (CBOW is $3\times - 5\times$ faster).
  2. The corpus is smaller to medium-sized.
  3. The task prioritizes general syntactic and functional representations of common words.
- **Select Skip-Gram when:**
  1. The corpus is massive ($>10^8$ words).
  2. The domain contains technical, specialized, or rare terminology (medical, financial, legal). Skip-Gram does not average context words together, preserving distinct representations for infrequent tokens.

---

## 5. Pillar 5: Document Embeddings, FastText, GloVe & The Polysemy Problem

### Q20: How does Average Word2Vec work, and what is its primary failure mode?
**Mechanism:**  
Given a document $D$ with $M$ tokens, the document vector $\mathbf{v}_{\text{doc}}$ is the arithmetic mean of its constituent word embeddings:
$$\mathbf{v}_{\text{doc}} = \frac{1}{M} \sum_{i=1}^M \mathbf{v}_{w_i} \in \mathbb{R}^d$$
**Primary Failure Modes:**
1. **Semantic Centroid Dilution:** As document length increases ($>300$ words), summing hundreds of vectors in different directions forces the mean vector toward the corpus origin, washing out distinct topical signals.
2. **Word Order Blindness:** `"Cat chases mouse"` and `"Mouse chases cat"` produce identical centroids.
3. **Negation Blindness:** In `"The service was good"` vs `"The service was not good"`, averaging 5 vectors dilutes the single negation word `"not"`, leaving both document centroids highly similar.

---

### Q21: What is TF-IDF Weighted Average Word2Vec, and why is it superior to naive Average Word2Vec?
**Mathematical Formulation:**
$$\mathbf{v}_{\text{doc}}^{\text{TF-IDF}} = \frac{\sum_{i=1}^M \text{TF-IDF}(w_i, D) \cdot \mathbf{v}_{w_i}}{\sum_{i=1}^M \text{TF-IDF}(w_i, D)}$$
**Advantage:** Naive averaging treats common words (`"good"`, `"make"`, `"day"`) with the same weight as critical domain terms (`"oncology"`, `"bankruptcy"`). By weighting each vector by its corpus uniqueness ($\text{TF-IDF}$), rare informative terms dictate the document centroid's position in vector space.

---

### Q22: How does FastText solve the Out-of-Vocabulary (OOV) problem that Word2Vec cannot handle?
**Quick Take:**  
Word2Vec assigns atomic vectors to whole word strings; an unseen word or typo has no vector. FastText (Facebook, 2016) treats each word as a bag of **character n-grams** wrapped in boundary markers:
- For word `"where"` with $n=3$: `<wh`, `whe`, `her`, `ere`, `re>` and special sequence `<where>`.
- The vector for `"where"` is the sum of its character n-gram vectors.
- If an unseen word like `"whereever"` appears at test time, FastText extracts its known character n-grams (`<wh`, `whe`, `ere`, `eve`, `ver>`), sums their embeddings, and constructs a highly accurate semantic vector for the unseen word.

---

### Q23: What is the fundamental difference between Word2Vec and GloVe?
**Comparison:**
- **Word2Vec (Predictive):** Iterates over local context windows with stochastic gradient descent. Captures local context patterns but discards global corpus statistics.
- **GloVe (Global Vectors, Stanford):** A **count-based matrix factorization** model. It first compiles a global word-word co-occurrence matrix $X$ across the entire corpus, then minimizes a log-bilinear squared loss:
  $$J = \sum_{i,j=1}^{|V|} f(X_{i,j}) \left(\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \ln X_{i,j}\right)^2$$
  GloVe explicitly fits the ratios of co-occurrence probabilities, directly capturing global corpus statistics.

---

### Q24: What is the Polysemy Problem, and why are Word2Vec, GloVe, and FastText incapable of solving it?
**Quick Take:**  
**Polysemy** refers to words having multiple distinct meanings based on context (e.g., `"bank"` as a financial institution vs `"bank"` of a river; `"apple"` as a fruit vs `"apple"` as a tech brand).  
**Why static embeddings fail:** Word2Vec, GloVe, and FastText generate **one static vector per vocabulary string**. The single vector for `"bank"` is forced to settle into an unnatural compromise coordinate equidistant between finance and geology.  
**Resolution:** Modern **Contextual Embeddings** (BERT, RoBERTa, ELMo) use multi-head self-attention to generate dynamic vectors where the embedding of `"bank"` changes based on every other token in the sentence.

---

## 6. Pillar 6: Production Pipeline Architecture, Diagnostics & Evaluation

### Q25: Why is Multinomial Naive Bayes exceptionally well-suited for Bag of Words and TF-IDF text classification?
**Technical Rationale:**
1. **Discrete Count Modeling:** Multinomial Naive Bayes assumes features are generated by a multinomial distribution over word counts:
   $$P(d \mid c) \propto \prod_{i=1}^{|V|} P(w_i \mid c)^{f_{i,d}}$$
   This matches the discrete frequency histograms produced by CountVectorizer.
2. **High-Dimensional Sparsity Handling:** Naive Bayes evaluates conditional probabilities independently per feature. Sparse zero entries simply drop out or are smoothed with Laplace smoothing ($\alpha = 1.0$), making it immune to the Curse of Dimensionality.
3. **Blazing Fast Training & Inference:** Parameter estimation requires single-pass count aggregation $O(N \cdot L)$, executing in milliseconds on millions of records where neural networks require hours.

---

### Q26: Why is Accuracy a dangerous evaluation metric for text classification pipelines (e.g., Spam or Hate Speech detection)?
**Production Reality:**  
Text classification datasets are almost always heavily imbalanced (e.g., 99% legitimate emails, 1% spam; 98% neutral comments, 2% toxic comments).  
- A trivial model predicting "Not Spam" for every single email achieves **$99.0\%$ Accuracy**, yet is completely non-functional.
- **Production Metrics to Use:**
  - **Precision:** $\frac{TP}{TP + FP}$ — Critical when False Positives carry high cost (e.g., deleting an important client email).
  - **Recall:** $\frac{TP}{TP + FN}$ — Critical when False Negatives carry high cost (e.g., missing fraud or malware).
  - **F1-Score / PR-AUC:** Harmonic mean of precision and recall; evaluates the precision-recall trade-off across imbalanced classes regardless of majority class volume.

---

### Q27: How can Data Leakage occur during text vectorization, and how do you prevent it?
**The Danger:**  
Fitting a `CountVectorizer` or `TfidfVectorizer` on the entire dataset *before* performing train/test split.  
- **What leaks:**
  1. The vocabulary of the test set leaks into the training vocabulary.
  2. The Inverse Document Frequency ($\text{IDF}$) weights incorporate document frequencies from the test set, giving the model prior knowledge of the test distribution.
- **The Fix:**  
  Always split your dataset **first**. Call `.fit_transform()` strictly on the training set, and `.transform()` on the test/validation set:
  ```python
  from sklearn.model_selection import train_test_split
  from sklearn.feature_extraction.text import TfidfVectorizer

  X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2)

  tfidf = TfidfVectorizer(max_features=5000)
  X_train = tfidf.fit_transform(X_train_raw)  # Learns vocabulary & IDF from TRAIN only
  X_test = tfidf.transform(X_test_raw)        # Transforms TEST using TRAIN parameters
  ```
