# Module 4 — Recurrent Neural Networks

> **Simple RNNs · PyTorch · IMDB Sentiment Analysis**

---

## 📁 Module Contents

```
4. Recurrent Neural Networks/
├── SIMPLE_RNN_MASTER_NOTES.md                    # 📒 Complete concept reference
├── 01_simple_rnn_imdb_sentiment_analysis.ipynb   # 🧪 End-to-end master notebook
├── app.py                                         # 🚀 Streamlit inference app
├── simple_rnn_imdb.pth                           # 💾 Trained model weights
├── imdb_word_index.json                          # 📖 IMDB vocabulary (10k words)
└── README.md                                     # ← this file
```

---

## 🏗️ Structure Decision: 3 Files → 1 Unified Notebook

The original TensorFlow structure had three separate files:

| Old File | Purpose |
|---|---|
| `embedding.ipynb` | Embedding layer demo |
| `simplernn.ipynb` | RNN architecture definition |
| `prediction.ipynb` | Inference only |

### Why That Structure Fails for Learning

1. **Broken dependency chain** — `prediction.ipynb` requires a `.h5` file that `simplernn.ipynb` must have produced in a prior run. If you open prediction first, it crashes. There is no self-contained starting point.
2. **Duplicated preprocessing** — both the training notebook and the prediction notebook implement tokenisation and padding independently. When one changes, the other silently diverges, causing mismatched predictions.
3. **Conceptual fragmentation** — separating embedding from RNN from prediction hides the most important insight: these are *one continuous computational graph*. A learner who never sees them wired together misses how data flows end-to-end.
4. **No gradient narrative** — splitting the files obscures where backpropagation actually happens (it spans embedding, RNN weights *and* the linear head simultaneously).

### New Structure: Unified Notebook + Standalone App

```
01_simple_rnn_imdb_sentiment_analysis.ipynb   ← learn everything here
app.py                                         ← deploy / demo here
```

| Criterion | 3 Separate Files | 1 Notebook + app.py |
|---|---|---|
| Beginner friendliness | ❌ Confusing load order | ✅ Top-to-bottom flow |
| Dependency safety | ❌ Silent crashes | ✅ Self-contained |
| Preprocessing consistency | ❌ Duplicated & diverges | ✅ Single source of truth |
| Conceptual clarity | ❌ Hides the full graph | ✅ Full pipeline visible |
| Reusability | ❌ Hard to copy-adapt | ✅ `app.py` imports nothing from notebook |
| Debugging | ❌ Must trace across files | ✅ All state in one kernel |

---

## 🧠 Architecture

```
Input text  →  Tokenise & Pad  →  [B, 500]
                                       │
                               nn.Embedding(10000, 128)
                                       │
                                  [B, 500, 128]
                                       │
                          nn.RNN(128 → 128, tanh, batch_first)
                                       │
                            last hidden state  [B, 128]
                                       │
                               nn.Linear(128, 1)
                                       │
                                  logit  [B, 1]
                                       │
                                  sigmoid → ŷ ∈ (0,1)
```

### Tensor Shape Cheatsheet

| Layer | Input Shape | Output Shape | Note |
|---|---|---|---|
| `nn.Embedding` | `[B, T]` | `[B, T, 128]` | Lookup table: int → dense vector |
| `nn.RNN` | `[B, T, 128]` | `output [B,T,128]`, `h_n [1,B,128]` | We only use `h_n` |
| `.squeeze(0)` | `[1, B, 128]` | `[B, 128]` | Remove layer dimension |
| `nn.Linear` | `[B, 128]` | `[B, 1]` | Final classification logit |

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `vocab_size` | 10,000 | Top-10k IMDB words cover ~95% of tokens |
| `embed_dim` | 128 | Balance between expressiveness and memory |
| `hidden_dim` | 128 | Match embedding dim for simplicity |
| `max_len` | 500 | Covers ~90th percentile review length |
| `batch_size` | 64 | Stable gradient estimates on a single GPU |
| `lr` | 1e-3 | Adam default; works well here |
| `grad_clip` | 1.0 | Prevents exploding gradients over 500 steps |
| `nonlinearity` | `tanh` | Bounded activations → stable BPTT |

---

## ⚡ Quickstart

### 1 · Install dependencies

```bash
pip install torch torchvision streamlit jupyter scikit-learn matplotlib seaborn
```

### 2 · Open the study notebook

```bash
jupyter notebook "01_simple_rnn_imdb_sentiment_analysis.ipynb"
```

Run all cells from top to bottom. The notebook will:
- Download the IMDB dataset (~17 MB)
- Train the model for 5 epochs (~5 min on GPU, ~20 min on CPU)
- Save `simple_rnn_imdb.pth` and `imdb_word_index.json`
- Print a classification report and plot loss / accuracy curves

### 3 · Launch the demo app

```bash
# from this directory:
streamlit run app.py
```

The app loads the saved weights and lets you type any review to see the model's prediction with a confidence score.

---

## 📊 Results

| Metric | Value |
|---|---|
| Training accuracy (epoch 5) | ~84% |
| Test accuracy | **~76.5%** |
| Loss function | `BCEWithLogitsLoss` |
| Optimizer | `Adam` |

> **Why ~76.5%?** Simple RNNs suffer from **vanishing gradients** over long sequences — by the time the gradient flows back 400 time steps, it has shrunk toward zero. The weights at early time steps receive almost no learning signal. LSTMs and GRUs solve this with gating mechanisms; Transformers use attention and bypass the sequential bottleneck entirely.

---

## 📒 Master Notes Highlights

The [`SIMPLE_RNN_MASTER_NOTES.md`](./SIMPLE_RNN_MASTER_NOTES.md) covers:

- **First-principles derivation** of the forward pass equations
- **Unrolled RNN diagrams** with shared weight visualisation
- **BPTT deep-dive** — why gradients are *accumulated* (not averaged) across time steps for shared weights
- **Vanishing / exploding gradient analysis** — Jacobian eigenvalue argument
- **TensorFlow vs PyTorch API comparison table**
- **Common beginner confusions** addressed explicitly

---

## 🔗 Key Concepts

| Concept | Where Explained |
|---|---|
| What is an RNN and why sequences need memory | Notes § 1 |
| Hidden state as a learned summary | Notes § 2 |
| Weight sharing across time steps | Notes § 3 |
| Forward propagation math | Notes § 4 + Notebook Cell 5 |
| BPTT and gradient accumulation | Notes § 5 |
| Vanishing gradients | Notes § 6 |
| `nn.Embedding` vs one-hot | Notes § 7 + Notebook Cells 3-4 |
| Why we take the *last* hidden state | Notebook Cell 6 (model definition) |
| `BCEWithLogitsLoss` vs `BCE + sigmoid` | Notebook Cell 8 (training loop) |
| Gradient clipping | Notebook Cell 8 + Notes § 6.3 |
| Inference with `torch.no_grad()` | Notebook Cell 11 + `app.py` |
