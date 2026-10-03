"""
Streamlit App: IMDB Sentiment Analysis with Simple RNN (PyTorch)
================================================================
Run: streamlit run app.py

Requirements:
    pip install streamlit torch

Place these files in the same directory:
    - simple_rnn_imdb.pth   (trained model weights)
    - imdb_word_index.json  (vocabulary mapping)
"""

import json
import os
import torch
import torch.nn as nn
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# 1. Model Definition  (must match training exactly)
# ──────────────────────────────────────────────────────────────────────────────

class SentimentRNN(nn.Module):
    """
    Simple RNN for binary sentiment classification.

    Architecture:
        Embedding → RNN → (take last hidden state) → Linear → scalar logit

    Note: We output a single logit (no sigmoid here).
          BCEWithLogitsLoss (training) / torch.sigmoid (inference) handles that.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        # Word index → dense vector  [B, T] → [B, T, embed_dim]
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,  # index 0 = <PAD> → always zero vector
        )

        # Recurrent layer: tanh is stable for long sequences
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,           # input shape: [B, T, embed_dim]
            nonlinearity="tanh",
        )

        # Single linear head mapping hidden → logit
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T]  (token ids, padded to fixed length)
        Returns: [B, 1]  (raw logit per sample)
        """
        embedded = self.embedding(x)           # [B, T, embed_dim]
        _, hidden = self.rnn(embedded)         # hidden: [1, B, hidden_dim]
        hidden = hidden.squeeze(0)             # [B, hidden_dim]
        return self.fc(hidden)                 # [B, output_dim=1]


# ──────────────────────────────────────────────────────────────────────────────
# 2. Config  (must match training hyperparameters)
# ──────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "vocab_size":  10_000,
    "embed_dim":   128,
    "hidden_dim":  128,
    "output_dim":  1,
    "max_len":     500,    # sequences padded / truncated to this length
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "simple_rnn_imdb.pth")
VOCAB_PATH = os.path.join(BASE_DIR, "imdb_word_index.json")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Cached Loading  (runs once per Streamlit session)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model() -> SentimentRNN:
    """Load weights into the model and set to eval mode."""
    model = SentimentRNN(**CONFIG)
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_vocab() -> dict[str, int]:
    """Load IMDB word-index mapping (word → integer id)."""
    with open(VOCAB_PATH, "r") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Preprocessing  (mirrors training pipeline exactly)
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(text: str, word_index: dict[str, int], max_len: int) -> torch.Tensor:
    """
    text → tokenised & padded integer sequence → [1, max_len] tensor

    IMDB word_index uses offset +3 internally:
        0 = <PAD>
        1 = <START>
        2 = <UNK>
        3 = <UNUSED>
        4+ = actual words

    Words not in vocab or with id >= vocab_size are mapped to <UNK> (2).
    """
    tokens = text.lower().split()

    # Map each token to its integer id (with the +3 offset)
    ids = [
        min(word_index.get(tok, 2) + 3, CONFIG["vocab_size"] - 1)
        for tok in tokens
    ]

    # Truncate long sequences (keep the last max_len tokens, matching training)
    ids = ids[-max_len:]

    # Left-pad with zeros so the sequence is exactly max_len long
    padded = [0] * (max_len - len(ids)) + ids

    return torch.tensor([padded], dtype=torch.long, device=DEVICE)  # [1, max_len]


# ──────────────────────────────────────────────────────────────────────────────
# 5. Inference
# ──────────────────────────────────────────────────────────────────────────────

def predict(text: str, model: SentimentRNN, word_index: dict) -> tuple[str, float]:
    """
    Returns (label, confidence_percent).

    Steps:
        1. Preprocess raw text → padded tensor
        2. Forward pass with no_grad (no BPTT, no graph construction)
        3. Apply sigmoid to convert logit → probability in [0, 1]
        4. Threshold at 0.5 → Positive / Negative
    """
    x   = preprocess(text, word_index, CONFIG["max_len"])

    with torch.no_grad():
        logit = model(x)                  # [1, 1]  raw score
        prob  = torch.sigmoid(logit).item()   # scalar in [0, 1]

    label = "Positive 😊" if prob >= 0.5 else "Negative 😞"
    confidence = prob if prob >= 0.5 else 1 - prob  # distance from boundary
    return label, confidence * 100


# ──────────────────────────────────────────────────────────────────────────────
# 6. Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="IMDB Sentiment — Simple RNN",
        page_icon="🎬",
        layout="centered",
    )

    # ── Header ──
    st.title("🎬 IMDB Sentiment Analysis")
    st.caption("Simple RNN (PyTorch) · Trained on IMDB 25k Reviews")
    st.divider()

    # ── Model Loading ──
    with st.spinner("Loading model..."):
        try:
            model      = load_model()
            word_index = load_vocab()
        except FileNotFoundError as e:
            st.error(f"❌ Missing file: {e}\n\nMake sure `simple_rnn_imdb.pth` and `imdb_word_index.json` are in the same directory as `app.py`.")
            st.stop()

    device_label = "GPU 🚀" if DEVICE.type == "cuda" else "CPU 💻"
    st.success(f"Model loaded · Running on {device_label}")

    # ── Input ──
    st.subheader("Enter a Movie Review")
    review = st.text_area(
        label="Review text",
        placeholder="e.g. This film was absolutely brilliant! The acting was superb and the plot kept me on the edge of my seat.",
        height=160,
        label_visibility="collapsed",
    )

    # ── Example Reviews ──
    with st.expander("📋 Try example reviews"):
        examples = {
            "Positive Example": "This movie was absolutely brilliant. The performances were outstanding and the direction was masterful. One of the best films I have seen in years.",
            "Negative Example": "What a complete waste of time. The plot made no sense, the acting was terrible, and the ending was beyond disappointing. I want my two hours back.",
            "Mixed Example":    "The visuals were stunning and the soundtrack was great, but the story felt hollow and the characters were poorly written.",
        }
        for label, text in examples.items():
            if st.button(label, use_container_width=True):
                st.session_state["example"] = text
                st.rerun()

    # Populate text area from selected example
    if "example" in st.session_state:
        review = st.session_state.pop("example")
        st.text_area("Review text", value=review, height=160, key="filled", label_visibility="collapsed")

    # ── Predict ──
    if st.button("Analyse Sentiment", type="primary", use_container_width=True):
        text = review.strip()

        if not text:
            st.warning("Please enter a review first.")
        elif len(text.split()) < 3:
            st.warning("Please enter at least a few words for a meaningful prediction.")
        else:
            with st.spinner("Analysing..."):
                label, confidence = predict(text, model, word_index)

            # ── Result Card ──
            st.divider()
            col1, col2 = st.columns([1, 2])

            with col1:
                st.metric("Sentiment", label)

            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")

            # Confidence bar (green for positive, red for negative)
            color = "green" if "Positive" in label else "red"
            st.markdown(
                f"""
                <div style="background:#e9ecef;border-radius:8px;height:22px;overflow:hidden;margin-top:8px">
                    <div style="background:{color};width:{confidence:.1f}%;height:100%;border-radius:8px;
                                transition:width 0.4s ease;display:flex;align-items:center;
                                padding-left:8px;color:white;font-size:13px;font-weight:600">
                        {confidence:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Technical Details (collapsible) ──
            with st.expander("🔬 Technical Details"):
                st.markdown(f"""
| Property | Value |
|---|---|
| Words in review | `{len(text.split())}` |
| Sequence length (padded) | `{CONFIG['max_len']}` |
| Vocabulary size | `{CONFIG['vocab_size']:,}` |
| Embedding dim | `{CONFIG['embed_dim']}` |
| RNN hidden dim | `{CONFIG['hidden_dim']}` |
| Device | `{DEVICE}` |
                """)

    # ── Sidebar Info ──
    st.sidebar.title("ℹ️ About This Model")
    st.sidebar.markdown("""
**Architecture**: Simple RNN

```
Input: [B, 500]
  ↓ Embedding [10k → 128]
  ↓ [B, 500, 128]
  ↓ RNN (hidden=128, tanh)
  ↓ last hidden [B, 128]
  ↓ Linear(128 → 1)
Output: logit → sigmoid
```

**Training**:
- Dataset: IMDB (25k train / 25k test)
- Optimizer: Adam (lr=1e-3)
- Loss: BCEWithLogitsLoss
- Gradient Clipping: max_norm=1.0
- Test Accuracy: ~76.5%

**Limitations**:
Simple RNNs struggle with very long sequences due to vanishing gradients.
LSTMs / GRUs / Transformers achieve 85–95% on this task.
    """)

    st.sidebar.divider()
    st.sidebar.caption("Built with PyTorch · Streamlit")


if __name__ == "__main__":
    main()
