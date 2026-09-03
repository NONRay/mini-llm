# mini-llm

A minimal from-scratch LLM implementation in PyTorch, featuring a LLaMA-style decoder architecture (RMSNorm + SwiGLU + causal self-attention) with tied input/output embeddings.

## Architecture

| Component | Detail |
|---|---|
| Attention | Multi-head causal self-attention (QKV fused, triangular mask) |
| Normalization | RMSNorm (pre-norm) |
| Feed-forward | SwiGLU (`w2(silu(w1(x)) * w3(x))`) |
| Embeddings | Token + learned positional; `lm_head` shares `token_embedding` weights (tied) |
| Tokenizer | SentencePiece BPE (vocab 32000) |

Default model config (see [model/config.py](model/config.py)):

```
vocab_size   32000
max_seq_len  512
n_layer      12
n_head       12
hidden_dim   768
dropout      0.1
```

## Project Structure

```
.
├── model/
│   ├── config.py          # ModelConfig & TrainConfig
│   ├── attention.py       # CausalSelfAttention
│   ├── llama_block.py     # RMSNorm, SwiGLU, TransformerBlock
│   └── transformer.py     # MiniLLM (full model)
├── dataset.py             # TextDataset (uint32 memmap, sliding-window)
├── training/
│   └── train.py           # Training loop (AdamW + grad clip)
├── inference.py           # Text generation (sampling)
├── scripts/
│   ├── download_fineweb_edu.py  # Download Chinese Fineweb Edu (high-quality curated corpus)
│   ├── download_chinese_c4.py   # Download Chinese C4 (alternative raw web corpus)
│   ├── train_tokenizer.py       # Train BPE tokenizer
│   ├── prepare_data.py          # Tokenize raw text -> train.bin (memmap)
│   └── evaluate.py              # Compute average loss
├── analysis/
│   ├── activation.py      # Extract per-layer hidden states
│   └── attention_map.py   # Visualize attention weights
├── data/
│   └── raw/train.txt      # Raw training text
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Download and clean the raw corpus

**Recommended: Chinese Fineweb Edu** (high-quality, education-filtered corpus):

```bash
python scripts/download_fineweb_edu.py
```

Downloads `opencsg/chinese-fineweb-edu` from Hugging Face. This dataset is filtered by
an educational-value scoring model (score > 4) and deduplicated via MinHash, making it
well-suited for small models where data quality matters more than scale.

Useful options:

```bash
python scripts/download_fineweb_edu.py --limit 500000
python scripts/download_fineweb_edu.py --output /tmp/fineweb_edu.txt
```

**Alternative: Chinese C4** (raw web crawl, lower quality):

```bash
python scripts/download_chinese_c4.py
```

### 2. Train the tokenizer

```bash
python scripts/train_tokenizer.py
```

Trains a SentencePiece BPE tokenizer on `data/raw/train.txt` and saves `data/tokenizer/tokenizer.model`.

### 3. Prepare the dataset

```bash
python scripts/prepare_data.py
```

Tokenizes the raw text and writes `data/processed/train.bin` as a `uint32` memmap array.

### 4. Train the model

```bash
python training/train.py
```

Runs the training loop (AdamW, gradient clipping) and saves a checkpoint per epoch (`checkpoint_epochN.pt`).

### 5. Generate text

```bash
python inference.py
```

Loads `checkpoint_epoch9.pt` and the tokenizer, then autoregressively samples 50 tokens from a prompt.

## License

MIT
