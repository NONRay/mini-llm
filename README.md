# mini-llm

A minimal from-scratch LLM implementation in PyTorch, featuring a LLaMA-style decoder architecture (RMSNorm + SwiGLU + causal self-attention) with tied input/output embeddings.

Training follows a two-stage pipeline: **pretraining** on a raw Chinese corpus, then **SFT** on instruction data (COIG-CQIA). Checkpoints of the two stages are stored in separate directories so an SFT run can never overwrite your pretrained base.

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
max_seq_len  1024
n_layer      8
n_head       8
hidden_dim   512
dropout      0.1
```

## Project Structure

```
.
├── model/
│   ├── config.py          # ModelConfig & TrainConfig (stage-specific defaults)
│   ├── attention.py       # CausalSelfAttention
│   ├── llama_block.py     # RMSNorm, SwiGLU, TransformerBlock
│   └── transformer.py     # MiniLLM (full model)
├── dataset.py             # TextDataset (pretrain) + SFTDataset (masked loss)
├── training/
│   └── train.py           # Stage-aware pretraining / SFT loop
├── inference.py           # Chat-style generation with checkpoint auto-discovery
├── scripts/
│   ├── download_fineweb_edu.py  # Download Chinese Fineweb Edu (pretrain corpus)
│   ├── download_chinese_c4.py   # Download Chinese C4 (alternative corpus)
│   ├── train_tokenizer.py       # Train BPE tokenizer
│   ├── prepare_data.py          # Tokenize raw text -> train.bin (memmap)
│   ├── prepare_sft_data.py      # Normalize instruction data (local / COIG-CQIA)
│   └── evaluate.py              # Compute average loss
├── analysis/
│   ├── activation.py      # Extract per-layer hidden states
│   └── attention_map.py   # Visualize attention weights
├── checkpoints_pretrain/  # Stage 1 outputs (auto-created)
├── checkpoints_sft/       # Stage 2 outputs (auto-created)
├── data/
│   └── raw/               # Corpora & SFT JSONL (generated, gitignored)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

### Stage 0 — Data & tokenizer

**Pretrain corpus** (recommended: Chinese Fineweb Edu, education-filtered and MinHash-deduplicated):

```bash
python scripts/download_fineweb_edu.py            # or download_chinese_c4.py
python scripts/train_tokenizer.py                 # -> data/tokenizer/tokenizer.model
python scripts/prepare_data.py                    # -> data/processed/train.bin
```

**SFT data** — COIG-CQIA via HF mirror, balanced-sampled to 5000 examples across all 13 subsets (30 major / 122 minor task types):

```bash
HF_ENDPOINT=https://hf-mirror.com python scripts/prepare_sft_data.py \
  --hf-dataset m-a-p/COIG-CQIA \
  --hf-configs chinese_traditional,coig_pc,exam,finance,douban,human_value,logi_qa,ruozhiba,segmentfault,wiki,wikihow,xhs,zhihu \
  --sample-strategy coig_cqia_balanced \
  --limit 5000 \
  --output-jsonl data/raw/coig_cqia_5k_normalized.jsonl \
  --output-text data/raw/coig_cqia_5k_train.txt
```

Local JSONL (`instruction/input/output` or `messages`) works too; see `--input-jsonl`.

### Stage 1 — Pretraining (several hours to ~10h)

```bash
python training/train.py
```

No `--sft-jsonl` means pretraining. Defaults (see [model/config.py](model/config.py)):

| Parameter | Default |
|---|---|
| checkpoint dir | `checkpoints_pretrain/` |
| learning rate | `3e-4` |
| epochs | `5` |
| warmup steps | `1000` |
| save frequency | `latest.pt` every `2000` steps + one `checkpoint_epochN.pt` per epoch |

Useful overrides:

```bash
python training/train.py --epochs 3 --save-every-steps 1000
```

Resume an interrupted pretrain:

```bash
python training/train.py --resume checkpoints_pretrain/latest.pt
```

### Stage 2 — SFT on COIG-CQIA

```bash
python training/train.py \
  --sft-jsonl data/raw/coig_cqia_5k_normalized.jsonl \
  --tokenizer data/tokenizer/tokenizer.model \
  --resume checkpoints_pretrain/latest.pt
```

Passing `--sft-jsonl` switches the stage automatically. Defaults become SFT-appropriate:

| Parameter | Default |
|---|---|
| checkpoint dir | `checkpoints_sft/` |
| learning rate | `5e-5` |
| epochs | `3` |
| warmup steps | `30` |
| save frequency | per-epoch + `latest.pt` |

Behavior details:

- Loss is computed **only on assistant tokens**; `系统：/用户：` prompt tokens are masked with `-100`.
- Cross-stage `--resume` (pretrain → SFT) loads **model weights only** and resets optimizer / scheduler / epoch counter, so the SFT run starts a fresh, shorter LR schedule. Same-stage `--resume` is a true continuation (optimizer + scheduler + epoch restored).
- Running SFT **without** `--resume` prints a warning, because that trains from random init (fine for smoke tests, not for real results).

### Stage 3 — Inference

```bash
python inference.py --prompt "请简要介绍一下人工智能。"
```

- Checkpoint auto-discovery order: `checkpoints_sft/latest.pt` → `checkpoints_pretrain/latest.pt` → `checkpoints/latest.pt`; override with `--checkpoint path/to/ckpt.pt`.
- If the prompt has no `系统：/用户：/助手：` markers it is wrapped into the chat template automatically.
- Sampling supports `--temperature`, `--top-k`, `--repetition-penalty`, and stops at EOS.

## Checkpoint layout

| Directory | Written by | Contents |
|---|---|---|
| `checkpoints_pretrain/` | Stage 1 | `checkpoint_epochN.pt` + `latest.pt` (refreshed every 2000 steps) |
| `checkpoints_sft/` | Stage 2 | `checkpoint_epochN.pt` + `latest.pt` |
| `checkpoints/` | legacy | Kept only for backward compatibility |

Each checkpoint stores `model`, `optimizer`, `scheduler`, `epoch`, `global_step`, `model_config`, and `train_args` (used to detect the stage on resume).

## License

MIT
