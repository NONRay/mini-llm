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
│   ├── convert_psycho_pdfs.py    # Convert psychology PDFs to Markdown
│   ├── prepare_psycho_data.py    # Clean Markdown into a plain-text corpus
│   ├── quality_filter_psycho.py  # LLM-based OCR quality filter for psychology corpus
│   ├── run_psycho_pretrain_prep.sh # Persistent GPU cleaning/QC pipeline
│   ├── export_book.py            # Export one reviewed book for inspection
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
│   ├── psycho/
│   │   ├── pdf/           # Local source PDFs (gitignored)
│   │   └── markdown/      # Converted Markdown (committed)
│   └── raw/               # Reviewed datasets allowlisted; other outputs ignored
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The virtual environment is local-only: `.venv/`, `venv/`, and similar environment
directories are excluded by `.gitignore`. Reactivate it with
`source .venv/bin/activate` whenever you open a new shell.

## Pipeline

### Stage 0 — Data & tokenizer

**Pretrain corpus** (recommended: Chinese Fineweb Edu, education-filtered and MinHash-deduplicated):

```bash
python scripts/download_fineweb_edu.py            # or download_chinese_c4.py
python scripts/train_tokenizer.py                 # -> data/tokenizer/tokenizer.model
python scripts/prepare_data.py                    # -> data/processed/train.bin
```

### Psychology corpus: PDF to reviewed pretraining text

This repository includes a reproducible OCR-cleaning pipeline for psychology books.
The central design is to separate **deterministic cleaning** from **model-based
quality decisions**:

- Rules remove artifacts whose form is known: Markdown syntax, page markers,
  repeated headers/footers, isolated page numbers, hard line breaks, and broken
  English words.
- Qwen3-8B only decides whether a chunk is useful (`keep`, `drop`, or `review`).
  It does not rewrite the source, which avoids silently introducing hallucinations.

#### Step 1: Convert PDFs to Markdown

Put local PDFs in `data/psycho/pdf/`, then run:

```bash
pip install "markitdown[pdf]"
python scripts/convert_psycho_pdfs.py
```

Text PDFs use MarkItDown; scanned PDFs fall back to MinerU. Existing Markdown files
are skipped, so rerunning resumes rather than starting over. To force remaining
files through OCR:

```bash
python scripts/convert_psycho_pdfs.py --mode mineru --mineru-args "-l ch"
```

Markdown is committed because it is searchable and diffable. Source PDFs and MinerU
temporary fragments remain local.

#### Step 2: Apply deterministic text cleaning

```bash
python scripts/prepare_psycho_data.py
# output: data/raw/psycho_train.txt (local intermediate file)
```

The cleaner:

1. Uses NFC Unicode normalization. NFKC is deliberately avoided because it changes
   Chinese punctuation such as `，；：（）` into ASCII punctuation.
2. Removes Markdown/HTML syntax and `<!-- page N of M -->` markers.
3. Detects isolated page-number lines and repeated short headers/footers.
4. Joins lines split by page layout while preserving paragraphs and headings.
5. Repairs English line wrapping without splitting words, and inserts missing
   spaces after Latin punctuation.
6. Normalizes ASCII punctuation only in an unambiguous Chinese context, while
   leaving citations such as `S.Freud,1856` unchanged.

This stage is deterministic and fast. Inspect it before spending GPU time on review.

#### Step 3: Review chunks with local Qwen3-8B

The tested model path is `/root/autodl-tmp/models/Qwen3-8B`. Local inference is
forced onto `cuda:0`; the script exits instead of silently falling back to CPU.

Run the complete pipeline as a persistent background job:

```bash
CUDA_VISIBLE_DEVICES=0 nohup \
  scripts/run_psycho_pretrain_prep.sh \
  </dev/null >/dev/null 2>&1 & echo $!
```

The wrapper verifies CUDA allocation, rebuilds the rule-cleaned text, then runs Qwen
review. It is safe to disconnect from SSH.

Chunks are at most 1800 characters. Long English paragraphs are cut at sentence
boundaries or spaces, never inside a word. A `drop` decision is accepted only at
confidence `>= 0.85`; lower-confidence drops become `review` and remain in the
corpus. Invalid JSON is parsed defensively, retried, and ultimately retained as
`review` instead of stopping the run.

| File | Purpose | Git policy |
|---|---|---|
| `data/raw/psycho_train.txt` | Rule-cleaned intermediate corpus | ignored |
| `data/raw/psycho_train_qc.txt` | Final reviewed pretraining text | committed |
| `data/raw/psycho_qc_audit.jsonl` | One decision per current chunk | committed |
| `logs/psycho_prep_*.log` | Runtime progress and warnings | ignored |

Successful completion compacts the audit file, removing stale decisions from older
cleaning or chunking versions.

#### Step 4: Monitor and resume

```bash
# Follow the newest log.
tail -f "$(ls -t logs/psycho_prep_*.log | head -1)"

# Check process and GPU activity.
ps aux | grep quality_filter_psycho | grep -v grep
watch -n 5 nvidia-smi
```

The audit is flushed after every decision. If interrupted, launch the same command:
matching chunks are loaded from cache. The final corpus is written through a
temporary file and atomically renamed, so an interruption cannot publish a partial
result.

#### Step 5: Inspect or export one book

```bash
# Deterministic cleaning only.
python scripts/export_book.py \
  --book "20世纪西方现代心理学--人类心灵的神话：荣格的分析心理学.md" \
  --clean-only

# Export the Qwen-reviewed version.
python scripts/export_book.py \
  --book "20世纪西方现代心理学--人类心灵的神话：荣格的分析心理学.md"
```

Exports under `data/raw/psycho_books/` are inspection artifacts and are ignored.
The exporter validates cache keys rather than trusting chunk indexes, so old
decisions cannot be applied to newly cleaned text.

#### Step 6: Validate the final corpus

The reference run processed 51 books:

| Decision | Chunks |
|---|---:|
| keep | 5,541 |
| review (kept) | 1,070 |
| drop | 581 |
| total | 7,192 |

Validation found no HTML page markers, known `Thi` / `s` English word splits, or
`measure.For` joins, and confirmed restoration of Chinese punctuation. The reviewed
corpus is approximately 28.7 MB.

#### Step 7: Build tokenizer and binary training data

Tokenizer models and `.bin` files are local build artifacts:

```bash
python scripts/train_tokenizer.py \
  --input data/raw/psycho_train_qc.txt \
  --model-prefix data/tokenizer/psycho_tokenizer

python scripts/prepare_data.py \
  --input data/raw/psycho_train_qc.txt \
  --tokenizer data/tokenizer/psycho_tokenizer.model \
  --output data/processed/psycho_train.bin
```

Before distributing Markdown or reviewed text, verify that you have the right to
redistribute the underlying books. This pipeline does not grant content licenses.

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

## Reference run

Results from a complete pretrain → SFT cycle on a single GPU, as a calibration point for future runs:

| Stage | Steps | Wall clock | Loss | Notes |
|---|---|---|---|---|
| Pretrain | 268,000 | ~26.6 h | ~3.85 (plateaued) | ~17.5B tokens seen (≈38 passes over the 460M-token corpus); effective batch 8×8×1024 ≈ 65k tokens/step; ~2.8 steps/s |
| SFT | 234 (3 epochs × 78) | ~2 min | 4.53 → ~3.6 | Initial SFT loss ~4.5 confirms the pretrained base was loaded (vs ~350 when training from scratch) |

Observed behavior:

- **Base model** (`checkpoints_pretrain/latest.pt`): fluent grammar, but rambles in web-corpus style and ignores questions — the expected state before alignment.
- **SFT model** (`checkpoints_sft/latest.pt`): answer-style responses within the `系统：/用户：/助手：` template, stops at EOS.
- **Known limitations at this scale**: domain-skewed knowledge (the corpus sample is agriculture-heavy), occasional UNK tokens (`⁇`) from vocab coverage, and token repetition. Repetition can be mitigated at inference time with `--temperature 0.7 --top-k 20 --repetition-penalty 1.2`.

Practical notes:

- One DataLoader "epoch" of the sliding-window corpus ≈ 7.15M optimizer steps, so bound pretraining with `--max-steps` rather than `--epochs`. At this scale the cosine schedule stays near peak LR the whole run (total steps are effectively infinite), which is why loss plateaus — that is normal.
- `Ctrl+C` is safe during pretraining: `checkpoints_pretrain/latest.pt` is refreshed every 2000 steps, and `--resume checkpoints_pretrain/latest.pt` continues the run.
- Suggested next steps in order of impact: broaden pretrain corpus diversity (fixes domain skew), scale SFT data toward 20k–50k examples, then tune sampling params.

## License

MIT
