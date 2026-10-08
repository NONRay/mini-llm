# mini-llm

A minimal from-scratch LLM implementation in PyTorch, featuring a LLaMA-style decoder architecture (RMSNorm + SwiGLU + causal self-attention) with tied input/output embeddings.

Training follows a layered pipeline: **pretraining** on a raw Chinese corpus (Chinese C4), optional **domain-adaptive pretraining** on a reviewed psychology corpus, then **SFT** on instruction data (COIG-CQIA and/or generated psychology QA). Each stage writes to its own checkpoint directory so a later stage can never overwrite an earlier base.

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
│   ├── generate_psycho_sft.py    # Grounded psychology QA generation (Qwen3-8B)
│   ├── run_psycho_pretrain_prep.sh # Persistent GPU cleaning/QC pipeline
│   ├── export_book.py            # Export one reviewed book for inspection
│   ├── train_tokenizer.py       # Train BPE tokenizer
│   ├── prepare_data.py          # Tokenize raw text -> train.bin (memmap)
│   ├── prepare_sft_data.py      # Normalize instruction data (local / COIG-CQIA)
│   └── evaluate.py              # Compute average loss
├── analysis/
│   ├── activation.py      # Extract per-layer hidden states
│   └── attention_map.py   # Visualize attention weights
├── checkpoints_pretrain/       # Stage 1 outputs (auto-created)
├── checkpoints_psycho_pretrain/ # Domain-adaptive pretrain outputs
├── checkpoints_sft/            # General SFT outputs
├── checkpoints_psycho_sft*/    # Psychology SFT outputs (pure / mixed)
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

#### Step 7: Build binary training data

Continued pretraining must use the original Chinese C4 tokenizer. Training a new
psychology tokenizer would assign different meanings to the checkpoint's existing
embedding rows. Tokenize the reviewed corpus with the original model:

```bash
python scripts/prepare_data.py \
  --input data/raw/psycho_train_qc.txt \
  --tokenizer data/tokenizer/tokenizer.model \
  --output data/processed/psycho_train.bin
```

Before distributing Markdown or reviewed text, verify that you have the right to
redistribute the underlying books. This pipeline does not grant content licenses.

#### Step 8: Continue pretraining on psychology text

Load the Chinese C4 model weights while resetting the optimizer, scheduler, epoch,
and step counters for the new corpus. Keep the domain checkpoints in a separate
directory:

```bash
python training/train.py \
  --train-bin data/processed/psycho_train.bin \
  --learning-rate 3e-5 \
  --warmup-steps 30 \
  --max-steps 300 \
  --save-every-steps 100 \
  --checkpoint-dir checkpoints_psycho_pretrain \
  --resume checkpoints_pretrain/latest.pt \
  --reset-training-state
```

#### Step 9: Generate grounded psychology SFT data

The generator reconstructs only high-confidence `keep` chunks (confidence >= 0.9,
400–1800 chars) from the QC audit, re-validates their cache keys against the current
Markdown, and uses local Qwen3-8B to create standalone question/answer pairs. The
model may reject a chunk that cannot become self-contained QA; such chunks are
audited as skips. Generation is cached after every sample (interrupt-safe resume)
and requires CUDA — the script exits rather than falling back to CPU:

```bash
CUDA_VISIBLE_DEVICES=0 nohup \
  python -u scripts/generate_psycho_sft.py --limit 1000 \
  </dev/null >logs/psycho_sft_generate.log 2>&1 &
```

Tasks rotate across four types — concept explanation, comparison, application
analysis, misconception correction — and candidates round-robin across books so no
single author dominates. Prompts require questions that stand alone without the
source chunk, answers grounded in the chunk only, and theory/fact boundaries
for classic (Jung/Freud) material.

Reference run over the 5,520 eligible chunks:

| Metric | Value |
|---|---|
| samples written | 1,000 |
| model calls | 1,048 (2 cached from smoke test, 50 rejected) |
| books covered | 50 (3–22 samples per book) |
| task balance | 250 / 252 / 249 / 249 |
| duplicate instructions | 0 |
| context leakage (`根据上述文本` etc.) | 0 |
| avg output length | ~229 chars |

| File | Purpose | Git policy |
|---|---|---|
| `data/raw/psycho_sft_train.jsonl` | Training samples (`system` / `instruction` / `output` + source metadata) | committed |
| `data/raw/psycho_sft_generation_audit.jsonl` | One decision per chunk (keep/skip + parsed JSON) | committed |
| `logs/psycho_sft_generate*.log` | Runtime progress | ignored |

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

### Stage 2b — Psychology SFT

Two variants, both starting from the domain-adapted checkpoint
(`checkpoints_psycho_pretrain/latest.pt`, loaded weights-only with optimizer reset):

**Pure domain** (1,000 psychology samples, 3 epochs → 96 steps, ~1 min):

```bash
python training/train.py \
  --sft-jsonl data/raw/psycho_sft_train.jsonl \
  --tokenizer data/tokenizer/tokenizer.model \
  --batch-size 8 --grad-accum-steps 4 \
  --epochs 3 --learning-rate 5e-5 --warmup-steps 10 \
  --checkpoint-dir checkpoints_psycho_sft \
  --resume checkpoints_psycho_pretrain/latest.pt
```

Per-epoch running-average loss: 5.75 → 5.09 → 4.89.

**Domain-weighted mix** (psycho ×2 = 2,000 + COIG-CQIA 5,000 = 7,000 rows,
3 epochs → 657 steps). The mixed file is derived and regenerable, so it is not
committed; rebuild it with:

```bash
python - <<'PY'
import json, random
from pathlib import Path

def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

domain, general = load("data/raw/psycho_sft_train.jsonl"), load("data/raw/coig_cqia_5k_normalized.jsonl")
rows = [dict(s, mix_group="psycho", mix_repeat=i) for i in range(2) for s in domain]
rows += [dict(s, mix_group="general") for s in general]
random.Random(42).shuffle(rows)
Path("data/raw/psycho_sft_mixed_train.jsonl").write_text(
    "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
PY
```

Then train with `--sft-jsonl data/raw/psycho_sft_mixed_train.jsonl --warmup-steps 30
--log-interval 20 --checkpoint-dir checkpoints_psycho_sft_mixed` (same other flags).

Inference from either checkpoint:

```bash
python inference.py \
  --checkpoint checkpoints_psycho_sft_mixed/latest.pt \
  --prompt "请解释荣格理论中的集体无意识。" \
  --max-tokens 180 --temperature 0.7 --top-k 20 --repetition-penalty 1.2
```

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
| `checkpoints_psycho_pretrain/` | Step 8 (domain-adaptive pretrain) | `checkpoint_epoch0.pt` + `latest.pt` (refreshed every 100 steps) |
| `checkpoints_sft/` | Stage 2 | `checkpoint_epochN.pt` + `latest.pt` |
| `checkpoints_psycho_sft/` | Stage 2b (pure domain) | `checkpoint_epochN.pt` + `latest.pt` |
| `checkpoints_psycho_sft_mixed/` | Stage 2b (domain-weighted mix) | `checkpoint_epochN.pt` + `latest.pt` |
| `checkpoints/` | legacy | Kept only for backward compatibility |

Each checkpoint stores `model`, `optimizer`, `scheduler`, `epoch`, `global_step`, `model_config`, and `train_args` (used to detect the stage on resume).

## Reference run

Results from a complete pretrain → SFT cycle on a single GPU, as a calibration point for future runs:

| Stage | Steps | Wall clock | Loss | Notes |
|---|---|---|---|---|
| Pretrain | 268,000 | ~26.6 h | ~3.85 (plateaued) | ~17.5B tokens seen (≈38 passes over the 460M-token corpus); effective batch 8×8×1024 ≈ 65k tokens/step; ~2.8 steps/s |
| SFT | 234 (3 epochs × 78) | ~2 min | 4.53 → ~3.6 | Initial SFT loss ~4.5 confirms the pretrained base was loaded (vs ~350 when training from scratch) |
| Psycho domain pretrain | 300 | ~2 min | 19.8 → ~7.5 (running avg) | 6.9M tokens ≈ 3 passes; lr 3e-5; weights-only resume via `--reset-training-state` |
| Psycho SFT (pure) | 96 (3 epochs × 32) | ~1 min | 5.75 → 5.09 → 4.89 | 1,000 generated samples; effective batch 8×4 |
| Psycho SFT (mixed) | 657 (3 epochs × 219) | ~3 min | ~7.6 → ~5.9 | 2,000 psycho (×2 weight) + 5,000 COIG-CQIA |

Observed behavior:

- **Base model** (`checkpoints_pretrain/latest.pt`): fluent grammar, but rambles in web-corpus style and ignores questions — the expected state before alignment.
- **SFT model** (`checkpoints_sft/latest.pt`): answer-style responses within the `系统：/用户：/助手：` template, stops at EOS.
- **Psycho SFT models** (`checkpoints_psycho_sft*/latest.pt`): answer-style Chinese responses within the same template, with a psychology-tutor system prompt baked into the generated data.
- **Known limitations at this scale**: domain-skewed knowledge (the corpus sample is agriculture-heavy), occasional UNK tokens (`⁇`) from vocab coverage, and token repetition. Repetition can be mitigated at inference time with `--temperature 0.7 --top-k 20 --repetition-penalty 1.2`.

#### Psychology model: effects and limitations

What the psycho SFT checkpoints can realistically do:

- Answer questions about classic analytical-psychology concepts that the training
  corpus covers densely — collective unconscious, archetypes, complexes, persona,
  introversion/extraversion, individuation, Freud–Jung differences, dream/symbol
  interpretation, transference.
- Produce plausible short explanations and comparisons in the domain's register,
  stopping at EOS within the chat template.
- Serve as a compact demo of the full pipeline: book PDFs → cleaned corpus →
  reviewed pretraining text → domain-adaptive pretraining → generated QA → SFT.

Honest limitations (verified by held-out inference, e.g. "什么是投射"):

- **Semantic drift on unseen questions**: the model stays in Q&A format but drifts
  off-topic; output looks fluent while being unreliable. Adding 5k general
  instructions (mixed run, 657 steps) did not fix this — the bottleneck is not
  SFT data volume.
- **Bottleneck is the 50.5M-parameter base** and its pretraining scale (~460M
  unique tokens). More SFT epochs on 1k–7k samples would overfit rather than
  generalize.
- Do not treat it as a source of psychological facts, diagnoses, or crisis advice.

Recommended next step: reuse this data pipeline (QC → domain pretrain corpus →
grounded QA generation) on a 0.5B–1.5B Chinese base with QLoRA on the same 24GB
GPU; the generated `psycho_sft_train.jsonl` is model-agnostic JSONL.

Practical notes:

- One DataLoader "epoch" of the sliding-window corpus ≈ 7.15M optimizer steps, so bound pretraining with `--max-steps` rather than `--epochs`. At this scale the cosine schedule stays near peak LR the whole run (total steps are effectively infinite), which is why loss plateaus — that is normal.
- `Ctrl+C` is safe during pretraining: `checkpoints_pretrain/latest.pt` is refreshed every 2000 steps, and `--resume checkpoints_pretrain/latest.pt` continues the run.
- Suggested next steps in order of impact: broaden pretrain corpus diversity (fixes domain skew), scale SFT data toward 20k–50k examples, then tune sampling params.

## License

MIT
