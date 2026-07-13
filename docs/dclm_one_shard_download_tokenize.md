# DCLM One-Shard Download And Tokenization

This creates a one-local-shard DCLM-baseline dataset in the same Megatron
indexed-dataset format used by this repo: one prefix with `.bin` and `.idx`
files.

Use this path when the full DCLM tokenized dataset is too large to copy. One
DCLM local shard is roughly 38B GPT-NeoX tokens, which is plenty for a run that
consumes about 3B tokens.

## Goal

Produce:

```text
$TOKENIZED_ROOT/merged_0.bin
$TOKENIZED_ROOT/merged_0.idx
```

Then train with a data path file containing the prefix:

```text
/datasets/products/mmlaion/language/tokenized/DCLM-baseline-1shard-gs03-ls01/GPT-NeoX/merged_0
```

Adjust `/datasets/...` to whatever path is visible inside the training
environment on the target machine.

## 1. Choose Directories

```bash
cd /path/to/Megatron-LM

RAW_ROOT=/path/to/datasets/raw/mlfoundations-dclm-baseline-1.0
TOKENIZED_ROOT=/path/to/datasets/products/mmlaion/language/tokenized/DCLM-baseline-1shard-gs03-ls01/GPT-NeoX

mkdir -p "$RAW_ROOT" "$TOKENIZED_ROOT"
```

Plan for at least 200 GB of free disk. Tokenization is CPU and disk heavy; the
8 A6000 GPUs are not needed for this step.

## 2. Install Download/Tokenization Dependencies

Use the same Python environment that will run Megatron if possible. Otherwise:

```bash
python -m pip install --upgrade pip
python -m pip install huggingface_hub transformers tokenizers
```

Install `zstdcat` if it is missing:

```bash
command -v zstdcat || conda install -y -c conda-forge zstd
```

No AWS account is needed for the commands below.

## 3. Download One DCLM Local Shard From Hugging Face

This uses the shard from the DCLM README example:
`global-shard_03_of_10/local-shard_1_of_10`.

```bash
huggingface-cli download mlfoundations/dclm-baseline-1.0 \
  --repo-type dataset \
  --include 'global-shard_03_of_10/local-shard_1_of_10/*.jsonl.zst' \
  --local-dir "$RAW_ROOT"
```

Expected raw files:

```bash
find "$RAW_ROOT/global-shard_03_of_10/local-shard_1_of_10" \
  -type f -name '*.jsonl.zst' | wc -l
```

This should be around 179 files for this specific local shard.

## 4. Cache The GPT-NeoX Tokenizer

```bash
python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
print("len:", len(tok), "vocab_size:", tok.vocab_size, "eos:", tok.eos_token_id)
PY
```

Expected:

```text
len: 50277 vocab_size: 50254 eos: 0
```

## 5. Tokenize Into Megatron Indexed Dataset Format

Megatron's preprocessor reads plain JSONL, while DCLM is stored as `.jsonl.zst`.
Use a FIFO so the raw shard is decompressed as a stream instead of writing a huge
temporary uncompressed file.

```bash
FIFO=/tmp/dclm_gs03_ls01.jsonl
rm -f "$FIFO"
mkfifo "$FIFO"

find "$RAW_ROOT/global-shard_03_of_10/local-shard_1_of_10" \
  -type f -name '*.jsonl.zst' -print0 \
  | sort -z \
  | xargs -0 -r zstdcat > "$FIFO" &

PYTHONPATH="$PWD" python tools/preprocess_data.py \
  --input "$FIFO" \
  --output-prefix "$TOKENIZED_ROOT/merged" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model EleutherAI/gpt-neox-20b \
  --json-keys text \
  --append-eod \
  --workers 32 \
  --log-interval 10000

rm -f "$FIFO"
```

The preprocessor writes:

```text
$TOKENIZED_ROOT/merged_text_document.bin
$TOKENIZED_ROOT/merged_text_document.idx
```

Rename these to match the existing DCLM style in this repo:

```bash
mv "$TOKENIZED_ROOT/merged_text_document.bin" "$TOKENIZED_ROOT/merged_0.bin"
mv "$TOKENIZED_ROOT/merged_text_document.idx" "$TOKENIZED_ROOT/merged_0.idx"
```

## 6. Create The Data Path File

The path in this file must be the path that training sees. If the host dataset
root is bind-mounted into the container as `/datasets`, use `/datasets/...`.

```bash
cat > configs/dclm_one_shard_gpt_neox_paths.txt <<'EOF'
/datasets/products/mmlaion/language/tokenized/DCLM-baseline-1shard-gs03-ls01/GPT-NeoX/merged_0
EOF
```

If training runs directly on the host, replace the line with the host path:

```text
/path/to/datasets/products/mmlaion/language/tokenized/DCLM-baseline-1shard-gs03-ls01/GPT-NeoX/merged_0
```

## 7. Update The Training Config

Use a new cache directory. Do not reuse a cache made for the old seven-prefix
DCLM dataset.

```yaml
data_args_path: ["configs/dclm_one_shard_gpt_neox_paths.txt"]
data_cache_path: ["/path/to/new/megatron_data_cache/dclm_one_shard_gpt_neox"]

split: ["98,1,1"]
tokenizer_type: ["NullTokenizer"]
vocab_size: [50277]
padded_vocab_size: [50304]
null_tokenizer_eod_id: [0]
null_tokenizer_pad_id: [-1]
```

Keep `NullTokenizer` for training because the data is already tokenized. The
Hugging Face tokenizer is only used during preprocessing.

## Notes

- This does not reproduce the exact same sample stream as the old full-DCLM
  runs. It creates a smaller, reproducible one-shard DCLM dataset.
- The training seed will be reproducible for this one-shard dataset if the
  data path file, split, sequence length, and cache directory contents are the
  same.
- If `run_megatron.sh` is used, make sure its bind mounts expose
  `$TOKENIZED_ROOT` at the same path written in
  `configs/dclm_one_shard_gpt_neox_paths.txt`.
