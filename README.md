# re-stitch-folder-tree
Offline-first data discovery &amp; identity correlation tool — scans disks, extracts content/metadata, links files to users with explainable confidence, and clusters orphaned data for cleanup/audit (NDJSON + cached rescans).

## What this is

`scan.py` is an offline-first CLI that scans a folder tree, extracts document content + metadata, correlates files to users, and produces:

- **`outputs/findings.ndjson`**: one JSON object per file (streamable, large-scale)
- **`outputs/summary.json`**: rollups, including orphan/ambiguous clustering with folder segmentation capped to **3 levels deep** and **top 3 folders** per level
- **`outputs/cache.sqlite`**: scan cache (incremental rescans)

## Why NDJSON instead of one big JSON

Real scans can be millions of files. A single giant JSON becomes slow, huge, and hard to open. NDJSON lets you stream results and keep “no cap” output realistic.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python3 scan.py \
  --root "/path/to/scan" \
  --users "samples/users.sample.json" \
  --outdir "outputs" \
  --min-confidence 0.80
```

## Key flags

- `--min-confidence` (**required**): must be `>= 0.80`
- `--per-user-cap`: max files to list per user (default 200). Use `0` for no cap.
- `--raw-mode`: `snippets` (default) or `full`
- `--max-bytes`: max bytes to read from plain text files (default 2MB)
- `--exclude-dir-glob` / `--exclude-path-glob`: add noise filters

## Privacy levels (roadmap)

Current v1 extracts locally and does **not** require an LLM.

- **Level A (default v1)**: local extraction + local matching
- **Level B (future)**: send derived signals (tokens/snippets) to an LLM provider
- **Level C (future)**: send content snippets to an LLM provider

## LLM connector (commented out by default)

There is a commented OpenAI-compatible connector stub in the code. You can wire it to:
- OpenAI / Azure OpenAI
- A self-hosted OpenAI-compatible endpoint

Only enable it if your privacy policy allows it.


