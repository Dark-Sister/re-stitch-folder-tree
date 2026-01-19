# re-stitch-folder-tree
Offline-first data discovery &amp; identity correlation tool — scans disks, extracts content/metadata, links files to users with explainable confidence, and clusters orphaned data for cleanup/audit (NDJSON + cached rescans).

## What this is

`scan.py` is an **LLM-driven** CLI that scans a folder tree, extracts document content + metadata, **clusters files**, and uses an **OpenAI-compatible LLM endpoint** to correlate clusters/files to users. It produces:

- **`outputs/findings.ndjson`**: one JSON object per file (streamable, large-scale)
- **`outputs/summary.json`**: rollups, including orphan/ambiguous clustering with folder segmentation capped to **3 levels deep** and **top 3 folders** per level
- **`outputs/cache.sqlite`**: scan cache (incremental rescans)
- **`outputs/cluster_labels.json`**: LLM cluster attribution outputs (debuggable)

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
  --min-confidence 0.80 \
  --llm-api-key "$OPENAI_API_KEY"
```

## Key flags

- `--min-confidence` (**required**): must be `>= 0.80`
- `--per-user-cap`: max files to list per user (default 200). Use `0` for no cap.
- `--raw-mode`: `snippets` (default) or `full`
- `--match-chars`: chars used for matching (default 50k; independent of snippet size)
- `--max-bytes`: max bytes to read from plain text files (default 2MB)
- `--exclude-dir-glob` / `--exclude-path-glob`: add noise filters
- `--llm-endpoint`: OpenAI-compatible endpoint (default: OpenAI)
- `--llm-model`: model name (default: `gpt-4o-mini`)
- `--llm-dry-run`: build clusters but don't call the LLM

## Privacy warning

This version sends **raw extracted content** (snippets/full, depending on flags) to the configured LLM endpoint for attribution. Do not use on sensitive data unless you understand and accept the exposure.

