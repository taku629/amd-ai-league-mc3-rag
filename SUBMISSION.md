# MC-3 RAG submission — build & run notes

## Layout

```
submit/
  Dockerfile            # mandated ROCm base + baked model weights
  app/
    app.py              # thin unix-socket client (the contract entry point)
    server.py           # persistent model+index server (CMD)
    rag.py              # parse/index/retrieve/answer/citation pipeline
    requirements.txt
  models/
    llm/                # Qwen/Qwen3-VL-8B-Instruct  (17 GB, baked in)
    embed/              # BAAI/bge-small-en-v1.5    (~0.4 GB, baked in)
    llm-test/           # HuggingFaceTB/SmolVLM-256M-Instruct (local CPU test only)
```

## Build

```bash
cd submit
docker build -t mc3-rag:submit .
# local CPU test image (tiny model):
docker build -t mc3-rag:test --build-arg LLM_DIR=models/llm-test .
```

## Run

```bash
docker run -d --name rag --network none mc3-rag:submit
docker exec rag mkdir -p /app/corpus /app/output
docker cp <corpus>/. rag:/app/corpus/
docker exec rag python3 /app/app.py --index /app/corpus
docker exec rag python3 /app/app.py --corpus /app/corpus \
  --query-id query_01 --query "..."
# -> /app/output/query_01_output.json
```

The container CMD respawns `server.py` in a loop, so the container stays up
for the whole evaluation even if the server process ever dies. `app.py` only
talks to it over `/tmp/mc3-rag.sock`; it never loads models itself, and it
spawns the server as a fallback if the socket is dead.

## Architecture notes

- `--index` parses every document once, runs VLM OCR on images, and persists
  chunks + embeddings + skip metadata to `/app/index`.
- Queries retrieve by BM25+embedding RRF fusion (top 12 chunks, ≤2 per file),
  then run a second hop: identifier tokens found in the top chunks (ticket
  ids, part numbers like ORR-1847 / E7731) are re-queried to pull in linked
  files that share no vocabulary with the question - the multi-hop route.
  Retrieved image files are attached as pixels to the VLM prompt, then:
  JSON/salvage parse →
  citation cleaning → unit normalisation → hallucination guard →
  NEEDED/UNNEEDED necessity pruning → completeness check that can add a
  missing bridge file for multi-hop questions. Image-derived answers absent
  from OCR text are verified against the pixels (YES/NO) and the image is
  cited automatically, so a missing model citation does not cost the point.
- Generation stops as soon as one complete JSON object closes (latency guard,
  disable via `MC3_JSON_STOP=0`).
- Refusal contract: no evidence → `{"answer":"","citations":[],"confidence":0.0}`.
- Encrypted PDFs, unreadable files, unknown binaries and empty dirs are each
  caught and recorded in `skipped` without stopping the index.
- Env knobs (defaults shown): `MC3_LLM_PATH=/models/llm`,
  `MC3_EMBED_PATH=/models/embed`, `MC3_INDEX_DIR=/app/index`,
  `MC3_OUTPUT_DIR=/app/output`, `MC3_QUERY_RPC_TIMEOUT=280`,
  `MC3_MAX_NEW_TOKENS` (unset = full length), `MC3_DEBUG` (raw generation log).

## Local validation (CPU-only WSL, SmolVLM-256M test image)

- selfcheck.py: 5/6 PASS — size, no-network start, `--index` 312s/600s,
  output shape. Step 4 (30s/query) fails only because CPU generation is slow.
- Full-corpus e2e: query_01 -> `94` + [r2.pdf] ✅, query_02 -> `Q3 FY27` +
  [roadmap_fy27.docx] ✅, query_10 -> refusal ✅ (all exact).
- Broken files: permission-denied / encrypted PDF / unknown binary each
  caught and recorded; index continues; image OCR-budget fallback keeps a
  filename stub retrievable.
- torch stays `2.13.0+rocm10.0.0` after pip (--no-deps for torch-adjacent pkgs).
- NOT validated locally (needs the AMD eval box): Qwen3-VL-8B generation
  quality, per-query latency, VRAM peak, all-10 scoring.
