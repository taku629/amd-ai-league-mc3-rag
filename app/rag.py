"""MC-3 RAG core: corpus parsing, indexing, retrieval and answering.

Everything here runs inside the persistent server process. The heavy objects
(models, the parsed index) are loaded once and reused across queries.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

INDEX_DIR = Path(os.environ.get("MC3_INDEX_DIR", "/app/index"))

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

TEXT_EXTS = {
    ".txt", ".log", ".md", ".py", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".cfg", ".conf", ".sh", ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".java",
    ".js", ".ts", ".xml", ".html", ".htm", ".sql", ".r", ".lua", ".pl", ".rb",
}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def _is_probably_text(data: bytes) -> bool:
    if not data:
        return False
    sample = data[:4096]
    if b"\x00" in sample:
        return False
    printable = sum(1 for b in sample if b in b"\t\n\r" or 32 <= b < 127 or b >= 128)
    return printable / len(sample) > 0.75


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    # Encrypted files are never opened - even when an empty password would
    # decrypt them. Their content must never feed an answer.
    if reader.is_encrypted:
        raise ValueError("encrypted pdf")
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            parts.append("")
    return "\n\f\n".join(parts)


_WNS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _docx_cell_text(tc) -> str:
    return " ".join(t.text or "" for t in tc.iter(_WNS + "t"))


def _parse_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    body = root.find(_WNS + "body")
    lines = []

    def walk(el):
        for child in el:
            tag = child.tag
            if tag == _WNS + "p":
                txt = "".join(t.text or "" for t in child.iter(_WNS + "t"))
                if txt.strip():
                    lines.append(txt)
            elif tag == _WNS + "tbl":
                for tr in child.iter(_WNS + "tr"):
                    cells = [_docx_cell_text(tc) for tc in tr.iter(_WNS + "tc")]
                    if any(c.strip() for c in cells):
                        lines.append(" | ".join(cells))
            elif tag == _WNS + "sdt":
                content = child.find(_WNS + "sdtContent")
                if content is not None:
                    walk(content)
            else:
                walk(child)

    walk(body)
    return "\n".join(lines)


_XNS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_RNS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"


def _xlsx_shared_strings(z) -> list[str]:
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    return ["".join(t.text or "" for t in si.iter(_XNS + "t")) for si in root.iter(_XNS + "si")]


def _xlsx_cell_value(c, shared) -> str:
    t = c.get("t")
    if t == "inlineStr":
        is_el = c.find(_XNS + "is")
        if is_el is None:
            return ""
        return "".join(x.text or "" for x in is_el.iter(_XNS + "t"))
    v = c.find(_XNS + "v")
    if v is None or v.text is None:
        return ""
    if t == "s":
        try:
            return shared[int(v.text)]
        except (ValueError, IndexError):
            return ""
    if t == "str" or t == "b":
        return v.text
    return v.text


def _xlsx_rows(z, sheet_path: str, shared) -> list[list[str]]:
    root = ET.fromstring(z.read(sheet_path))
    data = root.find(_XNS + "sheetData")
    if data is None:
        return []
    rows = []
    for row in data.iter(_XNS + "row"):
        cells = {}
        for c in row.findall(_XNS + "c"):
            ref = c.get("r", "")
            col = re.match(r"[A-Z]+", ref)
            col = col.group(0) if col else ""
            val = _xlsx_cell_value(c, shared)
            cells[col] = val
        if cells:
            ordered = [cells[k] for k in sorted(cells, key=lambda x: (len(x), x))]
            rows.append(ordered)
    return rows


def _parse_xlsx(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        shared = _xlsx_shared_strings(z)
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rid_to_target = {}
        for rel in rels:
            rid_to_target[rel.get("Id")] = rel.get("Target")
        out = []
        for sheet in wb.iter(_XNS + "sheet"):
            name = sheet.get("name", "sheet")
            rid = sheet.get(_RNS + "id")
            target = rid_to_target.get(rid, "")
            if not target:
                continue
            target = target.lstrip("/")
            sheet_path = target if target.startswith("xl/") else "xl/" + target
            if sheet_path not in z.namelist():
                continue
            rows = _xlsx_rows(z, sheet_path, shared)
            if not rows:
                continue
            out.append(f"[sheet: {name}]")
            header = rows[0] if len(rows) > 1 else None
            for i, row in enumerate(rows):
                if header and i > 0 and len(header) == len(row) and all(
                    h and not h.replace(".", "").replace("-", "").isdigit() for h in header
                ):
                    out.append(" | ".join(f"{h}={v}" for h, v in zip(header, row)))
                else:
                    out.append(" | ".join(row))
    return "\n".join(out)


def _parse_csv(path: Path) -> str:
    text = _read_text(path)
    try:
        rows = list(csv.reader(io.StringIO(text)))
    except Exception:
        return text
    if not rows:
        return text
    header = rows[0]
    has_header = all(h and not re.fullmatch(r"[-\d.]+", h) for h in header) and len(rows) > 1
    out = []
    for i, row in enumerate(rows):
        if has_header and i == 0:
            continue
        if has_header and len(row) == len(header):
            out.append(" | ".join(f"{h}={v}" for h, v in zip(header, row)))
        else:
            out.append(" | ".join(row))
    return "\n".join(out)


def iter_corpus(corpus: Path):
    for root, dirs, files in os.walk(corpus):
        dirs.sort()
        for name in sorted(files):
            p = Path(root) / name
            yield p, p.relative_to(corpus).as_posix()


def _parse_nonimage(path: Path, ext: str):
    """Parse one non-image file. Raises on failure (caller catches)."""
    if ext == ".pdf":
        return _parse_pdf(path), "pdf"
    if ext == ".docx":
        return _parse_docx(path), "docx"
    if ext == ".xlsx":
        return _parse_xlsx(path), "xlsx"
    if ext == ".csv":
        return _parse_csv(path), "csv"
    if ext in TEXT_EXTS:
        return _read_text(path), "text"
    data = path.read_bytes()
    if _is_probably_text(data):
        return data.decode("utf-8", "replace"), "text"
    raise ValueError("unknown/binary type")


def parse_corpus(corpus: Path, ocr_fn=None, ocr_budget_s=240, parse_budget_s=300):
    """Two passes: all non-image files first (they are cheap and must not be
    starved by a slow OCR), then images with their own OCR budget.

    docs:    [{path, text, kind}]
    skipped: [{path, reason}]
    Never raises on a single file.
    """
    docs, skipped = [], []
    entries = list(iter_corpus(corpus))

    t_deadline = time.time() + parse_budget_s
    for path, rel in entries:
        ext = path.suffix.lower()
        if ext in IMAGE_EXTS:
            continue
        if time.time() > t_deadline:
            skipped.append({"path": rel, "reason": "index time budget reached"})
            continue
        try:
            text, kind = _parse_nonimage(path, ext)
            docs.append({"path": rel, "text": text, "kind": kind})
        except PermissionError:
            skipped.append({"path": rel, "reason": "permission denied"})
        except Exception as e:
            skipped.append({"path": rel, "reason": f"{type(e).__name__}: {e}"[:200]})

    ocr_deadline = time.time() + ocr_budget_s
    for path, rel in entries:
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        text = ""
        if ocr_fn is not None:
            if time.time() > ocr_deadline:
                skipped.append({"path": rel, "reason": "ocr budget reached"})
            else:
                try:
                    text = ocr_fn(path) or ""
                except PermissionError:
                    skipped.append({"path": rel, "reason": "permission denied"})
                    continue
                except Exception as e:
                    skipped.append({"path": rel,
                                    "reason": f"{type(e).__name__}: {e}"[:200]})
        # Always keep a stub doc so the file stays retrievable (by name) and
        # the query-time VLM can still read the actual pixels.
        docs.append({"path": rel, "kind": "image",
                     "text": f"Image file: {rel}\n{text}"})

    docs = [d for d in docs if d["text"].strip()]
    return docs, skipped


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, size=1300, overlap=180):
    lines = text.split("\n")
    chunks, cur, cur_len = [], [], 0
    for line in lines:
        if cur_len + len(line) + 1 > size and cur:
            chunks.append("\n".join(cur))
            tail = "\n".join(cur)[-overlap:]
            cur, cur_len = [tail, line], len(tail) + len(line) + 1
        else:
            cur.append(line)
            cur_len += len(line) + 1
    if cur:
        chunks.append("\n".join(cur))
    return [c for c in (c.strip() for c in chunks) if c]


SUPERSEDED_RE = re.compile(r"WITHDRAWN|SUPERSEDED|superseded|withdrawn", re.I)


def doc_status(doc) -> str:
    head = doc["text"][:250]
    name = doc["path"].lower()
    if SUPERSEDED_RE.search(head) or "withdrawn" in name:
        return "superseded"
    if re.search(r"\bsupersedes\b", head, re.I):
        return "current"
    return "normal"


# ---------------------------------------------------------------------------
# BM25 (dependency-free)
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._\-#][A-Za-z0-9]+)*")


def tokenize(s: str):
    return [t.lower() for t in TOKEN_RE.findall(s)]


class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = docs_tokens
        self.n = len(docs_tokens)
        self.dl = [len(d) for d in docs_tokens]
        self.avgdl = sum(self.dl) / max(1, self.n)
        df = {}
        for d in docs_tokens:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tfs = []
        for d in docs_tokens:
            tf = {}
            for t in d:
                tf[t] = tf.get(t, 0) + 1
            self.tfs.append(tf)

    def scores(self, query_tokens):
        out = [0.0] * self.n
        for t in query_tokens:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tfs):
                f = tf.get(t, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                    out[i] += idf * f * (self.k1 + 1) / denom
        return out


# ---------------------------------------------------------------------------
# Embedding helper (HF transformers, no sentence-transformers dep)
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(self, model_dir: str, device: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_dir, torch_dtype=dtype, trust_remote_code=False
        ).to(device).eval()
        self.torch = torch

    def embed(self, texts):
        torch = self.torch
        with torch.no_grad():
            enc = self.tok(texts, padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(self.device)
            out = self.model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(pooled, dim=-1)


# ---------------------------------------------------------------------------
# Index persistence
# ---------------------------------------------------------------------------

def save_index(docs, chunks, embeddings, skipped):
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    import numpy as np

    with open(INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    if embeddings is not None:
        np.save(INDEX_DIR / "emb.npy", np.asarray(embeddings, dtype="float32"))
    meta = {
        "files": [d["path"] for d in docs],
        "status": {d["path"]: doc_status(d) for d in docs},
        "kinds": {d["path"]: d["kind"] for d in docs},
        "skipped": skipped,
        "built": time.time(),
    }
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta, indent=2))


def load_index():
    import numpy as np

    chunks = [json.loads(l) for l in open(INDEX_DIR / "chunks.jsonl", encoding="utf-8")]
    emb = None
    if (INDEX_DIR / "emb.npy").exists():
        emb = np.load(INDEX_DIR / "emb.npy")
    meta = json.loads((INDEX_DIR / "meta.json").read_text()) if (INDEX_DIR / "meta.json").exists() else {}
    return chunks, emb, meta


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(chunks, emb, query, q_emb, top_k=12, per_file=2):
    """Dense + BM25 RRF fusion."""
    n = len(chunks)
    if n == 0:
        return []
    order_dense = []
    if emb is not None and q_emb is not None:
        import numpy as np
        scores = emb @ q_emb
        order_dense = list(np.argsort(-scores)[: top_k * 2])
    bm = BM25([tokenize(c["text"]) for c in chunks])
    bscores = bm.scores(tokenize(query))
    order_bm = list(sorted(range(n), key=lambda i: -bscores[i])[: top_k * 2])

    rrf = {}
    for rank, i in enumerate(order_dense):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rank + 1)
    for rank, i in enumerate(order_bm):
        rrf[i] = rrf.get(i, 0.0) + 1.0 / (60 + rank + 1)
    ranked = sorted(rrf, key=lambda i: -rrf[i])

    picked, file_count = [], {}
    for i in ranked:
        f = chunks[i]["file"]
        if file_count.get(f, 0) >= per_file:
            continue
        file_count[f] = file_count.get(f, 0) + 1
        picked.append(chunks[i])
        if len(picked) >= top_k:
            break
    return picked


# ---------------------------------------------------------------------------
# Answering
# ---------------------------------------------------------------------------

ANSWER_INSTRUCTION = """You answer questions using ONLY the context blocks below, each labelled [FILE: path].

Rules:
- "answer" is the exact VALUE only: a number, part number, version, code, quarter, revision, or name. No sentences, no explanation. Keep qualifiers that belong to the value itself (e.g. "Q3 FY27", "REV-C2", "Meridian 4.3.2").
- Use ONLY the context. These products are fictional - ignore any outside knowledge. If the context does not determine the answer, output {"answer": "", "citations": []}.
- If documents conflict and one is marked WITHDRAWN or superseded, use the current version's value.
- "citations" lists ONLY the files whose content was necessary to derive the answer - every file without which the answer could not be found. If you used two files in a chain (e.g. an identifier from one file, then the value from another), list both. Do NOT list files that merely discuss the topic.
- Reply with exactly one JSON object on one line, nothing else:
{"answer": "...", "citations": ["path", ...]}
"""


def build_prompt(query: str, context_blocks: list[tuple[str, str, str]]):
    ctx = []
    for path, text, status in context_blocks:
        note = ""
        if status == "superseded":
            note = "  [NOTE: this file is WITHDRAWN/superseded - values may be stale]"
        elif status == "current":
            note = "  [NOTE: this is the CURRENT revision - prefer its values]"
        ctx.append(f"[FILE: {path}]{note}\n{text}")
    context = "\n\n".join(ctx)
    return (
        f"{ANSWER_INSTRUCTION}\nContext:\n{context}\n\n"
        f"Question: {query}\n"
        'JSON answer:'
    )


def extract_json(text: str):
    # Collect every top-level balanced object; prefer the first that parses
    # and actually carries an "answer" key - the model may emit extra JSON
    # after the real one.
    depth = 0
    start = -1
    cands = []
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                cands.append(text[start : i + 1])
    if not cands:
        return salvage_json(text)
    first_valid = None
    for m in cands:
        try:
            obj = json.loads(m)
        except Exception:
            try:
                fixed = re.sub(r",\s*}", "}", m)
                fixed = re.sub(r",\s*]", "]", fixed)
                obj = json.loads(fixed)
            except Exception:
                continue
        if first_valid is None:
            first_valid = obj
        if isinstance(obj, dict) and "answer" in obj:
            return obj
    if first_valid is not None:
        return first_valid
    return salvage_json(text)


def salvage_json(text: str):
    """Last-resort parse for generation truncated mid-object: pull the answer
    string and any citation strings out of a partial JSON body."""
    m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if not m:
        return None
    out = {"answer": m.group(1), "citations": []}
    cm = re.search(r'"citations"\s*:\s*\[([^\]]*)', text)
    if cm:
        out["citations"] = re.findall(r'"((?:[^"\\]|\\.)*)"', cm.group(1))
    return out


def normalize_answer(ans: str) -> str:
    ans = (ans or "").strip().strip("`").strip()
    ans = ans.split("\n")[0].strip()
    ans = re.sub(r"^['\"]|['\"]$", "", ans)
    ans = re.sub(
        r"(?i)^(?:the\s+)?(?:answer|value|result)(?:\s+is)?[:\s]+", "", ans
    ).strip()
    ans = re.sub(r"\.$", "", ans).strip()
    if ans.lower() in {"i don't know", "unknown", "n/a", "none", "not found",
                       "cannot be determined", "unanswerable", "no answer"}:
        return ""
    return ans


def _answer_in_text(ans: str, text: str) -> bool:
    a = re.escape(ans.strip())
    if not a:
        return False
    return re.search(rf"(?<![A-Za-z0-9]){a}(?![A-Za-z0-9])", text, re.I) is not None


def _token_in_text(tok: str, text: str) -> bool:
    """Like _answer_in_text but lets a letter follow the token, so the bare
    value "94" matches the way it is written in a document ("94C")."""
    t = re.escape(tok.strip())
    if not t:
        return False
    return re.search(rf"(?<![A-Za-z0-9]){t}(?![0-9.])", text, re.I) is not None


class RagEngine:
    """Holds models + index; services index/query requests."""

    def __init__(self, llm, embedder, device):
        self.llm = llm
        self.embedder = embedder
        self.device = device
        self.chunks = []
        self.emb = None
        self.meta = {}
        self.corpus_docs = {}
        self.file_status = {}
        self.images = {}

    # ---- index phase ----
    def build_index(self, corpus: Path):
        docs, skipped = parse_corpus(
            corpus,
            ocr_fn=self.llm.ocr_transcribe if self.llm else None,
        )
        chunks = []
        for d in docs:
            rel = d["path"]
            self.corpus_docs[rel] = d["text"]
            self.file_status[rel] = doc_status(d)
            if d["kind"] == "image":
                self.images[rel] = str(corpus / rel)
            for ci, c in enumerate(chunk_text(d["text"])):
                chunks.append({"id": len(chunks), "file": rel, "text": c})

        emb = None
        if self.embedder and chunks:
            try:
                vecs = []
                bs = 16
                for i in range(0, len(chunks), bs):
                    v = self.embedder.embed([c["text"] for c in chunks[i : i + bs]])
                    if v is not None:
                        vecs.append(v)
                if vecs:
                    emb = self.torch_cat(vecs)
            except Exception:
                emb = None
        self.chunks = chunks
        self.emb = emb.float().cpu().numpy() if emb is not None else None
        save_index(docs, chunks, self.emb, skipped)
        self.meta = {"status": self.file_status}
        return {"files": len(docs), "chunks": len(chunks), "skipped": skipped}

    def torch_cat(self, vecs):
        import torch
        return torch.cat(vecs, dim=0)

    def ensure_index_loaded(self, corpus: Path | None = None):
        if self.chunks:
            return True
        if (INDEX_DIR / "chunks.jsonl").exists():
            chunks, emb, meta = load_index()
            self.chunks = chunks
            self.emb = emb
            self.meta = meta or {}
            self.file_status = self.meta.get("status", {})
            if corpus is not None:
                kinds = self.meta.get("kinds", {})
                for rel, kind in kinds.items():
                    if kind == "image":
                        self.images[rel] = str(corpus / rel)
                for c in chunks:
                    self.corpus_docs.setdefault(c["file"], "")
            return True
        return False

    # ---- query phase ----
    def answer(self, corpus: Path, query: str):
        if not self.chunks:
            self.ensure_index_loaded(corpus)
        if not self.chunks:
            return "", [], 0.0
        if self.llm is None:
            return "", [], 0.0

        q_emb = None
        if self.embedder and self.emb is not None:
            q_emb = self.embedder.embed([query])[0].float().cpu().numpy()
        top = retrieve(self.chunks, self.emb, query, q_emb)
        if not top:
            return "", [], 0.0

        # Second hop: identifiers (ticket ids, part numbers) in the top chunks
        # are often the only route to a linked file that shares no vocabulary
        # with the question (the multi-hop trap). Re-retrieve with them and
        # merge any newly-surfaced files into the context.
        bridge = self._bridge_tokens(top, query)
        if bridge:
            q2 = query + " " + " ".join(bridge)
            q2_emb = None
            if self.embedder and self.emb is not None:
                q2_emb = self.embedder.embed([q2])[0].float().cpu().numpy()
            top2 = retrieve(self.chunks, self.emb, q2, q2_emb, top_k=8)
            seen = {c["id"] for c in top}
            for c in top2:
                if c["id"] not in seen:
                    top.append(c)
                    seen.add(c["id"])
                if len(top) >= 14:
                    break

        blocks = [(c["file"], c["text"], self.file_status.get(c["file"], "normal"))
                  for c in top]
        retrieved_files = list(dict.fromkeys(c["file"] for c in top))
        result = self._generate_answer(query, blocks)

        ans = normalize_answer(result.get("answer", ""))
        cited = self._clean_citations(result.get("citations", []), retrieved_files)
        if not ans:
            return "", [], 0.0

        # If the model kept a unit the corpus value doesn't carry ("94C" when
        # the expected bare value is "94"), strip it only when the bare form
        # is literally present in the evidence.
        evidence = " ".join(c["text"] for c in top)
        stripped = re.sub(
            r"\s*(°[CF]|[CF]|V|A|W|Hz|kHz|MHz|GHz|[KMGT]?B|ms|s|%)$", "", ans
        ).strip()
        if stripped != ans and _token_in_text(stripped, evidence):
            ans = stripped

        # Hallucination guard: a value that appears nowhere in the retrieved
        # evidence cannot be cited, so it must be a refusal. An explicit
        # citation of an image file counts as evidence (its pixels are). If
        # the answer is absent from the text but an image was in context, ask
        # the model whether that image literally shows the value - then cite
        # it. This rescues image-derived answers the model failed to cite.
        if not self._evidence_has_answer(query, ans, evidence, cited):
            imgs = [f for f in retrieved_files if f in self.images]
            picked = self._verify_image_answer(ans, imgs)
            if picked:
                cited = list(dict.fromkeys(cited + [picked]))
            else:
                return "", [], 0.0
        else:
            picked = None

        cited = self._fix_citations(
            query, ans, cited, retrieved_files, verified_img=picked
        )
        # Verified image answers: the image alone yields the value - extending
        # from a stub could only add noise.
        if picked is None:
            cited = self._maybe_extend_citations(
                query, ans, cited, retrieved_files
            )
        conf = 0.85 if cited else 0.4
        return ans, cited, conf

    @staticmethod
    def _bridge_tokens(top_chunks, query):
        """Identifiers in retrieved chunks that the query itself does not
        mention - candidates for a second-hop lookup. Matches shapes like
        ORR-1847, E7731, ORR-FAN-2214-B, REV-C2."""
        q_lower = query.lower()
        out = []
        for c in top_chunks[:6]:
            for tok in re.findall(
                    r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b|\b[A-Z]\d{3,}\b", c["text"]):
                if tok.lower() in q_lower or tok in out:
                    continue
                out.append(tok)
                if len(out) >= 4:
                    return out
        return out

    def _evidence_has_answer(self, query, ans, evidence, cited):
        if _answer_in_text(ans, evidence):
            return True
        # A cited file's full text counts as evidence even if its chunks were
        # not retrieved (rescue path for retrieval misses).
        for c in cited:
            if _answer_in_text(ans, self._chunk_texts_of(c)):
                return True
        toks = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]*", ans)
                if len(t) >= 2]
        if not toks:
            return False
        q_toks = set(re.findall(r"[A-Za-z0-9]+", query.lower()))
        cands = [t for t in toks if t.lower() not in q_toks] or toks
        digit = [t for t in cands if any(ch.isdigit() for ch in t)]
        pool = sorted(digit or cands, key=len, reverse=True)
        if _token_in_text(pool[0], evidence):
            return True
        return any(c in self.images for c in cited)

    def _verify_image_answer(self, ans, images):
        """For an answer missing from text evidence: confirm a retrieved image
        really displays the value, and return its path (for citation)."""
        from PIL import Image
        for f in images:
            try:
                img = Image.open(self.images[f]).convert("RGB")
            except Exception:
                continue
            try:
                text = self.llm.chat(
                    f'Does this image visibly show the text or value "{ans}"? '
                    f"Answer exactly YES or NO.",
                    images=[img], max_new_tokens=8)
            except Exception:
                return f
            if "YES" in text.upper():
                return f
        return None

    def _generate_answer(self, query, blocks):
        prompt = build_prompt(query, blocks)
        images = self._context_images(blocks)
        text = self.llm.chat(prompt, images=images, max_new_tokens=220)
        parsed = extract_json(text)
        if parsed is None:
            print(f"[rag] unparseable gen: {text[:300]!r}", flush=True)
            repair = prompt + "\n\nREMINDER: reply with exactly one JSON object, no other text."
            text = self.llm.chat(repair, images=images, max_new_tokens=220)
            parsed = extract_json(text)
        if os.environ.get("MC3_DEBUG"):
            print(f"[rag] gen raw: {text[:300]!r} -> {parsed!r}", flush=True)
        return parsed or {"answer": "", "citations": []}

    def _context_images(self, blocks, max_images=3):
        imgs = []
        for path, _, _ in blocks:
            if path in self.images and len(imgs) < max_images:
                try:
                    from PIL import Image
                    imgs.append(Image.open(self.images[path]).convert("RGB"))
                except Exception:
                    pass
        return imgs or None

    def _clean_citations(self, citations, retrieved_files):
        if not isinstance(citations, list):
            citations = [citations] if citations else []
        known = set(retrieved_files)
        by_base = {}
        for f in retrieved_files:
            by_base.setdefault(Path(f).name, []).append(f)
        # A citation naming a real corpus file that retrieval missed is kept
        # too (matched by full path, or uniquely by basename).
        corpus = set(self.corpus_docs)
        corp_base = {}
        for f in corpus:
            corp_base.setdefault(Path(f).name, []).append(f)
        out = []
        for c in citations:
            if not isinstance(c, str):
                continue
            c = c.strip().lstrip("/").replace("\\", "/")
            c = re.sub(r"^(app|corpus)/", "", c)
            if c in known:
                out.append(c)
            elif Path(c).name in by_base:
                out.append(by_base[Path(c).name][0])
            elif c in corpus:
                out.append(c)
            elif len(corp_base.get(Path(c).name, [])) == 1:
                out.append(corp_base[Path(c).name][0])
        return list(dict.fromkeys(out))

    def _files_containing(self, query, ans, files):
        """Files whose chunks carry the answer - full string first, then the
        most distinctive token (a docx table may split 'Q3 FY27' across cells)."""
        hit = [f for f in files if _token_in_text(ans, self._chunk_texts_of(f))]
        if hit:
            return hit
        toks = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9.\-]*", ans)
                if len(t) >= 2]
        if not toks:
            return []
        q_toks = set(re.findall(r"[A-Za-z0-9]+", query.lower()))
        cands = [t for t in toks if t.lower() not in q_toks] or toks
        digit = [t for t in cands if any(ch.isdigit() for ch in t)]
        pool = sorted(digit or cands, key=len, reverse=True)
        for t in pool:
            hit = [f for f in files if _token_in_text(t, self._chunk_texts_of(f))]
            if hit:
                return hit
        return []

    def _fix_citations(self, query, ans, cited, retrieved_files,
                       verified_img=None):
        """Necessity pruning + evidence check."""
        if not cited:
            containing = self._files_containing(query, ans, retrieved_files)
            cited = containing[:1] if containing else []
            return cited

        cited_text = " ".join(self._chunk_texts_of(f) for f in cited)
        if not _token_in_text(ans, cited_text):
            for f in self._files_containing(query, ans, cited + retrieved_files):
                if f not in cited:
                    cited.append(f)
                    break

        if len(cited) >= 2:
            kept = []
            for f in cited:
                rest = [g for g in cited if g != f]
                if not rest:
                    kept.append(f)
                    continue
                if self._is_citation_needed(query, ans, f, rest, verified_img):
                    kept.append(f)
            if kept:
                cited = kept
            else:
                containing = [f for f in cited
                              if _token_in_text(ans, self._chunk_texts_of(f))]
                cited = containing[:1] if containing else cited[:1]
        return cited

    def _maybe_extend_citations(self, query, ans, cited, retrieved_files):
        """Completeness check (the reverse of pruning): a multi-hop chain can
        leave a bridge file uncited - ask whether any remaining retrieved file
        is also strictly required, and add it."""
        if not cited or len(cited) >= 2:
            return cited
        others = [f for f in retrieved_files if f not in cited]
        if not others:
            return cited
        evidence = "\n\n".join(
            f"[FILE: {g}]\n{self._chunk_texts_of(g, limit=2500)}" for g in cited
        )
        listing = "\n\n".join(
            f"[FILE: {f}]\n{self._chunk_texts_of(f, limit=400)}"
            for f in others[:8]
        )
        prompt = (
            f"A question was answered using ONLY this evidence:\n{evidence}\n\n"
            f"Question: {query}\nAnswer derived: {ans}\n\n"
            f"Other files that were also retrieved (previews):\n{listing}\n\n"
            f"Is any one of those files ALSO strictly required to derive or "
            f"connect the answer to the question - for example supplying an "
            f"identifier or link that the evidence above lacks? Merely "
            f"discussing the topic does NOT count; neither does a stale or "
            f"superseded copy of the same fact.\n"
            f"Reply with exactly one file path from the list, or NONE."
        )
        try:
            text = self.llm.chat(prompt, max_new_tokens=40)
        except Exception:
            return cited
        for f in others:
            if f in text or Path(f).name in text:
                if self._is_citation_needed(query, ans, f, cited):
                    return cited + [f]
                return cited
        return cited

    def _is_citation_needed(self, query, ans, dropped, rest, verified_img=None):
        """Ask the model whether `dropped` was NEEDED to connect the question
        to the answer, given only the remaining cited files' evidence. This is
        the 'removal makes the answer impossible' test, phrased to protect
        multi-hop chains (a file that only supplies an identifier still counts
        as needed)."""
        parts = []
        for g in rest:
            note = ""
            if g == verified_img:
                note = "  [VERIFIED: this image visibly shows the answer]"
            elif g in self.images:
                note = "  [image file - may contain text/values visually]"
            parts.append(f"[FILE: {g}]{note}\n"
                         f"{self._chunk_texts_of(g, limit=2500)}")
        evidence = "\n\n".join(parts)
        prompt = (
            f"A question was answered using evidence from several files.\n\n"
            f"Question: {query}\nAnswer given: {ans}\n"
            f"Files cited: {rest + [dropped]}\n\n"
            f"Here is the evidence WITHOUT the file {dropped}:\n{evidence}\n\n"
            f"Was the file {dropped} NEEDED to reach this answer - i.e. does it "
            f"supply information (a value, identifier, link or fact) that the "
            f"remaining evidence lacks and without which the answer could not be "
            f"determined or connected to the question?\n"
            f"Reply with exactly one word: NEEDED or UNNEEDED."
        )
        try:
            text = self.llm.chat(prompt, max_new_tokens=12)
        except Exception:
            return True
        # Drop the citation only on an explicit UNNEEDED verdict; anything
        # unclear keeps the file (protects multi-hop chains).
        return "UNNEEDED" not in text.upper()

    def _chunk_texts_of(self, path, limit=None):
        texts = [c["text"] for c in self.chunks if c["file"] == path]
        joined = "\n".join(texts)
        if limit:
            joined = joined[:limit]
        return joined
