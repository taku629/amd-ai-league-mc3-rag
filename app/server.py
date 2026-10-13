#!/usr/bin/env python3
"""Persistent server for MC-3.

Started by the container CMD. Loads the vision-language model and the embedding
model once, keeps them resident, and serves index/query requests from the thin
app.py client over a unix socket. Protocol: one JSON line in, one JSON line out.
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/app")
import rag

SOCK_PATH = os.environ.get("MC3_SOCK", "/tmp/mc3-rag.sock")
LOCK_PATH = os.environ.get("MC3_LOCK", "/tmp/mc3-rag.lock")
LLM_DIR = os.environ.get("MC3_LLM_PATH", "/models/llm")
EMBED_DIR = os.environ.get("MC3_EMBED_PATH", "/models/embed")

READY = threading.Event()
WORK = threading.Lock()
ENGINE = None
DEVICE = "cpu"


def log(msg):
    print(f"[server {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class VLM:
    """Vision-language model wrapper: chat() for answers, ocr_transcribe() for
    index-time image transcription."""

    def __init__(self, model_dir: str, device: str):
        import torch
        from transformers import AutoProcessor

        self.device = device
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(model_dir)
        model = None
        errs = []
        for cls_name in ("AutoModelForImageTextToText", "AutoModelForVision2Seq",
                         "AutoModelForCausalLM"):
            try:
                cls = getattr(__import__("transformers"), cls_name)
                model = cls.from_pretrained(
                    model_dir, torch_dtype=dtype, attn_implementation="sdpa",
                )
                log(f"model loaded via {cls_name}")
                break
            except Exception as e:
                errs.append(f"{cls_name}: {type(e).__name__} {e}")
        if model is None:
            raise RuntimeError("no auto class could load model: " + " | ".join(errs))
        self.model = model.to(device).eval()
        self.torch = torch

    def chat(self, prompt: str, images=None, max_new_tokens=200):
        max_new_tokens = min(
            max_new_tokens,
            int(os.environ.get("MC3_MAX_NEW_TOKENS", max_new_tokens)),
        )
        torch = self.torch
        msgs = [{"role": "user", "content": []}]
        if images:
            for _ in images:
                msgs[0]["content"].append({"type": "image"})
        msgs[0]["content"].append({"type": "text", "text": prompt})
        try:
            text = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = (("<image>\n" * (len(images or []))) + prompt
                    + "\nASSISTANT:")
        inputs = self.processor(
            text=[text], images=images or None, return_tensors="pt"
        ).to(self.device)
        stops = None
        if os.environ.get("MC3_JSON_STOP", "1") == "1":
            stops = StoppingCriteriaList(
                [_JsonStop(self.processor, inputs["input_ids"].shape[1])]
            )
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                stopping_criteria=stops,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            gen.unsqueeze(0), skip_special_tokens=True
        )[0].strip()

    def ocr_transcribe(self, path: Path):
        from PIL import Image
        img = Image.open(path).convert("RGB")
        prompt = (
            "Transcribe ALL text visible in this image verbatim, preserving "
            "line structure. Include every label, code, pin name, number, "
            "table cell and heading exactly as written. If the image is a "
            "diagram, also list each element and its label."
        )
        return self.chat(prompt, images=[img], max_new_tokens=512)


try:
    from transformers import StoppingCriteria, StoppingCriteriaList
except Exception:  # pragma: no cover - very old transformers
    StoppingCriteria, StoppingCriteriaList = object, list


class _JsonStop(StoppingCriteria):
    """Stop as soon as the generated suffix holds one complete JSON object.
    Cuts latency and prevents the model from rambling past the answer."""

    def __init__(self, processor, prompt_len):
        self.tok = getattr(processor, "tokenizer", processor)
        self.n = prompt_len

    def __call__(self, input_ids, scores, **kw):
        tail = self.tok.decode(input_ids[0][self.n:], skip_special_tokens=True)
        depth = 0
        seen = False
        for ch in tail:
            if ch == "{":
                depth += 1
                seen = True
            elif ch == "}":
                depth -= 1
            if seen and depth == 0:
                return True
        return False


def load_models():
    global ENGINE, DEVICE
    DEVICE = pick_device()
    log(f"device={DEVICE}; loading models")
    embedder = None
    try:
        embedder = rag.Embedder(EMBED_DIR, DEVICE)
        log("embedder ready")
    except Exception as e:
        log(f"embedder failed, BM25-only retrieval: {e}")
    llm = None
    try:
        llm = VLM(LLM_DIR, DEVICE)
        log("llm ready")
    except Exception as e:
        log(f"llm failed to load: {e}")
        traceback.print_exc()
    ENGINE = rag.RagEngine(llm, embedder, DEVICE)
    READY.set()
    log("models ready")


def handle(req: dict):
    cmd = req.get("cmd")
    if cmd == "ping":
        return {"ok": True, "ready": READY.is_set()}
    READY.wait(timeout=560)
    if ENGINE is None:
        return {"ok": False, "error": "engine not initialized"}
    if cmd == "index":
        with WORK:
            stats = ENGINE.build_index(Path(req["corpus"]))
        log(f"index done: {stats['files']} files, {stats['chunks']} chunks, "
            f"{len(stats['skipped'])} skipped")
        return {"ok": True, "stats": stats}
    if cmd == "query":
        t0 = time.time()
        with WORK:
            ans, cites, conf = ENGINE.answer(Path(req["corpus"]), req["query"])
        log(f"query {req.get('query_id', '?')}: {time.time()-t0:.1f}s "
            f"ans={ans!r} cites={cites}")
        return {"ok": True, "answer": ans, "citations": cites, "confidence": conf}
    return {"ok": False, "error": f"unknown cmd {cmd!r}"}


def conn_thread(conn):
    try:
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(1 << 20)
            if not chunk:
                conn.close()
                return
            buf += chunk
        req = json.loads(buf.decode("utf-8", "replace"))
        resp = handle(req)
    except Exception as e:
        traceback.print_exc()
        resp = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    try:
        conn.sendall((json.dumps(resp, ensure_ascii=False) + "\n").encode())
    except Exception:
        pass
    finally:
        conn.close()


def main():
    # Single-instance guard: a second server (CMD respawn racing a client
    # spawn) must not unlink/rebind the live server's socket.
    lock_fd = open(LOCK_PATH, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("another server holds the lock; exiting")
        return
    try:
        os.unlink(SOCK_PATH)
    except FileNotFoundError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK_PATH)
    os.chmod(SOCK_PATH, 0o777)
    srv.listen(16)
    log(f"listening on {SOCK_PATH}")

    threading.Thread(target=load_models, daemon=True).start()

    while True:
        conn, _ = srv.accept()
        threading.Thread(target=conn_thread, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
