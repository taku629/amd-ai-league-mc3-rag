#!/usr/bin/env python3
"""MC-3 submission client. Thin wrapper over the persistent server started by
the container CMD (see server.py): sends the request over a unix socket and
writes the output JSON. If the server is not yet listening (e.g. container just
booted or the server died), this client spawns it and waits.

Usage shapes (harness contract):
    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

OUTPUT_DIR = Path(os.environ.get("MC3_OUTPUT_DIR", "/app/output"))
QUERY_RPC_TIMEOUT = float(os.environ.get("MC3_QUERY_RPC_TIMEOUT", "280"))
SOCK_PATH = os.environ.get("MC3_SOCK", "/tmp/mc3-rag.sock")
LOCK_PATH = os.environ.get("MC3_LOCK", "/tmp/mc3-rag.lock")
SERVER = "/app/server.py"


def log(msg):
    print(f"[app] {msg}", flush=True)


def rpc(payload: dict, timeout: float) -> dict:
    """One request over the unix socket; returns the parsed response."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(SOCK_PATH)
        s.sendall((json.dumps(payload) + "\n").encode())
        s.shutdown(socket.SHUT_WR)
        buf = b""
        while True:
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode("utf-8", "replace") or "{}")
    finally:
        s.close()


def server_alive() -> bool:
    try:
        resp = rpc({"cmd": "ping"}, timeout=5)
        return bool(resp.get("ok"))
    except Exception:
        return False


def maybe_spawn_server():
    """If nobody is listening, start server.py under a flock so concurrent
    clients don't double-spawn."""
    with open(LOCK_PATH, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        if server_alive():
            return
        log("server not responding; spawning")
        subprocess.Popen(
            ["python3", SERVER],
            stdout=open("/tmp/mc3-server.log", "ab"),
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        for _ in range(120):
            if server_alive():
                log("server up")
                return
            time.sleep(1)


def write_output(query_id: str, answer: str, citations: list, confidence: float):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / (query_id + "_output.json")
    out.write_text(
        json.dumps(
            {"answer": answer, "citations": list(citations),
             "confidence": confidence},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path)
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--query-id")
    ap.add_argument("--query")
    args = ap.parse_args()

    if args.index is not None:
        if not server_alive():
            maybe_spawn_server()
        try:
            resp = rpc({"cmd": "index", "corpus": str(args.index)}, timeout=580)
        except Exception as e:
            log(f"index rpc failed: {e}")
            return 1
        if not resp.get("ok"):
            log(f"index failed: {resp.get('error')}")
            return 1
        log(f"index done: {json.dumps(resp.get('stats', {}))[:800]}")
        return 0

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    # Guarantee a well-formed output file exists even if we crash mid-answer:
    # write a refusal first, then overwrite with the real answer.
    write_output(args.query_id, "", [], 0.0)

    answer, citations, confidence = "", [], 0.0
    if not server_alive():
        maybe_spawn_server()
    try:
        resp = rpc(
            {"cmd": "query", "corpus": str(args.corpus), "query": args.query,
             "query_id": args.query_id},
            timeout=QUERY_RPC_TIMEOUT,
        )
        if resp.get("ok"):
            answer = resp.get("answer", "") or ""
            citations = resp.get("citations", []) or []
            confidence = float(resp.get("confidence", 0.0) or 0.0)
        else:
            log(f"query failed server-side: {resp.get('error')}")
    except Exception as e:
        log(f"query rpc failed: {e}")

    write_output(args.query_id, answer, citations, confidence)
    log(f"wrote {args.query_id}: answer={answer!r} citations={citations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
