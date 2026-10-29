"""Download raw datasets into SCRIPTSENSE_DATA_DIR/raw and verify published hashes.

Every source below has a publisher-provided checksum; a download that does not
match is deleted and the script exits non-zero. A provenance record (URL,
license note, size, hash, timestamp) is written to data/manifests/downloads.json.

Usage:
    python ml/scripts/download_datasets.py txd22 semeval raid_train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scriptsense_ml.paths import MANIFEST_DIR, RAW_DIR, ensure_dirs  # noqa: E402

SOURCES = {
    "txd22": {
        "url": "https://data.mendeley.com/public-files/datasets/prcjcggtjf/files/"
               "1e842f9d-4ff8-4986-bc7c-cde460f806ee/file_downloaded",
        "dest": "txd22/TXD-22 A Large-Scale Benchmark Dataset.csv",
        "algo": "sha256",
        "hash": "232c260f8caf7ea61a331f39fa730956862afbe0c5fc554720daa07d215a2f7d",
        "size": 121656267,
        "landing": "https://data.mendeley.com/datasets/prcjcggtjf/1",
        "license": "CC BY 4.0 (per Mendeley Data record; attribution required)",
        "version": "Mendeley Data v1, DOI 10.17632/prcjcggtjf.1, published 2026-07-01",
    },
    "semeval": {
        "url": "https://zenodo.org/records/11002158/files/SemEval2024-Task8-data.zip?download=1",
        "dest": "semeval2024_task8/SemEval2024-Task8-data.zip",
        "algo": "md5",
        "hash": "a126d3369b55931ac43da585aededef6",
        "size": 419460633,
        "landing": "https://zenodo.org/records/11002158",
        "license": "UNSTATED in Zenodo record metadata; user confirmed research use on 2026-10-25. "
                   "Do not redistribute until terms are confirmed.",
        "version": "Zenodo record 11002158",
    },
    "raid_train": {
        "url": "https://huggingface.co/datasets/liamdugan/raid/resolve/main/train.csv",
        "dest": "raid/train.csv",
        "algo": "sha256",
        "hash": "52f04ceebc126064e68fbd22d8b736964065745464f4bfd52e488150b49f84e4",
        "size": 11779491051,
        "landing": "https://huggingface.co/datasets/liamdugan/raid",
        "license": "MIT (per Hugging Face dataset card); generations derive from third-party "
                   "source texts - confirm terms before redistribution",
        "version": "HF liamdugan/raid main, LFS oid above",
    },
}


def _hash_file(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(name: str) -> dict:
    spec = SOURCES[name]
    dest = RAW_DIR / spec["dest"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size == spec["size"]:
        print(f"[{name}] present, verifying {spec['algo']}...", flush=True)
    else:
        part = dest.with_suffix(dest.suffix + ".part")
        start = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(spec["url"], headers={"User-Agent": "scriptsense-data/1.0"})
        if start:
            req.add_header("Range", f"bytes={start}-")
        print(f"[{name}] downloading {spec['size'] / 1e6:.1f} MB (resume at {start})", flush=True)
        with urllib.request.urlopen(req, timeout=60) as resp:
            if start and resp.status != 206:
                start = 0  # server ignored Range; restart
            with open(part, "ab" if start else "wb") as out:
                done, last = start, time.time()
                while block := resp.read(4 * 1024 * 1024):
                    out.write(block)
                    done += len(block)
                    if time.time() - last > 30:
                        print(f"[{name}] {done / spec['size']:.1%}", flush=True)
                        last = time.time()
        part.replace(dest)

    digest = _hash_file(dest, spec["algo"])
    if digest != spec["hash"]:
        dest.unlink()
        raise SystemExit(f"[{name}] HASH MISMATCH ({digest} != {spec['hash']}); file deleted")
    print(f"[{name}] OK {dest}", flush=True)
    return {
        "name": name,
        "url": spec["url"],
        "landing_page": spec["landing"],
        "version": spec["version"],
        "license": spec["license"],
        "local_path": str(dest.relative_to(RAW_DIR)),
        "size_bytes": dest.stat().st_size,
        "hash_algo": spec["algo"],
        "hash": digest,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+", choices=sorted(SOURCES))
    args = ap.parse_args()
    ensure_dirs()
    manifest_path = MANIFEST_DIR / "downloads.json"
    for name in args.names:
        record = download(name)
        # Re-read right before writing so concurrent downloads don't clobber each other.
        records = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        records[name] = record
        manifest_path.write_text(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
