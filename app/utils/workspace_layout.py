import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import mimetypes


MANIFEST_VERSION = "1.0"

LAYOUT: Dict[str, object] = {
    "inputs": {
        "documents": "inputs/documents",
        "data": "inputs/data",
        "files": "inputs/files",
        "images": "inputs/images",
    },
    "outputs": "outputs",
    "scratch": "scratch",
    "manifest": "manifest.json",
}

_DOCUMENT_EXTS = {".pdf", ".doc", ".docx", ".md", ".txt", ".rtf", ".html", ".htm"}
_DATA_EXTS = {".csv", ".tsv", ".json", ".parquet", ".xlsx", ".xls", ".sql"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}


def ensure_workspace_layout(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    inputs = LAYOUT["inputs"]
    for _, rel in inputs.items():
        (root / rel).mkdir(parents=True, exist_ok=True)
    (root / str(LAYOUT["outputs"])).mkdir(parents=True, exist_ok=True)
    (root / str(LAYOUT["scratch"])).mkdir(parents=True, exist_ok=True)
    manifest_path = root / str(LAYOUT["manifest"])
    if not manifest_path.exists():
        manifest = {
            "version": MANIFEST_VERSION,
            "created_at": time.time(),
            "updated_at": time.time(),
            "layout": LAYOUT,
            "files": [],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))


def _guess_extension(mime_type: str) -> str:
    ext = mimetypes.guess_extension(mime_type) or ""
    if ext == ".jpe":
        ext = ".jpg"
    return ext


def classify_upload(filename: str, mime_type: str) -> str:
    ext = Path(filename).suffix.lower()
    if mime_type.startswith("image/") or ext in _IMAGE_EXTS:
        return "images"
    if mime_type.startswith("text/") or ext in _DOCUMENT_EXTS:
        return "documents"
    if (
        mime_type in {"application/json", "text/csv", "application/vnd.ms-excel"}
        or mime_type.startswith("application/vnd")
        or ext in _DATA_EXTS
    ):
        return "data"
    return "files"


def sanitize_filename(name: str, fallback: str) -> str:
    base = os.path.basename(name).strip()
    return base or fallback


def pick_upload_path(root: Path, filename: str, mime_type: str) -> Tuple[Path, str]:
    category = classify_upload(filename, mime_type)
    inputs = LAYOUT["inputs"]
    target_dir = root / inputs[category]
    target_dir.mkdir(parents=True, exist_ok=True)

    base_name = sanitize_filename(filename, f"upload_{int(time.time())}")
    ext = Path(base_name).suffix
    if not ext:
        ext = _guess_extension(mime_type)
        base_name = f"{base_name}{ext}"

    candidate = target_dir / base_name
    if not candidate.exists():
        return candidate, category

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        alt = target_dir / f"{stem}_{counter}{suffix}"
        if not alt.exists():
            return alt, category
        counter += 1


def load_manifest(root: Path) -> Dict:
    manifest_path = root / str(LAYOUT["manifest"])
    if not manifest_path.exists():
        ensure_workspace_layout(root)
    try:
        return json.loads(manifest_path.read_text())
    except Exception:
        return {
            "version": MANIFEST_VERSION,
            "created_at": time.time(),
            "updated_at": time.time(),
            "layout": LAYOUT,
            "files": [],
        }


def update_manifest(root: Path, entry: Dict) -> Dict:
    manifest = load_manifest(root)
    manifest.setdefault("files", [])
    manifest["files"].append(entry)
    manifest["updated_at"] = time.time()
    manifest_path = root / str(LAYOUT["manifest"])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def relative_to_root(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
