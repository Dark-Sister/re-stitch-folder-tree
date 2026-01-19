#!/usr/bin/env python3
"""
Offline-first data discovery + user correlation tool.

What it does:
- Walks a folder tree
- Extracts content + metadata (txt/log/md/csv/json, pdf, docx)
- Correlates each file to a provided users list (JSON array)
- Writes streaming results to NDJSON + a compact summary JSON
- Uses a SQLite cache to speed up rescans

Notes:
- LLM connector stub is included but commented out (OpenAI-compatible).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import fnmatch
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    def tqdm(it, **_kwargs):  # type: ignore
        return it

try:
    from pypdf import PdfReader  # type: ignore
except Exception:
    PdfReader = None  # type: ignore

try:
    import docx  # python-docx  # type: ignore
except Exception:
    docx = None  # type: ignore


TEXT_EXTS = {".txt", ".log", ".md", ".csv", ".json"}
PDF_EXTS = {".pdf"}
DOCX_EXTS = {".docx"}

EXTRACTOR_VERSION = "v1.1"

DEFAULT_EXCLUDE_DIR_GLOBS = [
    ".git",
    ".svn",
    ".hg",
    "node_modules",
    "dist",
    "build",
    ".next",
    "__pycache__",
    ".pytest_cache",
    ".venv",
    "venv",
    "env",
    ".cache",
    "Caches",
    "Cache",
    "Library/Caches",
    "System Volume Information",
    "$RECYCLE.BIN",
]

DEFAULT_EXCLUDE_PATH_GLOBS = [
    "**/.DS_Store",
    "**/Thumbs.db",
]

EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b")
WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-]{1,63}")


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def combine_confidences(weights: Sequence[float]) -> float:
    """
    Combine evidence weights in [0,1] into a single confidence (also [0,1]).
    Using a probabilistic OR: 1 - Π(1-w_i)
    """
    p_not = 1.0
    for w in weights:
        w = max(0.0, min(1.0, float(w)))
        p_not *= (1.0 - w)
    return 1.0 - p_not


def norm_token(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

def atomic_tokens(s: str) -> List[str]:
    """
    Break a string into matchable atomic tokens (lowercased).
    Used for both user indexing and file candidate extraction.
    """
    if not s:
        return []
    return [t.lower() for t in WORD_RE.findall(s)]


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def is_probably_binary(data: bytes) -> bool:
    if not data:
        return False
    # Heuristic: if NUL byte appears early, treat as binary
    return b"\x00" in data[:4096]


@dataclasses.dataclass(frozen=True)
class UserRecord:
    userid: str
    email: str
    full_name: str
    dob: str
    uid: int
    aliases: List[str]
    other_emails: List[str]
    usernames: List[str]
    home_dirs: List[str]

    @staticmethod
    def from_json(obj: Dict[str, Any]) -> "UserRecord":
        mandatory = ["userid", "email", "full_name", "dob", "uid"]
        missing = [k for k in mandatory if k not in obj]
        if missing:
            raise ValueError(f"User record missing keys: {missing}")
        return UserRecord(
            userid=str(obj["userid"]),
            email=str(obj["email"]),
            full_name=str(obj["full_name"]),
            dob=str(obj["dob"]),
            uid=int(obj["uid"]),
            aliases=[str(x) for x in obj.get("aliases", []) or []],
            other_emails=[str(x) for x in obj.get("other_emails", []) or []],
            usernames=[str(x) for x in obj.get("usernames", []) or []],
            home_dirs=[str(x) for x in obj.get("home_dirs", []) or []],
        )


class UserIndex:
    """
    A lightweight token index: token -> set(userid)
    This avoids scoring all users for every file.
    """

    def __init__(self, users: List[UserRecord]):
        self.users_by_userid: Dict[str, UserRecord] = {u.userid: u for u in users}
        self.token_to_userids: Dict[str, List[str]] = defaultdict(list)
        self.all_user_tokens: set[str] = set()
        self.uid_to_userids: Dict[int, List[str]] = defaultdict(list)

        for u in users:
            self.uid_to_userids[int(u.uid)].append(u.userid)
            for tok in self._tokens_for_user(u):
                t = norm_token(tok)
                if not t:
                    continue
                self.all_user_tokens.add(t)
                self.token_to_userids[t].append(u.userid)

                # Also index atomic tokens so "Alex Brown" matches "alex" / "brown"
                for a in atomic_tokens(tok):
                    at = norm_token(a)
                    if not at:
                        continue
                    self.all_user_tokens.add(at)
                    self.token_to_userids[at].append(u.userid)

            # Special handling for emails (local part + domain)
            for em in [u.email, *u.other_emails]:
                if "@" in em:
                    local, _, domain = em.partition("@")
                    for part in [local, domain]:
                        for a in atomic_tokens(part):
                            self.token_to_userids[norm_token(a)].append(u.userid)

        # Deduplicate lists
        for t, lst in list(self.token_to_userids.items()):
            self.token_to_userids[t] = sorted(set(lst))
        for uid, lst in list(self.uid_to_userids.items()):
            self.uid_to_userids[uid] = sorted(set(lst))

    def _tokens_for_user(self, u: UserRecord) -> Iterable[str]:
        yield u.userid
        yield u.email
        yield u.full_name
        for x in u.aliases:
            yield x
        for x in u.other_emails:
            yield x
        for x in u.usernames:
            yield x
        for x in u.home_dirs:
            yield x

    def candidate_userids_from_tokens(self, tokens: Iterable[str]) -> List[str]:
        cands: set[str] = set()
        for tok in tokens:
            t = norm_token(tok)
            if t in self.token_to_userids:
                cands.update(self.token_to_userids[t])
        return sorted(cands)

    def candidate_userids_from_uid(self, uid: Optional[int]) -> List[str]:
        if uid is None:
            return []
        return list(self.uid_to_userids.get(int(uid), []))


def load_users(users_path: Path) -> List[UserRecord]:
    with users_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("users.json must be a JSON array")
    return [UserRecord.from_json(x) for x in data]


def should_exclude_dirname(name: str, exclude_dir_globs: Sequence[str]) -> bool:
    for g in exclude_dir_globs:
        if fnmatch.fnmatch(name, g):
            return True
    return False


def should_exclude_path(relpath: str, exclude_path_globs: Sequence[str]) -> bool:
    for g in exclude_path_globs:
        if fnmatch.fnmatch(relpath, g):
            return True
    return False


def iter_files(root: Path, exclude_dir_globs: Sequence[str], exclude_path_globs: Sequence[str], follow_symlinks: bool) -> Iterable[Path]:
    root = root.resolve()
    stack = [root]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    try:
                        if entry.is_symlink() and not follow_symlinks:
                            continue
                        if entry.is_dir(follow_symlinks=follow_symlinks):
                            if should_exclude_dirname(entry.name, exclude_dir_globs):
                                continue
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=follow_symlinks):
                            p = Path(entry.path)
                            rel = safe_relpath(p, root)
                            if should_exclude_path(rel, exclude_path_globs):
                                continue
                            yield p
                    except PermissionError:
                        continue
        except PermissionError:
            continue


def file_stat(path: Path, follow_symlinks: bool) -> Optional[os.stat_result]:
    try:
        return path.stat() if follow_symlinks else path.lstat()
    except Exception:
        return None


def read_text_file(path: Path, max_bytes: int) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"kind": "text"}
    try:
        with path.open("rb") as f:
            raw = f.read(max_bytes)
        meta["bytes_read"] = len(raw)
        if is_probably_binary(raw):
            meta["binary"] = True
            return "", meta
        text = raw.decode("utf-8", errors="replace")
        return text, meta
    except Exception as e:
        meta["error"] = repr(e)
        return "", meta


def read_pdf(path: Path, max_text_chars: int) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"kind": "pdf"}
    if PdfReader is None:
        meta["error"] = "pypdf not installed"
        return "", meta
    try:
        reader = PdfReader(str(path))
        # Document info dict
        info = {}
        try:
            if reader.metadata:
                for k, v in reader.metadata.items():
                    info[str(k)] = str(v)
        except Exception:
            pass
        meta["docinfo"] = info
        parts: List[str] = []
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t:
                parts.append(t)
            if sum(len(x) for x in parts) >= max_text_chars:
                break
            if i >= 2000:  # safety
                break
        text = "\n".join(parts)
        if len(text) > max_text_chars:
            text = text[:max_text_chars]
        return text, meta
    except Exception as e:
        meta["error"] = repr(e)
        return "", meta


def read_docx(path: Path, max_text_chars: int) -> Tuple[str, Dict[str, Any]]:
    meta: Dict[str, Any] = {"kind": "docx"}
    if docx is None:
        meta["error"] = "python-docx not installed"
        return "", meta
    try:
        d = docx.Document(str(path))
        core = {}
        try:
            cp = d.core_properties
            # Common useful fields
            core = {
                "author": getattr(cp, "author", None),
                "last_modified_by": getattr(cp, "last_modified_by", None),
                "created": str(getattr(cp, "created", "") or ""),
                "modified": str(getattr(cp, "modified", "") or ""),
                "title": getattr(cp, "title", None),
                "subject": getattr(cp, "subject", None),
                "keywords": getattr(cp, "keywords", None),
                "comments": getattr(cp, "comments", None),
                "category": getattr(cp, "category", None),
            }
        except Exception:
            pass
        meta["core_properties"] = {k: ("" if v is None else str(v)) for k, v in core.items()}

        parts: List[str] = []
        for para in d.paragraphs:
            if para.text:
                parts.append(para.text)
            if sum(len(x) for x in parts) >= max_text_chars:
                break
        text = "\n".join(parts)
        if len(text) > max_text_chars:
            text = text[:max_text_chars]
        return text, meta
    except Exception as e:
        meta["error"] = repr(e)
        return "", meta


def extract_content_and_meta(path: Path, ext: str, max_bytes: int, max_text_chars: int) -> Tuple[str, Dict[str, Any]]:
    ext = ext.lower()
    if ext in TEXT_EXTS:
        return read_text_file(path, max_bytes=max_bytes)
    if ext in PDF_EXTS:
        return read_pdf(path, max_text_chars=max_text_chars)
    if ext in DOCX_EXTS:
        return read_docx(path, max_text_chars=max_text_chars)
    return "", {"kind": "unknown", "skipped": True}


def extract_candidate_tokens(path: Path, relpath: str, text: str, meta: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Returns (tokens, notes_about_tokens_source)
    """
    tokens: List[str] = []
    notes: List[str] = []

    # Path / filename tokens
    tokens.extend(atomic_tokens(path.name))
    tokens.extend(atomic_tokens(relpath))
    notes.append("path+filename atomic tokens")

    # Emails in text
    if text:
        emails = EMAIL_RE.findall(text)
        if emails:
            tokens.extend(emails[:50])
            notes.append(f"text emails ({min(len(emails), 50)} captured)")

    # Words-ish tokens from text (bounded)
    if text:
        words = WORD_RE.findall(text)
        # Keep a bounded, frequency-biased set
        counts = Counter([w.lower() for w in words if 3 <= len(w) <= 64])
        for w, _ in counts.most_common(2000):
            tokens.append(w)
        notes.append("text word tokens (top 2000)")

    # Metadata tokens (pdf/docx)
    if meta.get("kind") == "pdf":
        docinfo = meta.get("docinfo") or {}
        for k in ["Author", "/Author", "author", "Creator", "/Creator", "creator", "Producer", "/Producer"]:
            v = docinfo.get(k)
            if v:
                tokens.append(str(v))
                tokens.extend(atomic_tokens(str(v)))
                notes.append(f"pdf docinfo {k}")
    if meta.get("kind") == "docx":
        cp = meta.get("core_properties") or {}
        for k in ["author", "last_modified_by", "title", "subject", "keywords", "comments", "category"]:
            v = cp.get(k)
            if v:
                tokens.append(str(v))
                tokens.extend(atomic_tokens(str(v)))
                notes.append(f"docx core_properties {k}")

    return tokens, notes


def score_user_for_file(
    user: UserRecord,
    *,
    abs_path: str,
    relpath: str,
    text: str,
    meta: Dict[str, Any],
    st: Optional[os.stat_result],
) -> Tuple[float, List[str], Dict[str, float]]:
    """
    Returns (confidence, notes, evidence_weights_by_reason).
    """
    evid: Dict[str, float] = {}
    notes: List[str] = []

    # Filesystem owner uid
    if st is not None and hasattr(st, "st_uid"):
        try:
            if int(st.st_uid) == int(user.uid):
                evid["fs uid match"] = max(evid.get("fs uid match", 0.0), 0.88)
                notes.append(f"filesystem owner uid {st.st_uid} == user uid {user.uid}")
        except Exception:
            pass

    # Path-based signals
    rp = relpath.lower()
    ap = abs_path.lower()
    if user.userid and user.userid.lower() in rp:
        evid["path contains userid"] = max(evid.get("path contains userid", 0.0), 0.75)
        notes.append(f"path contains userid '{user.userid}'")
    for hd in user.home_dirs:
        hdn = norm_token(hd)
        if hdn and hdn.lower().strip("/") and hdn.lower() in ap:
            evid["path contains home_dir"] = max(evid.get("path contains home_dir", 0.0), 0.80)
            notes.append(f"path contains home dir '{hd}'")
            break

    # Strong: email in text (or snippet)
    if user.email and user.email.lower() in text.lower():
        evid["email in text"] = max(evid.get("email in text", 0.0), 0.97)
        notes.append(f"found email '{user.email}' in extracted text")
    for em in user.other_emails:
        if em and em.lower() in text.lower():
            evid["other_email in text"] = max(evid.get("other_email in text", 0.0), 0.95)
            notes.append(f"found email '{em}' in extracted text")
            break

    # Userid/username in text
    if user.userid and user.userid.lower() in text.lower():
        evid["userid in text"] = max(evid.get("userid in text", 0.0), 0.92)
        notes.append(f"found userid '{user.userid}' in extracted text")
    for un in user.usernames:
        if un and re.search(rf"\b{re.escape(un.lower())}\b", text.lower()):
            evid["username in text"] = max(evid.get("username in text", 0.0), 0.86)
            notes.append(f"found username token '{un}' in extracted text")
            break

    # Names in metadata
    if meta.get("kind") == "pdf":
        docinfo = meta.get("docinfo") or {}
        author = (docinfo.get("Author") or docinfo.get("/Author") or docinfo.get("author") or "")
        creator = (docinfo.get("Creator") or docinfo.get("/Creator") or docinfo.get("creator") or "")
        a = norm_token(str(author))
        c = norm_token(str(creator))
        if user.email.lower() in a or user.email.lower() in c:
            evid["pdf author/creator email"] = max(evid.get("pdf author/creator email", 0.0), 0.97)
            notes.append("pdf metadata author/creator contains user email")
        if norm_token(user.full_name) and (norm_token(user.full_name) in a or norm_token(user.full_name) in c):
            evid["pdf author/creator name"] = max(evid.get("pdf author/creator name", 0.0), 0.82)
            notes.append("pdf metadata author/creator contains user full name")

    if meta.get("kind") == "docx":
        cp = meta.get("core_properties") or {}
        author = norm_token(cp.get("author", ""))
        lmb = norm_token(cp.get("last_modified_by", ""))
        if user.email.lower() in author or user.email.lower() in lmb:
            evid["docx author/lmb email"] = max(evid.get("docx author/lmb email", 0.0), 0.97)
            notes.append("docx core properties contain user email")
        if norm_token(user.full_name) and (norm_token(user.full_name) in author or norm_token(user.full_name) in lmb):
            evid["docx author/lmb name"] = max(evid.get("docx author/lmb name", 0.0), 0.84)
            notes.append("docx core properties contain user full name")

    # Weak: full_name in text/snippet
    if user.full_name and norm_token(user.full_name) in norm_token(text) and len(user.full_name) >= 6:
        evid["full_name in text"] = max(evid.get("full_name in text", 0.0), 0.72)
        notes.append(f"found full name '{user.full_name}' in extracted text")
    for al in user.aliases:
        if al and norm_token(al) in norm_token(text) and len(al) >= 4:
            evid["alias in text"] = max(evid.get("alias in text", 0.0), 0.68)
            notes.append(f"found alias '{al}' in extracted text")
            break

    conf = combine_confidences(list(evid.values()))
    return conf, notes, evid


def build_file_preview(text: str, raw_mode: str, snippet_chars: int) -> str:
    if not text:
        return ""
    if raw_mode == "full":
        return text
    return text[:snippet_chars]


def init_cache(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
          path TEXT PRIMARY KEY,
          mtime_ns INTEGER,
          size INTEGER,
          kind TEXT,
          meta_json TEXT,
          match_text TEXT,
          text_preview TEXT,
          text_hash TEXT,
          config_hash TEXT
        )
        """
    )
    # Backwards-compatible migration if an older cache exists
    cols = {row[1] for row in conn.execute("PRAGMA table_info(files)").fetchall()}
    if "config_hash" not in cols:
        conn.execute("ALTER TABLE files ADD COLUMN config_hash TEXT;")
    if "match_text" not in cols:
        conn.execute("ALTER TABLE files ADD COLUMN match_text TEXT;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def cache_get(conn: sqlite3.Connection, path: str) -> Optional[Dict[str, Any]]:
    cur = conn.execute("SELECT mtime_ns,size,kind,meta_json,match_text,text_preview,text_hash,config_hash FROM files WHERE path=?", (path,))
    row = cur.fetchone()
    if not row:
        return None
    return {
        "mtime_ns": row[0],
        "size": row[1],
        "kind": row[2],
        "meta": json.loads(row[3]) if row[3] else {},
        "match_text": row[4] or "",
        "text_preview": row[5] or "",
        "text_hash": row[6] or "",
        "config_hash": row[7] or "",
    }


def cache_put(
    conn: sqlite3.Connection,
    *,
    path: str,
    mtime_ns: int,
    size: int,
    kind: str,
    meta: Dict[str, Any],
    match_text: str,
    text_preview: str,
    text_hash: str,
    config_hash: str,
) -> None:
    conn.execute(
        """
        INSERT INTO files(path,mtime_ns,size,kind,meta_json,match_text,text_preview,text_hash,config_hash)
        VALUES(?,?,?,?,?,?,?,?,?)
        ON CONFLICT(path) DO UPDATE SET
          mtime_ns=excluded.mtime_ns,
          size=excluded.size,
          kind=excluded.kind,
          meta_json=excluded.meta_json,
          match_text=excluded.match_text,
          text_preview=excluded.text_preview,
          text_hash=excluded.text_hash,
          config_hash=excluded.config_hash
        """,
        (path, mtime_ns, size, kind, json.dumps(meta, ensure_ascii=False), match_text, text_preview, text_hash, config_hash),
    )


def topk_folder_tree(paths: List[str], sizes: List[int], max_levels: int = 3, topk: int = 3) -> List[Dict[str, Any]]:
    """
    Build a compact folder segmentation tree.
    Input paths should be relative paths using '/' separators.
    """
    # Normalize separators
    rels = [p.replace("\\", "/").lstrip("/") for p in paths]

    def build(prefix: str, level: int) -> List[Dict[str, Any]]:
        if level >= max_levels:
            return []
        # Count immediate child folders under prefix
        agg_count: Dict[str, int] = defaultdict(int)
        agg_bytes: Dict[str, int] = defaultdict(int)
        bucket: Dict[str, List[int]] = defaultdict(list)  # folder -> indices

        for i, rp in enumerate(rels):
            if prefix:
                if not rp.startswith(prefix.rstrip("/") + "/"):
                    continue
                rest = rp[len(prefix.rstrip("/")) + 1 :]
            else:
                rest = rp
            parts = rest.split("/")
            if len(parts) < 2:
                continue  # file at this level
            folder = parts[0]
            agg_count[folder] += 1
            agg_bytes[folder] += int(sizes[i]) if i < len(sizes) else 0
            bucket[folder].append(i)

        top = sorted(agg_count.items(), key=lambda kv: (kv[1], agg_bytes[kv[0]]), reverse=True)[:topk]
        out: List[Dict[str, Any]] = []
        for folder, cnt in top:
            child_prefix = f"{prefix.rstrip('/')}/{folder}" if prefix else folder
            out.append(
                {
                    "folder": folder,
                    "count": cnt,
                    "bytes": agg_bytes[folder],
                    "children": build(child_prefix, level + 1),
                }
            )
        return out

    return build("", 0)


def orphan_cluster_key(meta: Dict[str, Any], preview: str, relpath: str) -> Tuple[List[str], str]:
    """
    Produce (signals, cluster_id).
    Cluster on a small set of strong-ish signals so groups are readable.
    """
    signals: List[str] = []

    # Emails found in preview (bounded)
    emails = EMAIL_RE.findall(preview or "")
    for e in sorted(set([norm_token(x) for x in emails]))[:3]:
        signals.append(f"email:{e}")

    if meta.get("kind") == "pdf":
        docinfo = meta.get("docinfo") or {}
        author = docinfo.get("Author") or docinfo.get("/Author") or docinfo.get("author") or ""
        creator = docinfo.get("Creator") or docinfo.get("/Creator") or docinfo.get("creator") or ""
        if author:
            signals.append(f"pdf_author:{norm_token(str(author))[:120]}")
        if creator:
            signals.append(f"pdf_creator:{norm_token(str(creator))[:120]}")

    if meta.get("kind") == "docx":
        cp = meta.get("core_properties") or {}
        author = cp.get("author") or ""
        lmb = cp.get("last_modified_by") or ""
        if author:
            signals.append(f"docx_author:{norm_token(str(author))[:120]}")
        if lmb:
            signals.append(f"docx_lmb:{norm_token(str(lmb))[:120]}")

    # Add a coarse location hint (top folder)
    top_folder = relpath.replace("\\", "/").split("/", 1)[0]
    if top_folder:
        signals.append(f"top:{norm_token(top_folder)[:80]}")

    if not signals:
        signals = ["(no_signals)"]
    key_str = "|".join(signals)
    cluster_id = sha256_text(key_str)[:16]
    return signals, cluster_id


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GenAI-ish offline data discovery + correlation tool (v1).")
    p.add_argument("--root", required=True, help="Root folder to scan")
    p.add_argument("--users", required=True, help="Users JSON array (see samples/users.sample.json)")
    p.add_argument("--outdir", default="outputs", help="Output directory (default: outputs)")

    p.add_argument(
        "--min-confidence",
        required=True,
        type=float,
        help="Required. Minimum confidence for 'guaranteed' linking (must be >= 0.80).",
    )
    p.add_argument("--per-user-cap", type=int, default=200, help="Max files listed per user in summary (0 = no cap).")
    p.add_argument("--raw-mode", choices=["snippets", "full"], default="snippets", help="Store snippets or full extracted text in findings.")
    p.add_argument("--snippet-chars", type=int, default=2000, help="Snippet size when raw-mode=snippets")
    p.add_argument("--match-chars", type=int, default=50_000, help="Chars used for matching (independent of output snippet size)")
    p.add_argument("--max-bytes", type=int, default=2 * 1024 * 1024, help="Max bytes to read from plain text files")
    p.add_argument("--max-text-chars", type=int, default=200_000, help="Max extracted chars from PDF/DOCX")

    p.add_argument("--follow-symlinks", action="store_true", help="Follow symlinks (default: false)")
    p.add_argument("--exclude-dir-glob", action="append", default=[], help="Exclude directories by glob (repeatable)")
    p.add_argument("--exclude-path-glob", action="append", default=[], help="Exclude paths by glob (repeatable, relative to root)")

    return p.parse_args(argv)


# --- LLM connector stub (commented out on purpose) ---
#
# def llm_cluster_labels_openai_compatible(endpoint: str, api_key: str, clusters: List[Dict[str, Any]]) -> None:
#     """
#     Optional future feature: send cluster signals/snippets to an OpenAI-compatible endpoint
#     and receive human-friendly cluster labels.
#     Keep disabled by default for privacy.
#     """
#     raise NotImplementedError


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    if args.min_confidence is None:
        print("ERROR: --min-confidence is required", file=sys.stderr)
        return 2
    if args.min_confidence < 0.80:
        print("ERROR: --min-confidence must be >= 0.80", file=sys.stderr)
        return 2
    if int(args.match_chars) <= 0:
        print("ERROR: --match-chars must be > 0", file=sys.stderr)
        return 2

    root = Path(args.root).expanduser().resolve()
    users_path = Path(args.users).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    users = load_users(users_path)
    uindex = UserIndex(users)

    # Cache invalidation key (when extraction-relevant settings change)
    config_hash = sha256_text(
        json.dumps(
            {
                "extractor_version": EXTRACTOR_VERSION,
                "raw_mode": args.raw_mode,
                "snippet_chars": int(args.snippet_chars),
                "match_chars": int(args.match_chars),
                "max_bytes": int(args.max_bytes),
                "max_text_chars": int(args.max_text_chars),
            },
            sort_keys=True,
        )
    )[:16]

    cache_path = outdir / "cache.sqlite"
    conn = init_cache(cache_path)

    findings_path = outdir / "findings.ndjson"
    summary_path = outdir / "summary.json"

    exclude_dir_globs = DEFAULT_EXCLUDE_DIR_GLOBS + list(args.exclude_dir_glob or [])
    exclude_path_globs = DEFAULT_EXCLUDE_PATH_GLOBS + list(args.exclude_path_glob or [])

    run_started = time.time()

    # Aggregates for summary
    user_stats: Dict[str, Dict[str, Any]] = {}
    orphan_records: List[Dict[str, Any]] = []
    counts = Counter()

    # For segmentation
    orphan_relpaths: List[str] = []
    orphan_sizes: List[int] = []

    # Write findings streaming
    with findings_path.open("w", encoding="utf-8") as fout:
        all_files = list(iter_files(root, exclude_dir_globs, exclude_path_globs, args.follow_symlinks))
        for path in tqdm(all_files, desc="Scanning files"):
            ext = path.suffix.lower()
            relpath = safe_relpath(path, root)
            abs_path = str(path.resolve())
            st = file_stat(path, follow_symlinks=args.follow_symlinks)
            if st is None:
                counts["stat_error"] += 1
                continue

            size = int(getattr(st, "st_size", 0) or 0)
            mtime_ns = int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))

            cached = cache_get(conn, str(path))
            used_cache = False

            if (
                cached
                and cached["mtime_ns"] == mtime_ns
                and cached["size"] == size
                and (cached.get("config_hash") or "") == config_hash
            ):
                meta = cached["meta"]
                match_text = cached.get("match_text") or ""
                preview = cached["text_preview"]
                kind = cached.get("kind") or meta.get("kind") or "unknown"
                used_cache = True
            else:
                text, meta = extract_content_and_meta(
                    path,
                    ext=ext,
                    max_bytes=args.max_bytes,
                    max_text_chars=args.max_text_chars,
                )
                kind = meta.get("kind", "unknown")
                match_text = (text or "")[: int(args.match_chars)]
                preview = build_file_preview(text, raw_mode=args.raw_mode, snippet_chars=args.snippet_chars)
                text_hash = sha256_text(preview) if preview else ""
                cache_put(
                    conn,
                    path=str(path),
                    mtime_ns=mtime_ns,
                    size=size,
                    kind=kind,
                    meta=meta,
                    match_text=match_text,
                    text_preview=preview,
                    text_hash=text_hash,
                    config_hash=config_hash,
                )
                used_cache = False

            # Candidate tokens for narrowing candidates (fast)
            tokens, _token_notes = extract_candidate_tokens(path, relpath, match_text, meta)
            candidate_userids = set(uindex.candidate_userids_from_tokens(tokens))
            # Add UID-derived candidates (helps when text has no obvious tokens but ownership is stable)
            try:
                candidate_userids.update(uindex.candidate_userids_from_uid(getattr(st, "st_uid", None)))
            except Exception:
                pass
            candidate_userids = sorted(candidate_userids)

            # Score candidates; if none, still attempt weak scoring against all users? That would be expensive.
            # We keep it tight by only scoring candidates derived from tokens. For v1, this is acceptable.
            best_userid: Optional[str] = None
            best_conf = 0.0
            best_notes: List[str] = []
            candidates_out: List[Dict[str, Any]] = []

            for uid in candidate_userids[:200]:  # safety bound; token collisions can create big fanouts
                user = uindex.users_by_userid[uid]
                conf, notes, evid = score_user_for_file(
                    user,
                    abs_path=abs_path,
                    relpath=relpath,
                    text=match_text,
                    meta=meta,
                    st=st,
                )
                if conf > 0:
                    candidates_out.append(
                        {
                            "userid": uid,
                            "confidence": round(conf, 4),
                            "evidence": {k: round(v, 3) for k, v in sorted(evid.items(), key=lambda kv: kv[1], reverse=True)},
                        }
                    )
                if conf > best_conf:
                    best_conf = conf
                    best_userid = uid
                    best_notes = notes

            candidates_out = sorted(candidates_out, key=lambda x: x["confidence"], reverse=True)[:5]

            status = "orphan"
            match_obj: Dict[str, Any] = {
                "best_userid": None,
                "confidence": round(best_conf, 4),
                "notes": [],
                "candidates": candidates_out,
            }
            if best_userid and best_conf >= float(args.min_confidence):
                status = "linked"
                match_obj["best_userid"] = best_userid
                match_obj["notes"] = best_notes[:12]

                # Update user stats
                us = user_stats.setdefault(
                    best_userid,
                    {"file_count": 0, "total_bytes": 0, "files": [], "notes_samples": []},
                )
                us["file_count"] += 1
                us["total_bytes"] += size
                if args.per_user_cap == 0 or len(us["files"]) < int(args.per_user_cap):
                    us["files"].append(
                        {
                            "relpath": relpath,
                            "size": size,
                            "kind": meta.get("kind", "unknown"),
                            "confidence": round(best_conf, 4),
                            "notes": best_notes[:8],
                        }
                    )
                if best_notes and len(us["notes_samples"]) < 25:
                    us["notes_samples"].append(best_notes[0])
                counts["linked"] += 1
            else:
                # Orphan record (for clustering + segmentation)
                signals, cluster_id = orphan_cluster_key(meta, match_text, relpath)
                orphan_records.append(
                    {
                        "relpath": relpath,
                        "size": size,
                        "kind": meta.get("kind", "unknown"),
                        "cluster_id": cluster_id,
                        "signals": signals,
                    }
                )
                orphan_relpaths.append(relpath)
                orphan_sizes.append(size)
                counts["orphan"] += 1

            # Emit per-file finding
            finding = {
                "path": str(path),
                "relpath": relpath,
                "size": size,
                "mtime_ns": mtime_ns,
                "ext": ext,
                "used_cache": used_cache,
                "extract": {
                    "kind": meta.get("kind", "unknown"),
                    "meta": meta,
                    "raw": preview,
                },
                "match": match_obj,
                "status": status,
            }
            fout.write(json.dumps(finding, ensure_ascii=False) + "\n")

    conn.commit()
    conn.close()

    # Cluster orphans
    clusters: Dict[str, Dict[str, Any]] = {}
    for rec in orphan_records:
        cid = rec["cluster_id"]
        c = clusters.setdefault(
            cid,
            {
                "cluster_id": cid,
                "signals": rec["signals"],
                "file_count": 0,
                "total_bytes": 0,
                "sample_files": [],
                "relpaths": [],
                "sizes": [],
            },
        )
        c["file_count"] += 1
        c["total_bytes"] += int(rec["size"])
        if len(c["sample_files"]) < 10:
            c["sample_files"].append(
                {"relpath": rec["relpath"], "size": rec["size"], "kind": rec["kind"]}
            )
        c["relpaths"].append(rec["relpath"])
        c["sizes"].append(int(rec["size"]))

    cluster_list = list(clusters.values())
    for c in cluster_list:
        c["folder_segments"] = topk_folder_tree(c["relpaths"], c["sizes"], max_levels=3, topk=3)
        # drop heavy lists
        c.pop("relpaths", None)
        c.pop("sizes", None)

    cluster_list = sorted(cluster_list, key=lambda x: (x["file_count"], x["total_bytes"]), reverse=True)

    summary = {
        "run": {
            "started_utc": dt.datetime.fromtimestamp(run_started, tz=dt.timezone.utc).isoformat(),
            "finished_utc": utc_now_iso(),
            "root": str(root),
            "users": str(users_path),
            "outdir": str(outdir),
            "min_confidence": float(args.min_confidence),
            "per_user_cap": int(args.per_user_cap),
            "raw_mode": args.raw_mode,
            "snippet_chars": int(args.snippet_chars),
            "match_chars": int(args.match_chars),
            "max_bytes": int(args.max_bytes),
            "max_text_chars": int(args.max_text_chars),
            "follow_symlinks": bool(args.follow_symlinks),
            "exclude_dir_globs": exclude_dir_globs,
            "exclude_path_globs": exclude_path_globs,
        },
        "counts": dict(counts),
        "users": user_stats,
        "orphans": {
            "file_count": int(counts.get("orphan", 0)),
            "total_bytes": int(sum(orphan_sizes)),
            "clusters": cluster_list[:500],  # keep summary bounded
            "folder_segments": topk_folder_tree(orphan_relpaths, orphan_sizes, max_levels=3, topk=3),
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Wrote findings: {findings_path}")
    print(f"Wrote summary:  {summary_path}")
    print(f"Cache:          {cache_path}")
    print(f"Counts:         {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
