"""Shared helpers for the public-data fetchers (package X1).

Rules (05 §8, 06 X1):
  - everything goes under ml/data/external/ (gitignored); it is for evaluation only and is
    never committed;
  - the size of a source is checked BEFORE it is downloaded, and a source larger than the
    limit is skipped; the limit also stops a download whose real size turns out larger;
  - what happened to each source (size, fetched / skipped / failed, why) is appended to
    <target>/fetch_log.json. Nothing is invented when a source cannot be reached.

Only the standard library is used.
"""
import datetime
import json
import os
import re
import shutil
import ssl
import subprocess
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 3 * 1024 ** 3           # skip any source larger than this (3 GB)
TIMEOUT_S = 60
USER_AGENT = "relearn-x1-fetch/1.0 (evaluation data fetcher)"


def default_target():
    """ml/data/external of the main checkout.

    A git worktree under <main>/.claude/worktrees/<name>/ is thrown away when its work is
    merged, so downloads made from a worktree go to the main checkout instead.
    RELEARN_EXTERNAL_DIR overrides everything.
    """
    override = os.environ.get("RELEARN_EXTERNAL_DIR")
    if override:
        return Path(override)
    root = REPO_ROOT
    parts = root.parts
    if ".claude" in parts and "worktrees" in parts[parts.index(".claude"):]:
        root = Path(*parts[:parts.index(".claude")])
    return root / "ml" / "data" / "external"


def mb(n_bytes):
    return "unknown" if n_bytes is None else f"{n_bytes / 1024 ** 2:.1f} MB"


class FetchError(RuntimeError):
    pass


def _request(url, method="GET", headers=None):
    request = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(request, timeout=TIMEOUT_S)


def _cert_problem(exc):
    """Python could not build the certificate chain (some servers do not send their intermediate
    certificate). curl, which uses the operating system's certificate store, is then tried. The
    certificate is still verified; verification is never switched off."""
    return isinstance(getattr(exc, "reason", exc), ssl.SSLCertVerificationError) and shutil.which("curl") is not None


def _curl_head(url):
    done = subprocess.run(["curl", "-sIL", "--max-time", str(TIMEOUT_S), "-A", USER_AGENT, url],
                          capture_output=True, text=True, errors="replace")
    blocks = [b for b in re.split(r"\r?\n\r?\n", done.stdout) if b.strip().startswith("HTTP/")]
    if done.returncode != 0 or not blocks:
        raise FetchError(f"HEAD {url}: curl exit {done.returncode} {done.stderr.strip()[:200]}")
    status = int(blocks[-1].split()[1])
    if not 200 <= status < 300:
        raise FetchError(f"HEAD {url}: HTTP {status}")
    length = re.search(r"^content-length:\s*(\d+)", blocks[-1], re.I | re.M)
    return int(length.group(1)) if length else None


def _curl_download(url, partial, max_bytes):
    done = subprocess.run(["curl", "-sSL", "--fail", "--max-time", "3600", "--max-filesize", str(max_bytes),
                           "-A", USER_AGENT, "-o", str(partial), url], capture_output=True, text=True, errors="replace")
    if done.returncode != 0:
        raise FetchError(f"GET {url}: curl exit {done.returncode} {done.stderr.strip()[:200]}")
    size = partial.stat().st_size
    if size > max_bytes:
        raise FetchError(f"{url}: more than {mb(max_bytes)}, download stopped")
    return size


def head_size(url):
    """Content-Length from an HTTP HEAD, following redirects. None when the server does not say.
    Raises FetchError when the URL cannot be reached."""
    try:
        with _request(url, method="HEAD") as response:
            length = response.headers.get("Content-Length")
            return int(length) if length and length.isdigit() else None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        if _cert_problem(exc):
            return _curl_head(url)
        raise FetchError(f"HEAD {url}: {exc}") from exc


def github_repo_info(owner, repo):
    """{size_bytes, default_branch, license, pushed_at, sha} from the GitHub API.
    `size_bytes` is GitHub's figure for the whole repository (with history), in bytes."""
    try:
        with _request(f"https://api.github.com/repos/{owner}/{repo}",
                      headers={"Accept": "application/vnd.github+json"}) as response:
            info = json.load(response)
        branch = info["default_branch"]
        sha = None
        try:
            with _request(f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}",
                          headers={"Accept": "application/vnd.github+json"}) as response:
                sha = json.load(response).get("sha")
        except (urllib.error.URLError, OSError, ValueError):
            pass
        return {"size_bytes": int(info["size"]) * 1024, "default_branch": branch,
                "license": (info.get("license") or {}).get("spdx_id"), "pushed_at": info.get("pushed_at"),
                "sha": sha, "html_url": info.get("html_url")}
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        raise FetchError(f"GitHub API {owner}/{repo}: {exc}") from exc


def download(url, dest, max_bytes=MAX_BYTES):
    """Stream `url` into the file `dest`. Stops and deletes the file if it grows past max_bytes
    or if the disk would be left with less than 1 GB free. Returns the number of bytes written."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    written = 0
    try:
        with _request(url) as response, open(partial, "wb") as out:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise FetchError(f"{url}: more than {mb(max_bytes)}, download stopped")
                if written % (64 * 1024 ** 2) < len(chunk) and shutil.disk_usage(dest.parent).free < 1024 ** 3:
                    raise FetchError(f"{url}: less than 1 GB of disk left, download stopped")
                out.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        partial.unlink(missing_ok=True)
        if not _cert_problem(exc):
            raise FetchError(f"GET {url}: {exc}") from exc
        try:
            written = _curl_download(url, partial, max_bytes)
        except FetchError:
            partial.unlink(missing_ok=True)
            raise
    except FetchError:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(dest)
    return written


def _safe_members(names, into):
    """Refuse archive entries that would land outside `into`."""
    base = Path(into).resolve()
    for name in names:
        if not str((base / name).resolve()).startswith(str(base)):
            raise FetchError(f"archive entry escapes the target folder: {name}")


def extract(archive, into, skip_suffixes=()):
    """Unpack a .zip, .tar.gz or .tgz into `into`, leaving out files whose name ends with one of
    `skip_suffixes`. Returns the number of files written."""
    archive, into, skip = Path(archive), Path(into), tuple(skip_suffixes)
    into.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            _safe_members(bundle.namelist(), into)
            wanted = [info for info in bundle.infolist() if not (skip and info.filename.endswith(skip))]
            bundle.extractall(into, members=wanted)
            return sum(1 for info in wanted if not info.is_dir())
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            members = bundle.getmembers()
            _safe_members([m.name for m in members], into)
            members = [m for m in members if not (m.issym() or m.islnk()) and not (skip and m.name.endswith(skip))]
            bundle.extractall(into, members=members)
            return sum(1 for m in members if m.isfile())
    raise FetchError(f"{archive.name} is neither a zip nor a tar archive")


def move_folder(src, dst):
    """Rename a folder. On Windows a freshly unpacked folder can be locked for a moment by the
    virus scanner or the indexer, so the rename is retried, then replaced by copy + delete."""
    src, dst = Path(src), Path(dst)
    shutil.rmtree(dst, ignore_errors=True)
    for _ in range(20):
        try:
            src.rename(dst)
            return
        except PermissionError:
            time.sleep(0.5)
    shutil.copytree(src, dst)
    shutil.rmtree(src, ignore_errors=True)


def folder_size(path):
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


def log_fetch(target, record):
    """Append one record to <target>/fetch_log.json and return it."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    record = {"when": datetime.datetime.now().isoformat(timespec="seconds"), **record}
    path = target / "fetch_log.json"
    entries = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    entries.append(record)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return record


def fetch_github_snapshot(name, owner, repo, target, max_bytes=MAX_BYTES, force=False, skip_suffixes=()):
    """Fetch the default branch of a GitHub repository as a zip snapshot (no history, no git
    needed) into <target>/<name>/. Size is checked first through the GitHub API; that figure is
    the packed repository, and the snapshot can be larger, so the download itself is also
    capped at max_bytes. Files ending with one of `skip_suffixes` are not unpacked.

    Returns the log record: {"source", "status": fetched | present | skipped | failed, ...}.
    """
    target = Path(target)
    folder = target / name
    record = {"source": name, "url": f"https://github.com/{owner}/{repo}", "limit_bytes": max_bytes}
    if folder.exists() and any(folder.iterdir()) and not force:
        return log_fetch(target, {**record, "status": "present", "folder": str(folder),
                                  "bytes_on_disk": folder_size(folder)})
    try:
        info = github_repo_info(owner, repo)
    except FetchError as exc:
        return log_fetch(target, {**record, "status": "failed", "error": str(exc)})
    record.update(reported_bytes=info["size_bytes"], commit=info["sha"], license=info["license"],
                  pushed_at=info["pushed_at"])
    if info["size_bytes"] > max_bytes:
        return log_fetch(target, {**record, "status": "skipped",
                                  "reason": f"repository is {mb(info['size_bytes'])}, limit is {mb(max_bytes)}"})
    archive = target / f"{name}.zip"
    ref = info["sha"] or info["default_branch"]
    try:
        downloaded = download(f"https://codeload.github.com/{owner}/{repo}/zip/{ref}", archive, max_bytes)
        scratch = target / f"{name}.unpack"
        shutil.rmtree(scratch, ignore_errors=True)
        files = extract(archive, scratch, skip_suffixes)
        roots = [p for p in scratch.iterdir()]
        move_folder(roots[0] if len(roots) == 1 and roots[0].is_dir() else scratch, folder)
        shutil.rmtree(scratch, ignore_errors=True)
    except (FetchError, OSError, zipfile.BadZipFile) as exc:
        archive.unlink(missing_ok=True)
        return log_fetch(target, {**record, "status": "failed", "error": str(exc)})
    archive.unlink(missing_ok=True)
    return log_fetch(target, {**record, "status": "fetched", "folder": str(folder), "downloaded_bytes": downloaded,
                              "files": files, "bytes_on_disk": folder_size(folder)})


def fetch_file(name, url, target, max_bytes=MAX_BYTES, unpack=True, force=False):
    """Fetch one file (usually an archive) into <target>/<name>/, after an HTTP HEAD size check.
    When the server does not report a size, the download is still capped at max_bytes."""
    target = Path(target)
    folder = target / name
    record = {"source": name, "url": url, "limit_bytes": max_bytes}
    if folder.exists() and any(folder.iterdir()) and not force:
        return log_fetch(target, {**record, "status": "present", "folder": str(folder),
                                  "bytes_on_disk": folder_size(folder)})
    try:
        size = head_size(url)
    except FetchError as exc:
        return log_fetch(target, {**record, "status": "failed", "error": str(exc)})
    record["reported_bytes"] = size
    if size is not None and size > max_bytes:
        return log_fetch(target, {**record, "status": "skipped",
                                  "reason": f"file is {mb(size)}, limit is {mb(max_bytes)}"})
    filename = re.sub(r"[^\w.\-]", "_", url.rstrip("/").rsplit("/", 1)[-1]) or "download"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        downloaded = download(url, folder / filename, max_bytes)
        files = 1
        if unpack and (zipfile.is_zipfile(folder / filename) or tarfile.is_tarfile(folder / filename)):
            files = extract(folder / filename, folder)
            (folder / filename).unlink()
    except (FetchError, OSError, zipfile.BadZipFile, tarfile.TarError) as exc:
        shutil.rmtree(folder, ignore_errors=True)
        return log_fetch(target, {**record, "status": "failed", "error": str(exc)})
    return log_fetch(target, {**record, "status": "fetched", "folder": str(folder), "downloaded_bytes": downloaded,
                              "files": files, "bytes_on_disk": folder_size(folder)})


def describe(record):
    """One line for the console."""
    status = record["status"]
    size = record.get("reported_bytes")
    line = f"{record['source']:<12} {status:<8} reported {mb(size):>10}"
    if status in ("fetched", "present"):
        line += f"   on disk {mb(record.get('bytes_on_disk'))}  ->  {record.get('folder')}"
    elif status == "skipped":
        line += f"   {record.get('reason')}"
    else:
        line += f"   {record.get('error')}"
    return line
