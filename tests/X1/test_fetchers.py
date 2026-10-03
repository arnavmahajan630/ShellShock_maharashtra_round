"""Package X1: the fetch helpers and the Codeflaws / IntroClass / Mohler loaders.

No network: sizes and downloads are faked, and the loaders run on tiny trees built in tmp_path.
The last tests look at the real downloads and are skipped where nothing has been fetched.
"""
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from ml.external import codeflaws, common, fetch_all, introclass, itsp, mohler


# ---------------------------------------------------------------- where downloads go

def test_default_target_is_the_main_checkout(monkeypatch):
    monkeypatch.delenv("RELEARN_EXTERNAL_DIR", raising=False)
    monkeypatch.setattr(common, "REPO_ROOT", Path("C:/work/repo"))
    assert common.default_target() == Path("C:/work/repo/ml/data/external")
    monkeypatch.setattr(common, "REPO_ROOT", Path("C:/work/repo/.claude/worktrees/agent-123"))
    assert common.default_target() == Path("C:/work/repo/ml/data/external")      # not inside the worktree
    monkeypatch.setenv("RELEARN_EXTERNAL_DIR", "D:/elsewhere")
    assert common.default_target() == Path("D:/elsewhere")


def test_the_limit_is_three_gigabytes():
    assert common.MAX_BYTES == 3 * 1024 ** 3


def test_external_data_is_gitignored():
    ignore = (common.REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "ml/data/external/" in ignore


# ---------------------------------------------------------------- size check before download

def test_file_over_the_limit_is_skipped_without_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "head_size", lambda url: 600 * 1024 ** 2)
    monkeypatch.setattr(common, "download", lambda *a, **k: pytest.fail("must not download"))
    record = common.fetch_file("big", "http://example.invalid/big.zip", tmp_path, max_bytes=500 * 1024 ** 2)
    assert record["status"] == "skipped" and record["reported_bytes"] == 600 * 1024 ** 2
    assert "600.0 MB" in record["reason"] and "500.0 MB" in record["reason"]
    assert not (tmp_path / "big").exists()
    log = json.loads((tmp_path / "fetch_log.json").read_text(encoding="utf-8"))
    assert [e["status"] for e in log] == ["skipped"] and log[0]["source"] == "big"


def test_repository_over_the_limit_is_skipped(tmp_path, monkeypatch):
    info = {"size_bytes": 4 * 1024 ** 3, "default_branch": "main", "license": None, "pushed_at": None, "sha": "abc", "html_url": ""}
    monkeypatch.setattr(common, "github_repo_info", lambda owner, repo: info)
    monkeypatch.setattr(common, "download", lambda *a, **k: pytest.fail("must not download"))
    record = common.fetch_github_snapshot("huge", "o", "r", tmp_path)
    assert record["status"] == "skipped" and "limit is 3072.0 MB" in record["reason"]


def test_unreachable_source_is_recorded_not_invented(tmp_path, monkeypatch):
    def unreachable(url):
        raise common.FetchError(f"HEAD {url}: no route to host")
    monkeypatch.setattr(common, "head_size", unreachable)
    record = common.fetch_file("gone", "http://example.invalid/x.zip", tmp_path)
    assert record["status"] == "failed" and "no route to host" in record["error"]
    assert not (tmp_path / "gone").exists()
    monkeypatch.setattr(common, "github_repo_info", lambda o, r: (_ for _ in ()).throw(common.FetchError("GitHub API o/r: 404")))
    assert common.fetch_github_snapshot("moved", "o", "r", tmp_path)["status"] == "failed"
    assert [e["status"] for e in json.loads((tmp_path / "fetch_log.json").read_text(encoding="utf-8"))] == ["failed", "failed"]


def test_file_is_fetched_and_unpacked_when_small_enough(tmp_path, monkeypatch):
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("data/a.txt", "hello")
        archive.writestr("data/b.txt", "world")
    payload = bundle.getvalue()

    def fake_download(url, dest, max_bytes=common.MAX_BYTES):
        Path(dest).write_bytes(payload)
        return len(payload)
    monkeypatch.setattr(common, "head_size", lambda url: len(payload))
    monkeypatch.setattr(common, "download", fake_download)
    record = common.fetch_file("small", "http://example.invalid/small.zip", tmp_path)
    assert record["status"] == "fetched" and record["files"] == 2 and record["reported_bytes"] == len(payload)
    assert (tmp_path / "small" / "data" / "a.txt").read_text() == "hello" and not (tmp_path / "small" / "small.zip").exists()
    again = common.fetch_file("small", "http://example.invalid/small.zip", tmp_path)
    assert again["status"] == "present"                                         # not downloaded twice


def test_archive_entries_cannot_escape_the_target(tmp_path):
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as archive:
        archive.writestr("../outside.txt", "x")
    with pytest.raises(common.FetchError):
        common.extract(evil, tmp_path / "into")
    assert not (tmp_path / "outside.txt").exists()


def test_extract_can_leave_out_files_by_suffix(tmp_path):
    """IntroClass: 1.96 GB of its 1.98 GB snapshot are repair-tool .log files, which are not unpacked."""
    bundle = tmp_path / "snap.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("repo/median/abc/000/median.c", "int main(){return 0;}")
        archive.writestr("repo/median/abc/000/gp-bb-01.log", "x" * 1000)
        archive.writestr("repo/README.md", "about")
    assert common.extract(bundle, tmp_path / "out", skip_suffixes=introclass.SKIP) == 2
    assert (tmp_path / "out" / "repo" / "median" / "abc" / "000" / "median.c").exists()
    assert not list((tmp_path / "out").rglob("*.log"))
    assert common.extract(bundle, tmp_path / "all") == 3


def test_describe_lines():
    assert "skipped" in common.describe({"source": "s", "status": "skipped", "reported_bytes": 10, "reason": "too big"})
    assert "unknown" in common.describe({"source": "s", "status": "failed", "error": "boom"})


def test_fetch_all_sizes_only_downloads_nothing(tmp_path, monkeypatch, capsys):
    info = {"size_bytes": 20 * 1024 ** 2, "default_branch": "master", "license": "MIT", "pushed_at": None, "sha": "abc", "html_url": ""}
    monkeypatch.setattr(common, "github_repo_info", lambda owner, repo: info)
    monkeypatch.setattr(common, "head_size", lambda url: 700 * 1024 ** 2 if "codeflaws" in url else 1024 ** 2)
    monkeypatch.setattr(common, "download", lambda *a, **k: pytest.fail("must not download"))
    assert fetch_all.main(["--sizes", "--target", str(tmp_path), "--max-mb", "500"]) == 0
    printed = capsys.readouterr().out
    assert "limit per source: 500.0 MB" in printed
    assert [line.split()[0] for line in printed.splitlines()[1:]] == ["itsp", "codeflaws", "introclass", "mohler"]
    assert "over the limit: will be skipped" in [l for l in printed.splitlines() if l.strip().startswith("codeflaws")][0]
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------- loaders on tiny trees

def test_codeflaws_unpacks_only_sources_and_loads_pairs(tmp_path):
    archive = tmp_path / "codeflaws" / "codeflaws.tar.gz"
    archive.parent.mkdir()
    with tarfile.open(archive, "w:gz") as bundle:
        for name, text in [("codeflaws/34-B-bug-111-222/34-B-111.c", "int main(){return 1;}"),
                           ("codeflaws/34-B-bug-111-222/34-B-222.c", "int main(){return 0;}"),
                           ("codeflaws/34-B-bug-111-222/input-pos1", "1 2"),
                           ("codeflaws/34-B-bug-111-222/heldout-output-pos1", "3"),
                           ("codeflaws/9-A-bug-5-6/9-A-5.c", "int main(){return 2;}"),
                           ("codeflaws/README", "about")]:
            data = text.encode()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    assert codeflaws.unpack(archive, archive.parent) == 3
    assert not (archive.parent / "codeflaws" / "34-B-bug-111-222" / "input-pos1").exists()
    rows = codeflaws.load(archive.parent)
    assert [r["defect"] for r in rows] == ["34-B-bug-111-222"]                  # 9-A has no fixed file: not a pair
    assert rows[0]["contest"] == "34" and rows[0]["problem"] == "B"
    assert Path(rows[0]["buggy"]).name == "34-B-111.c" and Path(rows[0]["fixed"]).name == "34-B-222.c"
    assert codeflaws.unpack(archive, tmp_path / "all", sources_only=False) == 6


def test_introclass_loader(tmp_path):
    for parts in [("median", "aaa111", "000"), ("median", "aaa111", "003"), ("smallest", "bbb222", "001"), ("median", "tests", "x")]:
        folder = tmp_path.joinpath(*parts)
        folder.mkdir(parents=True)
        (folder / f"{parts[0]}.c").write_text("int main(){return 0;}", encoding="utf-8")
    rows = introclass.load(tmp_path)
    assert [(r["assignment"], r["student"], r["version"]) for r in rows] == [
        ("median", "aaa111", "000"), ("median", "aaa111", "003"), ("smallest", "bbb222", "001")]


def test_mohler_loader(tmp_path):
    raw, scores = tmp_path / "data" / "raw", tmp_path / "data" / "scores" / "1.1"
    raw.mkdir(parents=True)
    scores.mkdir(parents=True)
    (raw / "questions").write_text("1.1 What is a prototype?\n1.2 What is a loop?\n", encoding="utf-8")
    (raw / "answers").write_text("1.1 A quick model of the product.\n", encoding="utf-8")
    (raw / "1.1").write_text("1.1 It simulates the product.<br><br>\n1.1 A   small   test program.\n", encoding="utf-8")
    (raw / "all").write_text("ignored\n", encoding="utf-8")
    (scores / "ave").write_text("4.5\n2\n", encoding="utf-8")
    rows = mohler.load(tmp_path)
    assert rows == [
        {"question_id": "1.1", "question": "What is a prototype?", "reference": "A quick model of the product.",
         "answer": "It simulates the product.", "score": 4.5},
        {"question_id": "1.1", "question": "What is a prototype?", "reference": "A quick model of the product.",
         "answer": "A small test program.", "score": 2.0}]


# ---------------------------------------------------------------- the real downloads, where present

def _external(name):
    folder = common.default_target() / name
    if not folder.exists():
        pytest.skip(f"{name} has not been fetched into {common.default_target()}")
    return folder


def test_real_itsp_download():
    pairs = itsp.find_pairs(_external("ITSP"))
    assert len(pairs) == 661                                                    # 03 §3.4: 661 student programs


def test_real_codeflaws_download():
    rows = codeflaws.load(_external("codeflaws"))
    if not rows:
        pytest.skip("codeflaws archive is present but not unpacked")
    assert len(rows) > 3000                                                     # 05 §8: 3,902 pairs


def test_real_introclass_download():
    assert len(introclass.load(_external("IntroClass"))) > 500


def test_real_mohler_download():
    rows = mohler.load(_external("mohler"))
    assert len(rows) > 2000 and all(0 <= r["score"] <= 5 for r in rows)
