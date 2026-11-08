"""Unit test for ``2_combine_corpora.py`` (no data, no GPU, no torch).

Builds fake preprocessed corpora in a temp dir -- the combiner only ever looks
at the *paths* of the tensors, so empty files are enough -- then drives the CLI
and checks:

* base speaker ids are preserved and new speakers get fresh ids above them,
* ``val_stage2.txt`` stays the base val unless ``--include_extra_val``,
* ``--cap_multiplier`` keeps the requested number of clips, per speaker,
* ``--sid_start`` / ``n_speakers`` bounds, duplicate clips, missing tensors and
  an ``--output_dir`` pointing at an input are all rejected,
* ``--dry_run`` writes nothing, and two runs are byte-identical,
* a ``*_speakers.json`` sidecar lands in ``speaker_manifest.json`` (and no
  ``voice_map.json`` is written: extra voices are conditioning only).

Usage: ``python test_combine_corpora.py`` (exits non-zero on failure).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "2_combine_corpora.py")
KEYS = ("phone_ids", "spec_codes", "pitch_codes")
TRAIN = "train_stage2.txt"
VAL = "val_stage2.txt"


def check(cond, msg):
    if not cond:
        raise SystemExit(f"FAIL: {msg}")


def make_corpus(root, train, val, skip=(), sidecar=None):
    """Write a fake preprocessed corpus: empty tensors + both filelists.

    Args:
        root: directory to create.
        train: ``{raw_sid: [clip_name, ...]}`` for train_stage2.txt.
        val: same, for val_stage2.txt.
        skip: ``(clip, key)`` pairs whose tensor file is *not* created.
        sidecar: optional dict written as ``<root>/fake_speakers.json``.
    """
    os.makedirs(root, exist_ok=True)

    def _write(spec, name):
        lines = []
        for sid, clips in spec.items():
            for clip in clips:
                paths = []
                for key in KEYS:
                    path = os.path.join(root, f"{clip}.{key}")
                    if (clip, key) not in skip:
                        with open(path, "w", encoding="utf-8"):
                            pass
                    paths.append(path)
                lines.append("|".join(paths + [str(sid)]))
        with open(os.path.join(root, name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return lines

    train_lines = _write(train, TRAIN)
    _write(val, VAL)
    if sidecar is not None:
        with open(os.path.join(root, "fake_speakers.json"), "w",
                  encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)
    return train_lines


def run(*args):
    return subprocess.run([sys.executable, SCRIPT, *[str(a) for a in args]],
                          capture_output=True, text=True)


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def sids(path):
    return [int(ln.split("|")[3]) for ln in read_lines(path)]


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_pair(tmp):
    """A base corpus (sids 0 and 5) and an extra corpus (raw sids 0 and 1)."""
    base = os.path.join(tmp, "base")
    extra = os.path.join(tmp, "extra")
    base_train = make_corpus(
        base, {0: ["b0a", "b0b"], 5: ["b5a"]}, {0: ["b0v"]})
    make_corpus(
        extra, {0: ["e0a", "e0b"], 1: ["e1a"]}, {1: ["e1v"]},
        sidecar={"corpus": "fake", "n_speakers": 2, "speakers": {
            "0": {"speaker": "p225", "gender": "female", "lines": 2},
            "1": {"speaker": "p226", "gender": "male", "lines": 1}}})
    return base, extra, base_train


def test_append(tmp):
    base, extra, base_train = build_pair(tmp)
    out = os.path.join(tmp, "out")
    r = run("--base", base, "--extra", extra, "--output_dir", out)
    check(r.returncode == 0, f"exit {r.returncode}\n{r.stdout}\n{r.stderr}")

    train = read_lines(os.path.join(out, TRAIN))
    check(len(train) == 6,
          f"expected 3 base + 3 extra train clips, got {len(train)}")
    got = sorted(set(sids(os.path.join(out, TRAIN))))
    check(got == [0, 5, 6, 7],
          f"new speakers must land above the base max (5): {got}")
    check(train[:3] == base_train, "base train lines must be preserved verbatim")
    check(sids(os.path.join(out, VAL)) == [0],
          "val must stay the base val by default")

    man = read_json(os.path.join(out, "speaker_manifest.json"))
    check(man["n_speakers_required"] == 8,
          f"n_speakers_required {man['n_speakers_required']}")
    check(man["extras"][0]["sid_map"] == {"0": 6, "1": 7},
          f"sid_map {man['extras'][0]['sid_map']}")
    check(man["extras"][0]["used_sidecar"] is True, "sidecar not detected")
    spk = man["extras"][0]["speakers"]
    check(spk["0"]["new_sid"] == 6 and spk["0"]["speaker"] == "p225"
          and spk["0"]["gender"] == "female",
          f"manifest speaker annotation: {spk['0']}")
    check(spk["1"]["speaker"] == "p226" and spk["1"]["kept_lines"] == 1,
          f"manifest speaker annotation: {spk['1']}")
    check(not os.path.exists(os.path.join(out, "voice_map.json")),
          "no voice_map.json should be written any more")
    print("  append + id allocation + manifest: OK")


def test_include_extra_val(tmp):
    base, extra, _ = build_pair(tmp)
    out = os.path.join(tmp, "out")
    r = run("--base", base, "--extra", extra, "--output_dir", out,
            "--include_extra_val")
    check(r.returncode == 0, f"exit {r.returncode}\n{r.stdout}\n{r.stderr}")
    got = sids(os.path.join(out, VAL))
    check(got == [0, 7],
          f"--include_extra_val must append the remapped extra val: {got}")
    print("  --include_extra_val: OK")


def test_cap(tmp):
    base = os.path.join(tmp, "cap", "base")
    extra = os.path.join(tmp, "cap", "extra")
    make_corpus(base, {0: ["b0a", "b0b", "b0c", "b0d"]}, {0: ["b0v"]})
    make_corpus(extra, {0: [f"e0_{i}" for i in range(10)],
                        1: [f"e1_{i}" for i in range(10)]}, {0: ["e0v"]})
    out = os.path.join(tmp, "cap", "out")
    r = run("--base", base, "--extra", extra, "--output_dir", out,
            "--cap_multiplier", "1.0")
    check(r.returncode == 0, f"exit {r.returncode}\n{r.stdout}\n{r.stderr}")
    man = read_json(os.path.join(out, "speaker_manifest.json"))
    kept = man["extras"][0]["n_train_kept"]
    check(kept == 4,
          f"--cap_multiplier 1.0 on a 4-clip base must keep 4, got {kept}")
    check(man["extras"][0]["capped"] is True, "capped flag not set")
    check(len(read_lines(os.path.join(out, TRAIN))) == 8, "combined train size")
    per = {int(k): v["kept_lines"]
           for k, v in man["extras"][0]["speakers"].items()}
    check(per == {0: 2, 1: 2},
          f"cap must stay proportional per speaker: {per}")
    print("  --cap_multiplier: OK")



def test_total_cap(tmp):
    """--cap_total_multiplier budgets the sum of the extras, not each corpus."""
    base = os.path.join(tmp, "tcap", "base")
    a = os.path.join(tmp, "tcap", "a")
    b = os.path.join(tmp, "tcap", "b")
    make_corpus(base, {0: ["b0a", "b0b", "b0c", "b0d"]}, {0: ["b0v"]})
    make_corpus(a, {0: [f"a_{i}" for i in range(10)]}, {0: ["av"]})
    make_corpus(b, {0: [f"b_{i}" for i in range(10)]}, {0: ["bv"]})
    out = os.path.join(tmp, "tcap", "out")
    r = run("--base", base, "--extra", a, "--extra", b, "--output_dir", out,
            "--cap_total_multiplier", "1.0")
    check(r.returncode == 0, f"exit {r.returncode}\n{r.stdout}\n{r.stderr}")
    man = read_json(os.path.join(out, "speaker_manifest.json"))
    kept = [e["n_train_kept"] for e in man["extras"]]
    check(kept == [2, 2],
          f"a 4-clip budget split over two equal corpora must be [2, 2]: {kept}")
    check(len(read_lines(os.path.join(out, TRAIN))) == 8,
          "combined train size after the total cap")
    check(man["cap_total_multiplier"] == 1.0, "manifest must record the knob")
    print("  --cap_total_multiplier: OK")


def test_errors(tmp):
    base, extra, _ = build_pair(tmp)

    r = run("--base", base, "--extra", extra, "--output_dir",
            os.path.join(tmp, "o2"), "--sid_start", "3")
    check(r.returncode != 0 and "collides" in r.stdout + r.stderr,
          f"--sid_start below the base max must be rejected: {r.stdout}{r.stderr}")

    r = run("--base", base, "--extra", extra, "--output_dir",
            os.path.join(tmp, "o3"), "--n_speakers", "7")
    check(r.returncode != 0 and "n_speakers" in r.stdout + r.stderr,
          f"ids beyond --n_speakers must be rejected: {r.stdout}{r.stderr}")

    r = run("--base", base, "--extra", base, "--output_dir",
            os.path.join(tmp, "o4"))
    check(r.returncode != 0 and "more than one corpus" in r.stdout + r.stderr,
          f"the same clips in two corpora must be rejected: {r.stdout}{r.stderr}")

    r = run("--base", base, "--extra", extra, "--output_dir", base)
    check(r.returncode != 0
          and "must not be one of the inputs" in r.stdout + r.stderr,
          f"--output_dir == input must be rejected: {r.stdout}{r.stderr}")

    broken = os.path.join(tmp, "broken")
    make_corpus(broken, {0: ["x0", "x1"]}, {0: ["x0v"]},
                skip={("x0", "pitch_codes")})
    r = run("--base", base, "--extra", broken, "--output_dir",
            os.path.join(tmp, "o5"))
    check(r.returncode != 0 and "do not exist" in r.stdout + r.stderr,
          f"missing tensors must be rejected: {r.stdout}{r.stderr}")
    check(not os.path.exists(os.path.join(tmp, "o5", TRAIN)),
          "a failed run must not leave a filelist behind")
    print("  error paths: OK")


def test_dry_run_and_determinism(tmp):
    base, extra, _ = build_pair(tmp)
    dry = os.path.join(tmp, "dry")
    r = run("--base", base, "--extra", extra, "--output_dir", dry, "--dry_run")
    check(r.returncode == 0, f"dry run failed: {r.stdout}{r.stderr}")
    check(not os.path.exists(dry), "--dry_run must not create the output dir")
    check("n_speakers must be >= 8" in r.stdout,
          f"dry run must report the required table size: {r.stdout}")

    a = os.path.join(tmp, "a")
    b = os.path.join(tmp, "b")
    for out in (a, b):
        r = run("--base", base, "--extra", extra, "--output_dir", out)
        check(r.returncode == 0, f"run failed: {r.stdout}{r.stderr}")
    check(read_lines(os.path.join(a, TRAIN)) == read_lines(os.path.join(b, TRAIN)),
          "two runs over the same inputs must be identical")
    print("  --dry_run + determinism: OK")


def main():
    check(os.path.exists(SCRIPT), f"missing {SCRIPT}")
    tmp = tempfile.mkdtemp(prefix="combine_test_")
    try:
        test_append(tmp)
        test_include_extra_val(tmp)
        test_cap(tmp)
        test_total_cap(tmp)
        test_errors(tmp)
        test_dry_run_and_determinism(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL COMBINE CHECKS PASSED")


if __name__ == "__main__":
    main()

