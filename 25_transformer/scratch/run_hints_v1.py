#!/usr/bin/env python3
"""One command for every HINTS_V1_PLAN.md test / ablation (Phases 1-4).

The plan is a measurement programme with a dozen near-identical invocations
that differ only in checkpoint list, hint arm, ``--ignore_eos`` and one or two
config keys.  Getting one of those flags wrong silently produces a number that
is not comparable to the others, which is exactly the failure this file exists
to prevent: the roster, the arms and the per-arm config overrides are declared
**once**, here, and every run is logged to a dated directory with the argv that
produced it.

Suites (``SUITE`` positional):

``phase1``   read-only measurement on the existing runs (no GPU training):
             the WER matrix (5 checkpoints x {null, cond} hints x
             {EOS, --ignore_eos}), same-batch checkpoint comparison,
             repetition probe, per-channel liveness, code-space probe.
``phase2``   the four 2 k-step injection-site wiring arms
             (``hint_film_memory`` x ``hint_decoder_proj``) + their gates.
``phase3``   classifier-free-guidance sweep against WER *and* repetition, then
             the knob report (level -> DSP intensity / f0 / duration / WER).
``phase4``   the 80 k short-anneal run from v0.7 best, plus a ``hints: false``
             control of the same recipe, and their acceptance gates.
``all``      phase1, phase2, phase3, phase4 (in that order).

Safety model:

* Read-only tasks (everything in ``phase1``/``phase3``'s eval parts) run when
  invoked, unless ``--dry-run``.
* **Training tasks never run unless ``--run-training`` is passed.**  Without
  it they are printed with their generated config path and exact command, so
  the arms can be reviewed before any GPU time is spent.
* Every task's log is written under ``--out`` (default
  ``scratch/hints_v1_runs/<UTC timestamp>``).  A task whose log already exists
  is *not* re-run (``--force`` overrides), so an interrupted suite resumes
  cheaply and the expensive WER matrix is paid for once.
* ``2_training.py`` auto-resumes ``checkpoints_stage2/{exp_name}/last.ckpt``
  whenever it exists (AGENTS.md §6.16), so a task whose derived ``exp_name``
  directory already exists is **blocked**, not launched.  Pass ``--tag`` to
  give the arms a new name.
* Each training arm gets its own generated config (``2_training.py`` has no
  ``--exp_name`` and no override flags, and ``exp_name`` lives in the YAML).
  The generated file is a full copy of ``--config`` with only the listed keys
  changed; ``configs_meta.json`` records the overrides so they are auditable.

Examples::

    # preview everything, run nothing
    python scratch/run_hints_v1.py all --dry-run

    # what Phase 1 would do, then run it
    python scratch/run_hints_v1.py phase1 --list
    python scratch/run_hints_v1.py phase1 --quick          # 24 utts, 4 batches

    # the queued WER matrix is long; run one arm at a time if you prefer
    python scratch/run_hints_v1.py phase1 --only wer_hints

    # materialise the four Phase 2 configs without training
    python scratch/run_hints_v1.py phase2 --tag w1

    # ...and actually train + gate them
    python scratch/run_hints_v1.py phase2 --tag w1 --run-training

Notes:
  * ``--num-items`` is the WER sample size.  The default is 200 (the size the
    v0.9-last measurement used); the binomial 95 % interval at ~2900 words is
    roughly +/-1.5 pp, so do not read sub-1 pp differences even at that size.
  * ``phase1``'s WER matrix is 4 arms x 5 checkpoints = 20 generations of 200
    utterances.  It is the expensive part; use ``--only`` to run one arm.
  * Gates of a training suite need the arm's checkpoint.  If it does not exist
    (training was skipped or failed), the gate is skipped with the reason
    printed, so running ``phase2`` without ``--run-training`` never errors.
  * The scripts called here all print their own docstrings; this file only
    chooses their arguments.  Nothing is re-implemented.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable

#: One canonical name per checkpoint the plan refers to.  Referenced by key so
#: a path typo can only happen in this one place.
CHECKPOINTS: Dict[str, str] = {
    "v0.7_best": "checkpoints_stage2/5090_v0.7_fim2/best-checkpoint.ckpt",
    "v0.7_last": "checkpoints_stage2/5090_v0.7_fim2/last.ckpt",
    "v0.8_last": "checkpoints_stage2/transformer_v0.8/last.ckpt",
    "v0.9_best": "checkpoints_stage2/transformer_v0.9/best-checkpoint.ckpt",
    "v0.9_last": "checkpoints_stage2/transformer_v0.9/last.ckpt",
}

#: The roster the plan's Phase 1 asks to compare (v0.8 has one useful ckpt).
PHASE1_ROSTER = ("v0.7_best", "v0.7_last", "v0.8_last", "v0.9_best",
                 "v0.9_last")

#: Runs that actually own trained hint tables (v0.7 predates them).
HINT_ROSTER = ("v0.8_last", "v0.9_best", "v0.9_last")


@dataclass
class Task:
    """One logged command.

    ``kind`` is ``eval`` (runs by default) or ``train`` (needs
    ``--run-training``).  ``requires`` are paths that must exist before the
    task can run at all (a previous arm's checkpoint); ``produces`` is the
    artefact a training task is expected to leave behind.
    """

    name: str
    argv: List[str]
    note: str = ""
    kind: str = "eval"
    requires: Sequence[str] = ()
    produces: Sequence[str] = ()
    config_overrides: Dict[str, object] = field(default_factory=dict)
    config_path: Optional[str] = None

    def command(self) -> str:
        return " ".join(shlex.quote(a) for a in self.argv)


# --------------------------------------------------------------------------- #
# Declaration helpers
# --------------------------------------------------------------------------- #
def repo_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(REPO, path)


def exists(path: str) -> bool:
    return os.path.exists(repo_path(path))


def ckpt(name: str) -> str:
    """Roster key -> repo-relative checkpoint path (existence checked later)."""
    return CHECKPOINTS[name]


def eval_script(script: str, *args: str) -> List[str]:
    return [PY, os.path.join("scratch", script), *args]


def wer_task(name: str, ckpts: Sequence[str], *, hints: bool, ignore_eos: bool,
             num_items: int, config: str, generator: Optional[str]) -> Task:
    argv = eval_script("eval_stage2_wer.py", "--config", config,
                       "--num_items", str(num_items))
    if hints:
        argv.append("--hints")
    if ignore_eos:
        argv.append("--ignore_eos")
    if generator:
        argv += ["--generator_checkpoint", generator]
    argv += ["--checkpoints", *ckpts]
    note = (f"{'conditional' if hints else 'null-state'} hints, "
            f"{'oracle length' if ignore_eos else 'model EOS'}, "
            f"{num_items} utterances x {len(ckpts)} checkpoint(s)")
    return Task(name, argv, note=note, requires=tuple(ckpts))


def knob_task(name: str, checkpoint: str, *, config: str, generator: Optional[str],
              channels: str, levels: str, num_items: int, wer: bool,
              guidance: Optional[float] = None,
              guidance_books: Optional[str] = None) -> Task:
    argv = eval_script("hint_knob_report.py", "--config", config,
                       "--checkpoint", checkpoint,
                       "--channel", channels, "--levels", levels,
                       "--num_items", str(num_items))
    if generator:
        argv += ["--generator_checkpoint", generator]
    if wer:
        argv.append("--wer")
    if guidance is not None:
        argv += ["--hint_guidance", str(guidance)]
    if guidance_books:
        argv += ["--hint_guidance_books", guidance_books]
    return Task(name, argv,
                note=f"level sweep {channels} on "
                     f"{os.path.basename(os.path.dirname(checkpoint))}/"
                     f"{os.path.basename(checkpoint)}"
                     + ("" if guidance is None else f" (guidance {guidance})"))


def repetition_task(name: str, ckpts: Sequence[str], *, config: str,
                    num_items: int, hints: bool = False,
                    guidance: Optional[float] = None,
                    guidance_books: Optional[str] = None) -> Task:
    argv = eval_script("eval_stage2_repetition.py", "--config", config,
                       "--num_items", str(num_items),
                       "--checkpoints", *ckpts)
    if hints:
        argv.append("--hints")
    if guidance is not None:
        argv += ["--hint_guidance", str(guidance)]
    if guidance_books:
        argv += ["--hint_guidance_books", guidance_books]
    return Task(name, argv,
                note=f"degeneration probe on {len(ckpts)} checkpoint(s)",
                requires=tuple(ckpts))


# --------------------------------------------------------------------------- #
# Config materialisation (2_training.py has no override flags)
# --------------------------------------------------------------------------- #
def write_arm_config(out_dir: str, base_config: str, arm: str,
                     overrides: Dict[str, object]) -> str:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(os.path.join(REPO, base_config))
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)
    path = os.path.join(out_dir, "configs", f"{arm}.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    OmegaConf.save(cfg, path, resolve=False)
    return path


# --------------------------------------------------------------------------- #
# Phase 1
# --------------------------------------------------------------------------- #
def phase1(args, cfg) -> List[Task]:
    n = args.num_items
    roster = [ckpt(k) for k in PHASE1_ROSTER]
    gen = cfg.train.get("generator_checkpoint", None)
    tasks = [
        wer_task("p1/wer_nohint_eos", roster, hints=False, ignore_eos=False,
                 num_items=n, config=args.config, generator=gen),
        wer_task("p1/wer_nohint_ignore_eos", roster, hints=False,
                 ignore_eos=True, num_items=n, config=args.config,
                 generator=gen),
        wer_task("p1/wer_hints_eos", roster, hints=True, ignore_eos=False,
                 num_items=n, config=args.config, generator=gen),
        wer_task("p1/wer_hints_ignore_eos", roster, hints=True,
                 ignore_eos=True, num_items=n, config=args.config,
                 generator=gen),
        eval_script_task(
            "p1/checkpoint_compare",
            ["scratch/eval_stage2_checkpoints.py", "--config", args.config,
             "--batches", str(args.batches),
             "--sample_items", str(args.sample_items),
             "--checkpoints", *roster],
            note="teacher-forced val_loss / FIM / free-running code match on "
                 "identical val batches",
            requires=roster),
        repetition_task("p1/repetition", roster, config=args.config,
                        num_items=max(args.sample_items, 4)),
        eval_script_task(
            "p1/codespace_probe",
            ["scratch/hint_codespace_probe.py", "--n",
             str(args.codespace_n)],
            note="R2(codes) vs R2(speaker): is the covariate even in the code "
                 "space? (CPU)"),
    ]
    for key in HINT_ROSTER:
        tasks.append(eval_script_task(
            f"p1/channel_probe_{key}",
            ["scratch/hint_channel_probe.py", "--config", args.config,
             "--checkpoint", ckpt(key), "--batches", str(max(args.batches // 4, 2)),
             "--sweep", "intensity,register"],
            note=f"per-channel CE liveness + level sweep ({key})",
            requires=[ckpt(key)]))
    return tasks


def eval_script_task(name: str, argv: Sequence[str], note: str = "",
                     **kw) -> Task:
    return Task(name, [PY, *argv], note=note, **kw)


# --------------------------------------------------------------------------- #
# Phase 2
# --------------------------------------------------------------------------- #
PHASE2_ARMS = (
    # slug, hint_film_memory, hint_decoder_proj
    ("baseline", False, False),
    ("film", True, False),
    ("proj", False, True),
    ("film+proj", True, True),
)


def phase2(args, cfg, out_dir: str) -> List[Task]:
    steps = args.phase2_steps
    transfer = args.transfer or ckpt("v0.9_last")
    tasks: List[Task] = []
    arms = []
    for slug, film, proj in PHASE2_ARMS:
        exp = f"hints_v1_p2_{slug.replace('+', '_')}{args.tag_suffix}"
        overrides = {
            "exp_name": exp,
            "transformer.hint_film_memory": bool(film),
            "transformer.hint_decoder_proj": bool(proj),
            "train.max_steps": steps,
            "train.cosine_anneal_end": steps,
            # one short epoch => validation + last.ckpt actually happen
            "train.limit_train_batches": steps,
            "train.val_interval": 1,
            "train.sample_interval": 1,
            "train.keep_ckpts": 1,
        }
        if args.lr is not None:
            overrides["train.lr"] = float(args.lr)
        path = write_arm_config(out_dir, args.config, exp, overrides)
        last = f"checkpoints_stage2/{exp}/last.ckpt"
        tasks.append(Task(
            f"p2/train_{slug}",
            [PY, "2_training.py", "--config", path,
             "--transfer_from", transfer],
            note=f"hint_film_memory={film} hint_decoder_proj={proj}, "
                 f"{steps} steps from "
                 f"{os.path.basename(os.path.dirname(transfer))}/"
                 f"{os.path.basename(transfer)} -> {exp}",
            kind="train", produces=(last,), config_path=path,
            config_overrides=overrides))
        arms.append((slug, exp, last))

    # Gates: offline equivalents of `val_hint_channel_delta/*` (the 2 k-step
    # arm cannot rely on the in-training probe alone, and these are the same
    # numbers on the finished weights), plus the audio knob sweep.
    lasts = [last for _, _, last in arms]
    tasks.append(repetition_task(
        "p2/gate_repetition", lasts, config=args.config,
        num_items=max(args.sample_items, 4), hints=True))
    tasks[-1].requires = lasts
    for slug, exp, last in arms:
        tasks.append(eval_script_task(
            f"p2/gate_channel_probe_{slug}",
            ["scratch/hint_channel_probe.py", "--config", args.config,
             "--checkpoint", last, "--batches", "2",
             "--sweep", "intensity,register"],
            note=f"channel liveness / level sweep ({exp})", requires=[last]))
        knob = knob_task(
            f"p2/gate_knob_{slug}", last, config=args.config,
            generator=cfg.train.get("generator_checkpoint", None),
            channels="intensity,register", levels="all",
            num_items=args.knob_items, wer=False)
        knob.requires = [last]
        tasks.append(knob)
    tasks.append(eval_script_task(
        "p2/gate_val_loss",
        ["scratch/eval_stage2_checkpoints.py", "--config", args.config,
         "--batches", str(args.batches),
         "--sample_items", str(args.sample_items),
         "--checkpoints", *lasts],
        note="teacher-forced val_loss of the four arms (sanity: the recipe "
             "must not move it far)", requires=lasts))
    return tasks


# --------------------------------------------------------------------------- #
# Phase 3
# --------------------------------------------------------------------------- #
def phase3(args, cfg) -> List[Task]:
    checkpoint = args.hints_checkpoint or ckpt("v0.9_last")
    gen = cfg.train.get("generator_checkpoint", None)
    weights = [1.0, 1.5, 2.0, 3.0]
    tasks: List[Task] = []
    for w in weights:
        tasks.append(Task(
            f"p3/wer_guidance_w{w:g}",
            eval_script("eval_stage2_wer.py", "--config", args.config,
                        "--num_items", str(args.guidance_items), "--hints",
                        "--ignore_eos", "--hint_guidance", str(w),
                        *(("--generator_checkpoint", gen) if gen else ()),
                        "--checkpoints", checkpoint),
            note=f"CFG over the hints, w={w} "
                 f"({'identity' if w == 1.0 else 'guided'})"))
        tasks.append(repetition_task(
            f"p3/repetition_w{w:g}", [checkpoint], config=args.config,
            num_items=max(args.sample_items, 4), hints=True, guidance=w))
    # Per-book guidance: pitch books pushed, timbre books untouched.
    books = "1,1,1,1,1,1,1,1,2,2"
    tasks.append(Task(
        "p3/wer_guidance_books",
        eval_script("eval_stage2_wer.py", "--config", args.config,
                    "--num_items", str(args.guidance_items), "--hints",
                    "--ignore_eos", "--hint_guidance_books", books,
                    *(("--generator_checkpoint", gen) if gen else ()),
                    "--checkpoints", checkpoint),
        note="guides only the pitch books (spec0..7 = 1, pitch8..9 = 2)"))
    tasks.append(repetition_task(
        "p3/repetition_books", [checkpoint], config=args.config,
        num_items=max(args.sample_items, 4), hints=True,
        guidance_books=books))
    # `slot` is the one channel with a measured free-running effect, and its
    # level is a *pacing* request ("stretch/shrink this word"), so it belongs in
    # the knob sweep: the `dur_s` column is the coarse pacing readout.
    tasks.append(knob_task(
        "p3/knob_report", checkpoint, config=args.config, generator=gen,
        channels="intensity,register,slot", levels="all",
        num_items=args.knob_items, wer=True))
    if args.knob_guidance is not None:
        tasks.append(knob_task(
            f"p3/knob_report_guidance_w{args.knob_guidance:g}", checkpoint,
            config=args.config, generator=gen, channels="intensity,register",
            levels="all", num_items=args.knob_items, wer=False,
            guidance=args.knob_guidance))
    for task in tasks:                     # every sweep needs the checkpoint
        task.requires = (checkpoint,)
    return tasks


# --------------------------------------------------------------------------- #
# Phase 4
# --------------------------------------------------------------------------- #
def phase4(args, cfg, out_dir: str) -> List[Task]:
    steps = args.phase4_steps
    transfer = args.transfer or ckpt("v0.7_best")
    tasks: List[Task] = []
    runs = [
        # slug, hints, film, proj
        ("final", True, args.injection in ("film", "both"),
         args.injection in ("proj", "both")),
        ("control", False, False, False),
    ]
    lasts = []
    for slug, hints, film, proj in runs:
        exp = f"hints_v1_p4_{slug}{args.tag_suffix}"
        overrides = {
            "exp_name": exp,
            "transformer.hints": bool(hints),
            "transformer.hint_film_memory": bool(film),
            "transformer.hint_decoder_proj": bool(proj),
            "train.lr": float(args.lr) if args.lr is not None else 5e-5,
            "train.max_steps": steps,
            "train.cosine_anneal_end": steps,
            # 10 epochs of `steps // 10` batches: the run ends on an epoch
            # boundary, so `last.ckpt` is the fully-annealed checkpoint.
            "train.limit_train_batches": max(steps // 10, 1),
            "train.val_interval": 1,
            "train.sample_interval": 1,
            "train.keep_ckpts": 2,
            "train.sample_hints": "cond",
        }
        path = write_arm_config(out_dir, args.config, exp, overrides)
        last = f"checkpoints_stage2/{exp}/last.ckpt"
        tasks.append(Task(
            f"p4/train_{slug}",
            [PY, "2_training.py", "--config", path,
             "--transfer_from", transfer],
            note=(f"{steps} steps, lr {overrides['train.lr']}, hints={hints}"
                  f"{'' if not hints else f' film={film} proj={proj}'} "
                  f"from {os.path.basename(os.path.dirname(transfer))}/"
                 f"{os.path.basename(transfer)} -> {exp}"),
            kind="train", produces=(last,), config_path=path,
            config_overrides=overrides))
        lasts.append((slug, hints, exp, last))

    gen = cfg.train.get("generator_checkpoint", None)
    final = lasts[0][3]
    control = lasts[1][3]
    # Acceptance (HINTS_V1_PLAN.md §5 Phase 4): hint-free WER >= v0.7's, the
    # live-channel probes non-zero, the register knob moves f0, the intensity
    # levels move the DSP readout, and the control reproduces v0.7's coherency.
    tasks.append(wer_task(
        "p4/gate_wer_nohint", [final, control], hints=False, ignore_eos=False,
        num_items=args.num_items, config=args.config, generator=gen))
    tasks.append(wer_task(
        "p4/gate_wer_hints", [final], hints=True, ignore_eos=False,
        num_items=args.num_items, config=args.config, generator=gen))
    tasks.append(repetition_task(
        "p4/gate_repetition", [final, control], config=args.config,
        num_items=max(args.sample_items, 4), hints=True))
    tasks.append(eval_script_task(
        "p4/gate_channel_probe",
        ["scratch/hint_channel_probe.py", "--config", args.config,
         "--checkpoint", final, "--batches", "2",
         "--sweep", "intensity,register"],
        note="the three live-channel probes must be non-zero", requires=[final]))
    tasks.append(knob_task(
        "p4/gate_knob", final, config=args.config, generator=gen,
        channels="intensity,register", levels="all",
        num_items=args.knob_items, wer=True))
    tasks.append(eval_script_task(
        "p4/gate_val_loss",
        ["scratch/eval_stage2_checkpoints.py", "--config", args.config,
         "--batches", str(args.batches),
         "--sample_items", str(args.sample_items),
         "--checkpoints", final, control],
        note="teacher-forced val_loss, hints vs control", requires=[final, control]))
    return tasks


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
WER_HEAD = re.compile(r"^=== (\S+)\s+\(epoch (\d+), step (\d+)\)$")
WER_LINE = re.compile(r"^\s+WER ([\d.]+)%\s+S(\d+) D(\d+) I(\d+)")
FLOOR_LINE = re.compile(r"^WER floor .*: ([\d.]+)%")


def summarize(results: Sequence[dict], out_dir: str) -> None:
    """Write ``results.md`` (WER table + every log) and ``manifest.json``."""
    done = ("ok", "cached")            # cached == a finished log from before
    wer_rows = []
    for res in results:
        if res["status"] not in done:
            continue
        tag = step = None
        with open(res["log"], "r", encoding="utf-8", errors="replace") as f:
            floor = None
            for line in f:
                m = FLOOR_LINE.match(line)
                if m:
                    floor = float(m.group(1))
                m = WER_HEAD.match(line)
                if m:
                    tag, step = m.group(1), m.group(3)
                    continue
                m = WER_LINE.match(line)
                if m and tag is not None:
                    wer_rows.append({
                        "task": res["name"], "checkpoint": tag, "step": int(step),
                        "wer": float(m.group(1)), "sub": int(m.group(2)),
                        "del": int(m.group(3)), "ins": int(m.group(4)),
                        "floor": floor})
                    tag = None

    md = [f"# HINTS v1 run {os.path.basename(out_dir)}", ""]
    md.append("Generated by `scratch/run_hints_v1.py`.  Each section is the "
              "verbatim log of one task; the WER table below is parsed from "
              "them.")
    if wer_rows:
        md += ["", "## WER", "",
               "| task | checkpoint | step | WER | S | D | I | floor |",
               "|---|---|---|---|---|---|---|---|"]
        for row in sorted(wer_rows, key=lambda r: (r["task"], r["wer"])):
            fl = "" if row["floor"] is None else f"{row['floor']:.2f}%"
            md.append(f"| {row['task']} | {row['checkpoint']} | {row['step']} | "
                      f"{row['wer']:.2f}% | {row['sub']} | {row['del']} | "
                      f"{row['ins']} | {fl} |")
    md.append("")
    for res in results:
        md += [f"## {res['name']}", "",
               f"`{res['command']}`", "" if res["status"] in done
               else f"**status: {res['status']}** — {res.get('reason', '')}", ""]
        if res["status"] in done:
            text = res.get("log_text") or ""
            if not text and os.path.exists(res["log"]):
                with open(res["log"], "r", encoding="utf-8",
                          errors="replace") as f:
                    text = f.read()
            if res["kind"] == "train":
                head = "\n".join(text.splitlines()[:40])
                md += ["```", head, "... (see the log file for the rest; the "
                       "curves are in TensorBoard)", "```", ""]
            else:
                md += ["```", text.strip(), "```", ""]
    path = os.path.join(out_dir, "results.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"\nsummary: {path}")


def run_task(task: Task, log_path: str) -> dict:
    """Run one task, streaming its output to stdout and the log file.

    The log is written to ``<log>.partial`` and only renamed to ``<log>`` on a
    zero exit code.  ``.log`` therefore means *"this task finished successfully
    once"*, which is what the cache check reads -- a killed run leaves
    ``.partial`` behind and is not mistaken for a completed result.
    """
    result = {"name": task.name, "kind": task.kind, "command": task.command(),
              "note": task.note, "log": log_path, "status": "ok",
              "config": task.config_path, "overrides": task.config_overrides}
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    partial = log_path + ".partial"
    lines: List[str] = []
    with open(partial, "w", encoding="utf-8") as log:
        log.write(f"# {task.name}\n# {task.command()}\n# note: {task.note}\n\n")
        log.flush()
        proc = subprocess.Popen(task.argv, cwd=REPO, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:                        # type: ignore[union-attr]
            sys.stdout.write(line)
            log.write(line)
            lines.append(line)
        rc = proc.wait()
    result["log_text"] = "".join(lines)
    result["status"] = "ok" if rc == 0 else f"failed (rc={rc})"
    if rc == 0:
        os.replace(partial, log_path)
    else:
        result["log"] = partial
        result["reason"] = f"exit code {rc}; partial log {partial}"
    return result


def write_manifest(out_dir: str, args, results: Sequence[dict]) -> None:
    """Persist the task list + statuses (also after every task, so a killed
    suite still records what it had done)."""
    with open(os.path.join(out_dir, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump({"suite": args.suite, "argv": sys.argv,
                   "num_items": args.num_items, "results": results},
                  f, indent=2)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suite", nargs="?", default="all",
                    choices=("phase1", "phase2", "phase3", "phase4", "all"))
    ap.add_argument("--config", default="configs/transformer.yaml")
    ap.add_argument("--out", default=None,
                    help="log directory (default scratch/hints_v1_runs/<UTC>)")
    ap.add_argument("--tag", default="",
                    help="suffix the generated exp_names with this (needed if "
                         "an arm's checkpoint directory already exists)")
    ap.add_argument("--num_items", type=int, default=200,
                    help="WER utterances per arm (default 200)")
    ap.add_argument("--batches", type=int, default=8,
                    help="val batches for teacher-forced comparisons")
    ap.add_argument("--sample_items", type=int, default=6,
                    help="items for repetition / free-running probes")
    ap.add_argument("--knob_items", type=int, default=8,
                    help="val items per level in the knob report")
    ap.add_argument("--guidance_items", type=int, default=100,
                    help="utterances per CFG weight in Phase 3 (24 is too few: "
                         "the ASR floor alone drifts by ~1.5 pp at that size)")
    ap.add_argument("--codespace_n", type=int, default=4000)
    ap.add_argument("--phase2_steps", type=int, default=2000)
    ap.add_argument("--phase4_steps", type=int, default=80000)
    ap.add_argument("--lr", type=float, default=None,
                    help="override train.lr in the training arms")
    ap.add_argument("--injection", default="both",
                    choices=("film", "proj", "both"),
                    help="winning Phase-2 injection for the Phase-4 run")
    ap.add_argument("--transfer", default=None,
                    help="checkpoint the training arms warm-start from "
                         "(default: v0.9 last for Phase 2, v0.7 best for Phase 4)")
    ap.add_argument("--hints_checkpoint", default=None,
                    help="checkpoint Phase 3 sweeps (default: v0.9 last)")
    ap.add_argument("--knob_guidance", type=float, default=None,
                    help="also run the knob report under this CFG weight")
    ap.add_argument("--quick", action="store_true",
                    help="cheap profile: 24 WER utts, 4 batches, 2 sample items")
    ap.add_argument("--only", action="append", default=[],
                    help="only tasks whose name contains one of these "
                         "substrings (repeatable)")
    ap.add_argument("--skip", action="append", default=[],
                    help="skip tasks whose name contains one of these")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every task and exit without running any (the "
                         "training-arm configs are still materialised, since "
                         "they are part of the plan being reviewed)")
    ap.add_argument("--run-training", action="store_true",
                    help="actually launch the GPU training arms (off by default)")
    ap.add_argument("--force", action="store_true",
                    help="re-run tasks whose log already exists")
    args = ap.parse_args(argv)

    if args.quick:
        args.num_items = min(args.num_items, 24)
        args.batches = min(args.batches, 4)
        args.sample_items = min(args.sample_items, 2)
        args.knob_items = min(args.knob_items, 4)
        args.guidance_items = min(args.guidance_items, 12)

    from omegaconf import OmegaConf

    config_path = os.path.join(REPO, args.config)
    cfg = OmegaConf.load(config_path)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    args.tag_suffix = f"_{args.tag}" if args.tag else ""
    out_dir = args.out or os.path.join(REPO, "scratch", "hints_v1_runs", stamp)

    tasks: List[Task] = []
    if args.suite in ("phase1", "all"):
        tasks += phase1(args, cfg)
    if args.suite in ("phase2", "all"):
        tasks += phase2(args, cfg, out_dir)
    if args.suite in ("phase3", "all"):
        tasks += phase3(args, cfg)
    if args.suite in ("phase4", "all"):
        tasks += phase4(args, cfg, out_dir)

    def wanted(task: Task) -> bool:
        if args.only and not any(s in task.name for s in args.only):
            return False
        if any(s in task.name for s in args.skip):
            return False
        return True

    tasks = [t for t in tasks if wanted(t)]

    print(f"suite {args.suite}  ->  {out_dir}")
    print(f"  {len(tasks)} task(s), "
          f"{sum(t.kind == 'train' for t in tasks)} training\n")
    if args.list or args.dry_run:
        for i, task in enumerate(tasks, 1):
            print(f"[{i:02d}] {task.name}  ({task.kind})")
            if task.note:
                print(f"     {task.note}")
            if task.config_overrides:
                print(f"     config {task.config_path}")
                for k, v in task.config_overrides.items():
                    print(f"       {k}: {v}")
            print(f"     $ {task.command()}\n")
        return 0

    os.makedirs(out_dir, exist_ok=True)

    def log_name(task: Task) -> str:
        return os.path.join(out_dir, "logs",
                            task.name.replace("/", "__") + ".log")

    results: List[dict] = []
    for i, task in enumerate(tasks, 1):
        log_path = log_name(task)
        missing = [p for p in task.requires if not exists(p)]
        if missing:
            reason = "missing prerequisite: " + ", ".join(missing)
        elif task.kind == "train" and not args.run_training:
            reason = "training skipped (pass --run-training to launch)"
        elif task.kind == "train" and any(
                exists(p) for p in task.produces):
            reason = ("checkpoint directory already exists; 2_training.py "
                      "would auto-resume it instead of transferring -- pass "
                      "a new --tag")
        elif os.path.exists(log_path) and not args.force:
            reason = "cached (log exists; --force re-runs)"
        else:
            reason = None
        if reason:
            if reason.startswith("cached"):
                status = "cached"
            elif reason.startswith(("missing", "checkpoint")):
                status = "blocked"
            else:
                status = "skipped"
            print(f"[{i:02d}/{len(tasks)}] {task.name}: {status} -- {reason}")
            results.append({"name": task.name, "kind": task.kind,
                            "command": task.command(), "note": task.note,
                            "log": log_path, "status": status,
                            "reason": reason,
                            "config": task.config_path,
                            "overrides": task.config_overrides})
            continue
        print(f"\n[{i:02d}/{len(tasks)}] {task.name}")
        if task.note:
            print(f"    {task.note}")
        print(f"    $ {task.command()}\n")
        results.append(run_task(task, log_path))
        write_manifest(out_dir, args, results)

    write_manifest(out_dir, args, results)
    summarize(results, out_dir)
    counts: Dict[str, int] = {}
    for res in results:
        status = res["status"].split(" ", 1)[0]
        counts[status] = counts.get(status, 0) + 1
    print(", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
          + f", {len(results)} task(s)")
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
