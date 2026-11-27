# Stage-2 web client

A small, mobile-first web app for the stage-2 TTS model: type text, pick a
voice, sample the codec codes and listen.  It mirrors what
`scratch/stage2_repl.py` exposes (phonemization incl. `{ARPAbet}` groups,
speaker ids, temperature/top-k, EOS-clamped lengths, conditioning hints,
checkpoint switching) but as an HTTP service + browser UI.

```
web_client/
  server.py          FastAPI app + job queue (single worker owns the GPU)
  engine.py          model registry, checkpoint inspection, generation
  models.yaml        which models / LoRA checkpoints are exposed
  static/            index.html, app.js, styles.css (no build step)
```

The engine reuses `scratch/stage2_repl.py`'s inference glue on purpose (layout
discovery for EOS/RoPE/hints, phonemization, hint handling, stage-1 audio
decode), so the browser output is the same function the REPL produces.

## Run

From the repository root (conda env `raraai25`):

```bash
python web_client/server.py                 # http://localhost:8765
python web_client/server.py --preload       # load the default model at startup
python web_client/server.py --device cpu --no-preload
python web_client/server.py --host 0.0.0.0 --port 8080
python web_client/server.py --model genshin_v2 --api-key secret   # pinned model
```

Open the printed URL.  On first use the default model is loaded in the
background while the job says `loading model…`; later generations reuse it.
`--preload` (also `server.preload: true` in `models.yaml`) does that at
startup instead.

The server is read-only with respect to checkpoints and log trees: it never
writes a checkpoint, TensorBoard event or filelist.  Generated audio lives in
memory (the job store); only `web_client/outputs/` is created (currently used
only as the REPL's nominal output directory).

## Configuration (`models.yaml`)

```yaml
server:
  single_client: true      # UI shows "single user"; queue is serialized anyway
  fixed_model: null        # set to an id to hide the model picker
  fixed_variant: null      # optionally pin the variant too
  api_key: null            # non-null -> X-API-Key (or ?key=) required
  preload: true
  max_loaded_models: 1     # 1.1 GB transformer + 0.5 GB vocoder per slot
  max_queue: 4
  precision: fp32          # bf16/fp16 for the transformer (vocoder stays fp32)
```

Each entry under `models:` points at a stage-2 config plus checkpoint and an
optional stage-1 `generator_checkpoint` (needed for audio; without it the API
still returns codes and metadata).  `variants:` are selectable checkpoints of
that model — a different `last`/`best`, or a LoRA fine-tune:

```yaml
models:
  genshin_v2:
    label: "Genshin v2.4"
    config: configs/transformer.yaml
    checkpoint: checkpoints_stage2/transformer_hints_v2.4/last.ckpt
    generator_checkpoint: checkpoints_stage1/base_v0.7/last.ckpt
    variants:
      base: { label: "v2.4 · last" }
      best: { checkpoint: .../best-checkpoint.ckpt, label: "v2.4 · best val" }
```

A variant with a `*lora_*` checkpoint is detected from the checkpoint itself:
the server reads `lora_marker` (rank/alpha), infers the adapted module groups
from the adapter tensor names, sets `finetune.enabled` accordingly and only
then builds the model.  The base keys and every adapter therefore load into the
right parameter set; a LoRA checkpoint can never silently fall back to its base
model.  Variants can override `config` when the fine-tune ran with a different
config file (see the shipped `mlp_lora` entry).

All layout switches (EOS, RoPE, hints,
monotone hints, pitch anchors, word BERT, LoRA) are derived from the loaded
checkpoint as well, so a picker entry cannot be scored with the wrong
parameter layout.

## Using it

* **Text** — plain English; `{HH AH0 L OW1}` splices raw ARPAbet (one group =
  one word).  `*sigh*`-style stage directions are stripped by the shared
  phonemizer.
* **Voice** — speaker ids from the model's `train.val_filelist`, with the
  speaker-name prefix of the file (e.g. `Paimon`) and its clip count.  Ids are
  in the stage-1 Generator space (see `speaker_map.py`).
* **Model / LoRA** — the registry above.  Switching unloads the previous model
  when `max_loaded_models: 1`.
* **Advanced** — max length (code frames, 25 Hz), temperature, top-k, seed,
  and, when the checkpoint has them, the hint channels: intensity, register,
  word-pitch shift (semitones), word-timing shift (log2 duration) and the
  per-word BERT channel (`off` / `from current text`).
* **History / server log** — per browser session; job ids remain valid for the
  lifetime of the server's bounded job store.
* **Seeds** — a blank seed is replaced by a random one that is returned in the
  result, so any clip can be reproduced.

## API

```
GET  /api/ping                     {ok,title,auth_required,single_client}
GET  /api/options                  models, variants, per-checkpoint capabilities
GET  /api/speakers?model=&variant= [{id,label,count}]
POST /api/preview  {text}          {phones,n_phone_ids,words}
POST /api/generate {text,...}      -> {job_id}
GET  /api/jobs/{id}                {status:queued|running|done|error, stage, result}
GET  /api/jobs/{id}/audio.wav      (also ?download=1)
GET  /api/jobs/{id}/codes.pt       torch.save({codes,sid,frame_rate,text,...})
GET  /api/health                   device, GPU memory, loaded models, queue
DELETE /api/history                drop this session's jobs
```

Jobs are polled; generation is deliberately not a long-held HTTP request, so a
model load (tens of seconds) and a long clip do not hit browser/proxy timeouts.
`X-Studio-Session` (or `?session=`) namespaces jobs; media URLs also accept it
so `<audio>` and download links work without custom headers.

Example:

```bash
curl -s -X POST localhost:8765/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello there, traveller.","sid":1228,"frames":160}' |
  python -c 'import json,sys; print(json.load(sys.stdin)["job_id"])'
# then GET /api/jobs/<id> until status == "done"
```

## Multi-user / fixed-model deployment

The app is intentionally single-client today, but nothing in the engine assumes
that: jobs are per session, the queue serializes GPU work, and every request
names an explicit model/variant.  To serve a fixed model to several users:

1. set `server.fixed_model` (and optionally `fixed_variant`) so the picker is
   hidden and requests cannot select another checkpoint;
2. set `server.api_key` and hand the key to your users (the UI prompts for it);
3. choose `max_queue`, `max_jobs` and `preload` for the box;
4. keep `max_loaded_models: 1` unless there is GPU memory for more loaded
   checkpoints.  One worker thread is enough for a single GPU; run multiple
   processes on different ports/GPUs if one queue is not enough.

The `/api/health` endpoint is the place to add per-user quotas; the job store
(`JobStore`) is the only mutable per-session state.

## Notes / limits

* Speaker ids are stage-1 ids.  A model trained with a `speaker_map` gets the
  mapping applied by `dataset_codes` when the voice list is built.
* Stage 2 must stay fp32 for *training*; this server's `fp32` default matches
  that convention.  `bf16`/`fp16` cast the transformer's weights (and autocast
  the sampling path) but the stage-1 vocoder still decodes in fp32.
* The audio path needs the stage-1 checkpoint named by the model entry (or
  `train.generator_checkpoint` from its config).  Without it, generation still
  works and returns codes + metadata with `audio_available: false`.
* The server imports `scratch/stage2_repl.py`; a change to that file is picked
  up on restart.
