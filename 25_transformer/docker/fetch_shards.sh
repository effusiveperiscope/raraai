#!/usr/bin/env bash
# =============================================================================
# fetch_shards.sh — pull the sharded stage-1 Genshin tensors onto a local disk
# =============================================================================
#
# The stage-1 corpus (`data/genshin_full_stage1/`, 106 064 clips -> 106 064
# `.wave` + `.spec` + `.f0` tensors plus `train.txt`/`val.txt`) is published as
# 1204 `.tar.zst` shards (~226 MB each, ~268 GB compressed, ~325 GB extracted)
# split over shard-0000 .. shard-1203; `shard-1203` additionally carries the two
# filelists.  This script streams them straight from the bucket to the target
# directory:
#
#     curl <shard> | zstd -dc | tar -x -C <dest>
#
# so a shard is decompressed while it downloads and **never lands on disk as a
# tarball** — that is the difference between needing ~325 GB and ~600 GB of
# volume, and it halves the I/O.  Several shards are in flight at once
# (`--jobs`), which is what actually saturates the link: one stream is bound by
# a single TCP connection, 8 are bound by the pipe decompressor.
#
# Resumability: because nothing is stored, a shard cannot be resumed half-way
# (it is only 226 MB, so re-fetching it is cheap).  Completion is therefore
# tracked per shard with a marker file under `<dest>/.shards/done/`, written
# only after the whole pipeline exited 0.  Re-running the same command skips
# finished shards and redoes the rest, so an instance that died mid-download
# costs at most `<jobs>` shards.  `tar -x` overwrites, so re-extracting a
# partially extracted shard is harmless.
#
# The markers live *inside* `--dest` on purpose: they describe that volume's
# contents, so wiping/moving the volume wipes the markers with it.
#
# -----------------------------------------------------------------------------
# Usage (inside the vast.ai image — `cd /workspace/25_transformer` first):
#
#   # everything onto the /data volume, 8 parallel streams
#   docker/fetch_shards.sh --dest /data/genshin_full_stage1
#
#   # and make the shipped configs work unchanged (configs/base.yaml points at
#   # data/genshin_full_stage1/train.txt, i.e. relative to the project dir)
#   docker/fetch_shards.sh --dest /data/genshin_full_stage1 \
#                          --link-project /workspace/25_transformer
#
#   # smoke test: two shards only, no verification of the rest
#   docker/fetch_shards.sh --shards 0-1 --dest /data/test_shards --no-verify
#
#   # nothing to do?  it prints the plan and every shard is skipped
#   docker/fetch_shards.sh --dest /data/genshin_full_stage1
#
#   # disk got wiped but the markers survived?  force a full re-extract
#   docker/fetch_shards.sh --dest /data/genshin_full_stage1 --force
#
# It is safe to run repeatedly and safe to interrupt (Ctrl-C / instance death);
# just run it again.  Exit code 0 == every selected shard extracted and (unless
# --no-verify) every file in the manifest is present at the right size.
#
# -----------------------------------------------------------------------------
# Notes
#
# * The default `--url` is the Cloudflare-fronted mirror.  Point it at the bucket
#   itself if you have one (`https://<b2-endpoint>/file/nirik-ai/...`); the only
#   thing the script asks of the URL is `GET <url>/<shard>` and
#   `GET <url>/manifest.tsv`.
# * `manifest.tsv` (18 MB, `shard<TAB>path<TAB>size`) is what makes the
#   preflight disk-space check exact and the final verification possible.  It is
#   cached in `<dest>/.shards/manifest.tsv`; pass `--manifest FILE` to use a
#   local copy (e.g. the one shipped next to the shards) and never fetch it.
# * Verification compares the size of every file listed in the manifest against
#   the size on disk (`find -printf '%f\t%s'` + one awk pass, a few seconds for
#   318 k files).  It catches truncated tensors from an interrupted extraction —
#   exactly the failure mode a `--skip_existing` preprocessing run would
#   otherwise accept, since `1_preprocess.py` only validates length/shape
#   headers.  Missing entries are printed with their shard so you can re-run
#   with `--shards N`.
# * Nothing here is GPU-specific: it runs fine before/without a GPU, and it can
#   run concurrently with training (it only writes the dataset directory).
# =============================================================================

set -Eeuo pipefail

readonly PROG="${0##*/}"

# --- defaults (override with flags or the matching environment variables) ----
URL="${SHARD_URL:-https://ai.nirik.net/genshin_shards_stage1}"
DEST="${SHARD_DEST:-/data/genshin_full_stage1}"
JOBS="${SHARD_JOBS:-8}"
RETRIES="${SHARD_RETRIES:-5}"
SHARDS_ARG="all"
MANIFEST_ARG=""
LINK_PROJECT=""
FORCE=0
VERIFY="auto"
DISK_CHECK=1
DRY_RUN=0
COUNT="${SHARD_COUNT:-1204}"

say()  { printf '%s\n' "$*"; }
info() { printf '[%s] %s\n' "$PROG" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$PROG" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$PROG" "$*" >&2; exit 1; }

usage() {
    cat <<'EOF'
Stream the sharded stage-1 Genshin tensors onto local disk.

Usage: fetch_shards.sh [options]

Options:
  --dest DIR            extract here (default: /data/genshin_full_stage1).
                        A `.shards/` state dir with the per-shard markers is
                        created inside it.  106 064 clips need ~325 GB.
  --url URL             base URL holding `shard-NNNN.tar.zst` + `manifest.tsv`
                        (default: https://ai.nirik.net/genshin_shards_stage1)
  --jobs N              shards downloaded/decompressed in parallel (default: 8)
  --retries N           attempts per shard before it is declared failed (5)
  --shards SPEC         which shards, e.g. `all` (default), `0-99`, `7` or
                        `0-3,9,40-42` (indices are zero padded to 4 digits)
  --manifest FILE       use this manifest.tsv instead of downloading it
  --link-project DIR    symlink DIR/data/genshin_full_stage1 -> DEST so that
                        configs/base.yaml (data/genshin_full_stage1/train.txt)
                        resolves without editing configs.  The filelist lines
                        inside train.txt/val.txt are hard-coded to that prefix.
  --verify              verify sizes of all manifest files (default when every
                        selected shard is complete)
  --no-verify           skip verification
  --force               re-extract shards that are already marked done
  --no-disk-check       skip the preflight free-space check
  --dry-run             print the plan, download nothing
  -h, --help            this text

Environment: SHARD_URL, SHARD_DEST, SHARD_JOBS, SHARD_RETRIES, SHARD_COUNT
(`SHARD_COUNT` is only used when the manifest is unavailable; it is the number
of shards to assume.  CURL_OPTS is appended to every curl invocation.)
EOF
}

while (( $# )); do
    case "$1" in
        --dest)           DEST="${2:?--dest needs a value}"; shift 2 ;;
        --url)            URL="${2:?--url needs a value}"; shift 2 ;;
        --jobs)           JOBS="${2:?--jobs needs a value}"; shift 2 ;;
        --retries)        RETRIES="${2:?--retries needs a value}"; shift 2 ;;
        --shards)         SHARDS_ARG="${2:?--shards needs a value}"; shift 2 ;;
        --manifest)       MANIFEST_ARG="${2:?--manifest needs a value}"; shift 2 ;;
        --link-project)   LINK_PROJECT="${2:?--link-project needs a value}"; shift 2 ;;
        --verify)         VERIFY=1; shift ;;
        --no-verify)      VERIFY=0; shift ;;
        --force)          FORCE=1; shift ;;
        --no-disk-check)  DISK_CHECK=0; shift ;;
        --dry-run)        DRY_RUN=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                usage >&2; die "unknown argument: $1" ;;
    esac
done

[[ $JOBS =~ ^[0-9]+$ ]] && (( JOBS > 0 ))   || die "--jobs must be a positive integer"
[[ $RETRIES =~ ^[0-9]+$ ]]                  || die "--retries must be an integer"
[[ $COUNT =~ ^[0-9]+$ ]] && (( COUNT > 0 )) || die "SHARD_COUNT must be a positive integer"
URL="${URL%/}"
DEST="${DEST%/}"
STATE="$DEST/.shards"

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
human() {  # bytes -> "12.3 MB"
    if command -v numfmt >/dev/null 2>&1; then
        numfmt --to=iec --format='%.1f' --suffix=B "${1:-0}"
    else
        awk -v b="${1:-0}" 'BEGIN{ split("B KB MB GB TB PB",u," "); i=1;
            while (b >= 1024 && i < 6) { b /= 1024; i++ } printf "%.1f %s", b, u[i] }'
    fi
}

fmt_dur() {  # seconds -> HH:MM:SS
    local s="${1:-0}"
    (( s < 0 )) && s=0
    printf '%02d:%02d:%02d' $(( s / 3600 )) $(( s % 3600 / 60 )) $(( s % 60 ))
}

now() { date +%s; }

shard_name() { printf 'shard-%04d.tar.zst' "$1"; }

# files/bytes of one shard according to the manifest (empty if unknown)
shard_stats() {
    [[ -n ${MANIFEST:-} && -s $STATE/shards.tsv ]] || { printf '0 0'; return; }
    awk -F'\t' -v s="$1" '$1 == s { printf "%s %s", $2, $3; found = 1; exit }
                          END { if (!found) printf "0 0" }' "$STATE/shards.tsv"
}

# ---------------------------------------------------------------------------
# manifest: per-shard file counts and uncompressed byte totals
# ---------------------------------------------------------------------------
load_manifest() {
    local out="$STATE/manifest.tsv"
    if [[ -n $MANIFEST_ARG ]]; then
        [[ -r $MANIFEST_ARG ]] || die "--manifest $MANIFEST_ARG is not readable"
        MANIFEST="$MANIFEST_ARG"
        info "using local manifest: $MANIFEST_ARG"
    elif [[ -s $out ]]; then
        MANIFEST="$out"
        info "using cached manifest: $out"
    else
        info "downloading manifest.tsv (~18 MB) ..."
        # shellcheck disable=SC2086
        if curl -fsSL --connect-timeout 20 --retry 3 --retry-delay 5 \
                ${CURL_OPTS:-} "$URL/manifest.tsv" -o "$out.part"; then
            mv "$out.part" "$out"
            MANIFEST="$out"
        else
            rm -f "$out.part"
            MANIFEST=""
            warn "could not fetch $URL/manifest.tsv"
            warn "continuing without it: no exact disk-space estimate, no size"
            warn "verification.  Pass --manifest FILE to use a local copy."
        fi
    fi

    [[ -n $MANIFEST ]] || return 0
    # shard -> "<name>\t<files>\t<uncompressed bytes>"
    awk -F'\t' 'NR > 1 && $1 ~ /^shard-/ { f[$1]++; b[$1] += $3 }
                END { for (s in f) printf "%s\t%d\t%d\n", s, f[s], b[s] }' \
        "$MANIFEST" | LC_ALL=C sort > "$STATE/shards.tsv"
    info "manifest: $(wc -l < "$STATE/shards.tsv") shards, $(wc -l < "$MANIFEST") entries"
}

# ---------------------------------------------------------------------------
# shard selection
# ---------------------------------------------------------------------------
parse_spec() {  # "0-3,9" -> shard names, one per line
    local spec="$1" part a b i
    local -a parts
    IFS=',' read -r -a parts <<< "$spec"
    for part in "${parts[@]}"; do
        part="${part//[[:space:]]/}"
        [[ -z $part ]] && continue
        if [[ $part =~ ^([0-9]+)-([0-9]+)$ ]]; then
            a=$(( 10#${BASH_REMATCH[1]} )); b=$(( 10#${BASH_REMATCH[2]} ))
            (( a <= b )) || die "bad --shards range: $part"
            for (( i = a; i <= b; i++ )); do shard_name "$i"; echo; done
        elif [[ $part =~ ^[0-9]+$ ]]; then
            shard_name "$(( 10#$part ))"; echo
        elif [[ $part == shard-*.tar.zst ]]; then
            printf '%s\n' "$part"
        else
            die "bad --shards entry: $part (want N, N-M or shard-NNNN.tar.zst)"
        fi
    done
}

select_shards() {
    local name
    SHARD_LIST=()
    if [[ $SHARDS_ARG == all ]]; then
        if [[ -n ${MANIFEST:-} && -s $STATE/shards.tsv ]]; then
            while IFS= read -r name; do SHARD_LIST+=("$name"); done \
                < <(cut -f1 "$STATE/shards.tsv")
        else
            local i
            for (( i = 0; i < COUNT; i++ )); do SHARD_LIST+=("$(shard_name "$i")"); done
        fi
    else
        while IFS= read -r name; do SHARD_LIST+=("$name"); done \
            < <(parse_spec "$SHARDS_ARG" | awk '!seen[$0]++')
    fi

    (( ${#SHARD_LIST[@]} > 0 )) || die "no shards selected"
    printf '%s\n' "${SHARD_LIST[@]}" > "$STATE/.selected.txt"

    TOTAL_N=${#SHARD_LIST[@]}
    if [[ -n ${MANIFEST:-} && -s $STATE/shards.tsv ]]; then
        TOTAL_BYTES=$(awk -F'\t' '
            NR == FNR { sel[$1] = 1; next }
            ($1 in sel) { b += $3 }
            END { printf "%.0f", b + 0 }' "$STATE/.selected.txt" "$STATE/shards.tsv")
    else
        TOTAL_BYTES=0
    fi
}

# ---------------------------------------------------------------------------
# state: per-shard completion markers + a running byte ledger
# ---------------------------------------------------------------------------
# The ledger is one `<shard>\t<uncompressed bytes>` line per finished shard, so
# printing progress costs one awk over a ~1200-line file instead of stat-ing
# 300 k tensors.  It is rebuilt from the markers on every start, which keeps it
# consistent with reality even after a crash.  It carries the shard name so a
# partial run (`--shards 0-99`) reports progress against *its* selection.
rebuild_ledger() {
    local -a markers
    shopt -s nullglob
    markers=("$STATE"/done/*.ok)
    shopt -u nullglob
    if (( ${#markers[@]} )); then
        paste <(printf '%s\n' "${markers[@]##*/}" | sed 's/\.ok$//') \
              <(cut -f2 "${markers[@]}") > "$STATE/.ledger"
        printf '%s\n' "${markers[@]##*/}" | sed 's/\.ok$//' | LC_ALL=C sort > "$STATE/.done.list"
    else
        : > "$STATE/.ledger"
        : > "$STATE/.done.list"
    fi
}

# sets PD_N / PD_BYTES: how much of the *selected* set is done
read_progress() {
    local out
    out=$(awk -F'\t' '
        NR == FNR { sel[$1] = 1; next }
        ($1 in sel) { n++; b += $2 }
        END { printf "%d %.0f", n + 0, b + 0 }' \
        "$STATE/.selected.txt" "$STATE/.ledger")
    read -r PD_N PD_BYTES <<< "$out"
}

prepare_state() {
    mkdir -p "$STATE/done" "$STATE/failed"
    local name
    if (( FORCE )); then
        while IFS= read -r name; do
            rm -f "$STATE/done/$name.ok" "$STATE/failed/$name"
        done < "$STATE/.selected.txt"
    else
        # a previous run may have left failure markers for shards we retry now
        while IFS= read -r name; do rm -f "$STATE/failed/$name"; done \
            < "$STATE/.selected.txt"
    fi
    rebuild_ledger
}

# ---------------------------------------------------------------------------
# preflight: free space must cover the not-yet-extracted shards
# ---------------------------------------------------------------------------
free_bytes() { df -Pk "$DEST" | awk 'NR == 2 { printf "%.0f", $4 * 1024 }'; }

disk_check() {
    (( DISK_CHECK )) || return 0
    (( TOTAL_BYTES > 0 )) || {
        info "size unknown for these shards (no manifest, or shards the manifest" \
             "does not list) — skipping the free-space check"
        return 0
    }

    local need free margin
    need=$(awk -F'\t' '
        FILENAME == ARGV[1] { have[$1] = 1; next }
        FILENAME == ARGV[2] { bytes[$1] = $3; next }
        ($1 in have) { next }
        { n += bytes[$1] }
        END { printf "%.0f", n + 0 }' \
        "$STATE/.done.list" "$STATE/shards.tsv" "$STATE/.selected.txt")
    free=$(free_bytes)
    margin=$(( 4 * 1024 * 1024 * 1024 ))   # 4 GiB headroom for filelists/metadata

    if (( need + margin > free )); then
        die "$(human "$need") still to extract but only $(human "$free") free on $DEST
       Free space, point --dest elsewhere, or pass --no-disk-check."
    fi
    info "disk: $(human "$need") to extract, $(human "$free") free on $DEST"
}

# ---------------------------------------------------------------------------
# fetching
# ---------------------------------------------------------------------------
# One attempt at one shard.  The three stages share a pipe, so:
#   * no tarball is ever written (needs ~600 GB -> ~325 GB of volume), and
#   * `set -o pipefail` makes *any* stage's failure fail the attempt (a truncated
#     download ends in an `Unexpected EOF` from zstd and a `tar: unexpected EOF`).
# curl's own `--retry` must NOT be used here: a retry restarts the HTTP transfer
# but the bytes already handed to zstd/tar cannot be un-sent, so the stream would
# be corrupt.  Retrying is done one level up, by re-running the whole pipeline.
# `--speed-limit/--speed-time` is the stall detector: an idle connection (the
# Cloudflare mirror answered 521/503 at times) aborts in ~2 min instead of
# hanging forever, and the attempt is retried.
stream_shard() {
    local name="$1"
    # shellcheck disable=SC2086
    curl -fsSL --connect-timeout 20 --speed-limit 5000 --speed-time 120 \
         --retry 0 ${CURL_OPTS:-} "$URL/$name" \
      | zstd -dc -q \
      | tar --no-same-owner --warning=no-timestamp -xf - -C "$DEST"
}

process_shard() {
    local name="$1"
    local marker="$STATE/done/$name.ok"
    local files bytes attempt=1 delay=5 rc=1 stats
    local -a st

    (( FORCE )) || [[ ! -e $marker ]] || return 0   # already done

    stats=$(shard_stats "$name")
    read -r -a st <<< "$stats"
    files="${st[0]:-0}"; bytes="${st[1]:-0}"

    while (( attempt <= RETRIES )); do
        if stream_shard "$name"; then
            printf '%s\t%s\n' "$files" "$bytes" > "$marker"
            printf '%s\t%s\n' "$name" "$bytes" >> "$STATE/.ledger"
            report_progress "$name"
            return 0
        else
            rc=$?
        fi
        warn "$name: attempt $attempt/$RETRIES failed (rc=$rc)"
        if (( attempt < RETRIES )); then
            (( delay > 60 )) && delay=60
            sleep "$delay"
            delay=$(( delay * 2 ))
        fi
        attempt=$(( attempt + 1 ))
    done

    : > "$STATE/failed/$name"
    warn "$name: giving up after $RETRIES attempts — files from the partial" \
         "attempt may remain; re-run the same command to redo this shard"
    return 1
}

report_progress() {
    local name="$1" elapsed rate eta pct
    read_progress
    elapsed=$(( $(now) - T0 ))
    (( elapsed < 1 )) && elapsed=1
    rate=$(( PD_BYTES / elapsed ))

    if (( TOTAL_BYTES > 0 )); then
        pct=$(awk -v d="$PD_BYTES" -v t="$TOTAL_BYTES" 'BEGIN { printf "%.1f", 100 * d / t }')
        eta=$(awk -v d="$PD_BYTES" -v t="$TOTAL_BYTES" -v e="$elapsed" \
              'BEGIN { r = d / e; printf "%d", (r > 0 ? (t - d) / r : 0) }')
        printf '[%4d/%d] %5s%%  %-21s  %8s  %9s/s  elapsed %s  eta %s\n' \
            "$PD_N" "$TOTAL_N" "$pct" "$name" "$(human "$PD_BYTES")" \
            "$(human "$rate")" "$(fmt_dur "$elapsed")" "$(fmt_dur "$eta")"
    else
        printf '[%4d/%d]  %-21s  elapsed %s\n' \
            "$PD_N" "$TOTAL_N" "$name" "$(fmt_dur "$elapsed")"
    fi
}

CHILD_PIDS=()
on_interrupt() {
    trap - INT TERM
    printf '\n[%s] interrupted — %s/%s selected shards were complete on disk;\n' \
        "$PROG" "${PD_N:-?}" "$TOTAL_N" >&2
    printf '[%s] partial shards are not marked done, re-run the same command to resume.\n' \
        "$PROG" >&2
    local p
    for p in "${CHILD_PIDS[@]:-}"; do kill -TERM "$p" 2>/dev/null || true; done
    wait 2>/dev/null || true
    exit 130
}

# Fixed-size worker pool: keep `--jobs` shards in flight, reap with `wait -n`.
run_pool() {
    local i=0 running=0
    CHILD_PIDS=()
    while (( i < TOTAL_N || running > 0 )); do
        while (( running < JOBS && i < TOTAL_N )); do
            process_shard "${SHARD_LIST[i]}" &
            CHILD_PIDS+=("$!")
            i=$(( i + 1 )); running=$(( running + 1 ))
        done
        # `wait -n` (bash >= 4.3) reaps exactly one child; a failing shard must
        # not abort the pool, its marker in .shards/failed/ is the verdict.
        wait -n 2>/dev/null || true
        running=$(( running - 1 ))
    done
    wait 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# verification: every manifest file present and at the size the manifest says
# ---------------------------------------------------------------------------
do_verify() {
    if [[ -z ${MANIFEST:-} ]]; then
        warn "no manifest — skipping size verification"
        return 0
    fi
    info "verifying on-disk sizes against the manifest ..."
    # refresh .done.list from the markers on disk: the copy built by
    # prepare_state predates this run's downloads.
    rebuild_ledger
    local present="$STATE/.present.tsv" marks="$STATE/.done.list" out="$STATE/.verify.out"
    find "$DEST" -maxdepth 1 -type f -printf '%f\t%s\n' | LC_ALL=C sort > "$present"

    awk -F'\t' '
        FILENAME == ARGV[1] { present[$1] = $2; next }
        FILENAME == ARGV[2] { sel[$1] = 1; next }
        ($1 in sel) {
            n++
            if (!($2 in present)) {
                missing++
                if (missing <= 20) printf "MISSING  %-64s  %s\n", $2, $1
                next
            }
            if (present[$2] != $3) {
                bad++
                if (bad <= 20)
                    printf "SIZE     %-64s  %s  (disk %s, manifest %s)\n", \
                           $2, $1, present[$2], $3
            }
        }
        END { printf "SUMMARY\t%d\t%d\t%d\n", n + 0, missing + 0, bad + 0 }
    ' "$present" "$marks" "$MANIFEST" > "$out"

    local checked=0 missing=0 bad=0
    IFS=$'\t' read -r _ checked missing bad < <(grep '^SUMMARY' "$out") || true
    grep -v '^SUMMARY' "$out" || true

    if (( ${missing:-0} == 0 && ${bad:-0} == 0 )); then
        info "verified $checked files: all present, all sizes match"
        return 0
    fi
    warn "$checked files checked: $missing missing, $bad with a wrong size"
    warn "re-run the shards named above, e.g. --shards 17,301, then re-verify"
    return 1
}

# ---------------------------------------------------------------------------
# optional: make `data/genshin_full_stage1/...` resolve from the project dir
# ---------------------------------------------------------------------------
do_link_project() {
    local proj="${LINK_PROJECT%/}" link
    [[ -d $proj ]] || die "--link-project $proj is not a directory"
    mkdir -p "$proj/data"
    # The link *name* is fixed: every line of train.txt/val.txt reads
    #   data/genshin_full_stage1/<basename>.wav.{wave,spec,f0}|...
    # so it must be linked as `genshin_full_stage1` wherever --dest points.
    link="$proj/data/genshin_full_stage1"
    if [[ -L $link ]]; then
        ln -sfn "$DEST" "$link"
        info "updated symlink $link -> $DEST"
    elif [[ -e $link ]]; then
        warn "$link exists and is not a symlink — leaving it alone"
        return 0
    else
        ln -s "$DEST" "$link"
        info "linked $link -> $DEST"
    fi
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
main() {
    local t
    for t in curl zstd tar awk sed paste find; do
        command -v "$t" >/dev/null 2>&1 \
            || die "missing required tool '$t' (needs curl, zstd, tar, awk, coreutils/findutils)"
    done
    [[ -f $PWD/configs/base.yaml ]] \
        || warn "cwd is not the project dir — harmless here, only --dest matters for fetching"

    mkdir -p "$DEST" "$STATE/done" "$STATE/failed" 2>/dev/null \
        || die "cannot create $DEST (permission denied).
       A vast.ai volume mount is root-owned; fix it once with:
         sudo mkdir -p $DEST && sudo chown \$(id -u):\$(id -g) $DEST"
    [[ -w $DEST ]] || die "$DEST is not writable by $(id -un) — same fix as above"

    # Two concurrent runs against the same --dest would fight over .selected.txt
    # and the markers (and write the same tensors).  Cheap advisory lock.
    if command -v flock >/dev/null 2>&1; then
        exec 9>"$STATE/.lock"
        flock -n 9 || die "another fetch_shards.sh is already running on $DEST"
    fi

    load_manifest
    select_shards
    prepare_state
    read_progress

    info "url:    $URL"
    info "dest:   $DEST"
    info "shards: $TOTAL_N selected, $PD_N already done, $(( TOTAL_N - PD_N )) to fetch, $JOBS in parallel"
    (( TOTAL_BYTES > 0 )) && info "size:   $(human "$TOTAL_BYTES") extracted in total"

    if (( DRY_RUN )); then
        info "dry run — first: $URL/${SHARD_LIST[0]}"
        (( TOTAL_N > 1 )) && info "dry run — last:  $URL/${SHARD_LIST[TOTAL_N - 1]}"
        disk_check
        info "nothing downloaded"
        return 0
    fi

    if (( TOTAL_N == PD_N )); then
        info "all selected shards are already extracted — nothing to download"
    else
        disk_check
        trap on_interrupt INT TERM
        T0=$(now)
        run_pool
        trap - INT TERM
    fi

    # per-shard verdicts
    local failed_n=0 name
    local -a failed_names=()
    while IFS= read -r name; do
        if [[ -e $STATE/failed/$name ]]; then
            failed_n=$(( failed_n + 1 ))
            failed_names+=("$name")
        fi
    done < "$STATE/.selected.txt"

    read_progress
    info "on disk: $PD_N/$TOTAL_N selected shards ($(human "$PD_BYTES"))"

    local verify_rc=0
    if [[ $VERIFY == 1 ]] \
       || { [[ $VERIFY == auto ]] && (( failed_n == 0 && PD_N == TOTAL_N )); }; then
        do_verify || verify_rc=1
    else
        info "skipping verification (pass --verify to force it)"
    fi

    [[ -n $LINK_PROJECT ]] && do_link_project

    if (( failed_n > 0 )); then
        warn "$failed_n shard(s) failed: ${failed_names[*]:0:10}"
        warn "re-run the same command to retry only those"
        return 1
    fi
    (( verify_rc == 0 )) || return 1

    info "done."
    if [[ -z $LINK_PROJECT ]]; then
        info "to make configs/base.yaml find this dataset (data/genshin_full_stage1/...):"
        info "  ln -s $DEST $(pwd)/data/genshin_full_stage1"
    fi
    return 0
}

main "$@"
