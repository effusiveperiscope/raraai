#!/usr/bin/env python
"""Ad-hoc smoke test for stage-2 phonemization (the two `2_preprocess.py` paths).

``phone_ids`` can be produced in two ways, and this script exercises both so a
g2p / dictionary / tokenizer change can be checked in seconds instead of by
re-running a corpus:

* **text path**  -- `FeatureExtractor.phonemize_text(text)`, i.e. what
  ``2_preprocess.py --only phones --phones_from transcript`` writes. Tags the
  whole transcript at once (`HorsePhonemizer.phonemize_sentence`), no audio, no
  ASR, no GPU weights.
* **audio path** -- `FeatureExtractor.align(...)` + `phonemize_items(...)`,
  i.e. what a normal run / ``--phones_from aligner`` writes. Recovers word
  timings from the forced aligner (or from ASR when no transcript is given) and
  phonemizes the *aligned tokens* with the tags of the whole utterance.

For every run it prints the per-word table (word | Penn tag | phones | id count)
plus the decoded phone string, and runs a handful of assertions (non-empty ids,
no ``[UNK]``, ids inside the vocabulary, ``decode(ids)`` round-trip). The two
paths are then compared on the same utterance: differences are *informational*
only, because the audio path tokenizes every word separately (each word starts
with a space-marked token) while the text path tokenizes the whole sentence
once, and that changes the ids by construction.

Punctuation is checked explicitly. The Qwen forced aligner strips it from its
tokens (``Qwen3ForceAlignProcessor.clean_token`` keeps letters, digits and
``'`` only), so ``align()`` puts it back from the transcript / ASR text with
``features.attach_punctuation`` and ``phonemize_items`` appends it to the
word's phones. Both paths must therefore carry the punctuation the transcript
has -- the same shape the published stage-2 corpora were built with.

Examples
--------
    # defaults: a few heteronym-heavy sentences + test.wav (ASR transcription)
    python test_phonemize.py

    # only the text path (cheap, CPU-only)
    python test_phonemize.py --skip_audio

    # only the audio path, forced alignment against a known transcript
    python test_phonemize.py --skip_text --audio test.wav \
        --transcript "Hey. you don't sound too excited."

    # pin a pronunciation (fails with exit code 1 if it is not produced)
    python test_phonemize.py --skip_audio \
        --text "Once again, you've read my mind." --expect REH1D

The exit code is 0 when every check passes, 1 otherwise.
"""
from __future__ import annotations

import argparse
import difflib
import itertools
import os
import re
import sys
import time

import librosa
import torch
from omegaconf import OmegaConf

from features import FeatureExtractor
from utils import win_longpath

# Sentences that exercise the interesting g2p behaviour: heteronyms
# (read / refuse / present / record / use / close), contractions, punctuation.
DEFAULT_TEXTS = [
    "Once again, you've read my mind, Spike.",
    "I refuse to present the record.",
    "Use the tool to close the door.",
    "Huh?",
]

# Strips everything that is not a phoneme character or a digit, so the two
# paths' phone strings can be compared without tripping over punctuation.
_PHONE_NOISE = re.compile(r"[^A-Z0-9]+")

# Punctuation characters the g2pen tokenizer has ids for, i.e. the ones that
# must survive from the transcript into `phone_ids` (`.` `,` `?` `!` `-`).
_KEPT_PUNCT = ',.?!-'


class Checks:
    """Tiny PASS/FAIL collector, so the script can exit non-zero on failure."""

    def __init__(self) -> None:
        self.results: list[tuple[bool, str, str]] = []

    def check(self, ok: bool, name: str, detail: str = "") -> bool:
        ok = bool(ok)
        self.results.append((ok, name, detail))
        print(f'  [{"PASS" if ok else "FAIL"}] {name}'
              + (f'  ({detail})' if detail else ''))
        return ok

    @property
    def n_failed(self) -> int:
        return sum(1 for ok, _, _ in self.results if not ok)

    def summary(self) -> int:
        n = len(self.results)
        failed = self.n_failed
        print()
        print('=' * 78)
        print(f'Summary: {n - failed}/{n} check(s) passed'
              + (f', {failed} FAILED' if failed else ''))
        for ok, name, detail in self.results:
            if not ok:
                print(f'  FAILED: {name}' + (f'  ({detail})' if detail else ''))
        print('=' * 78)
        return 0 if failed == 0 else 1


def print_header(title: str) -> None:
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


def normalise_phones(phones: str) -> str:
    """Phone string with punctuation/word separators collapsed to spaces."""
    return ' '.join(_PHONE_NOISE.sub(' ', phones.upper()).split())


def fmt_ids(ids: list[int], limit: int) -> str:
    shown = ', '.join(str(i) for i in ids[:limit])
    return f'[{shown}{", ..." if len(ids) > limit else ""}] ({len(ids)} ids)'


def unknown_ids(fe: FeatureExtractor, ids: list[int]) -> list[int]:
    """Ids the g2pen tokenizer could not map (`[UNK]`); any hit is a problem."""
    unk = fe.tokenizer_g2p.unk_token_id
    return [i for i in ids if i == unk]


def vocab_size(fe: FeatureExtractor) -> int:
    return int(fe.tokenizer_g2p.vocab_size)


def check_ids(fe: FeatureExtractor, ids: list[int], phones: str,
              label: str, checks: Checks) -> None:
    """The invariants `2_preprocess.py` relies on for a usable tensor."""
    checks.check(bool(ids), f'{label}: phone_ids non-empty', f'{len(ids)} ids')
    unk = unknown_ids(fe, ids)
    checks.check(not unk, f'{label}: no [UNK] ids', f'{len(unk)} unknown')
    n_vocab = vocab_size(fe)
    bad = [i for i in ids if not 0 <= i < n_vocab]
    checks.check(not bad, f'{label}: all ids inside the vocabulary',
                 f'vocab={n_vocab}' + (f', bad={bad[:5]}' if bad else ''))
    decoded = fe.tokenizer_g2p.decode(ids).strip() if ids else ''
    checks.check(decoded == phones,
                 f'{label}: decode(ids) round-trips to the phone string',
                 '' if decoded == phones else f'decoded={decoded!r}')
    print(f'         phones: {phones!r}')
    print(f'         ids:    {fmt_ids(ids, 40)}')


def print_word_table(words: list[str], tags: list[str],
                     per_word_phones: list[str],
                     per_word_ids: list[list[int]]) -> None:
    print(f'  {"#":>3}  {"word":<22} {"tag":<5} {"n":>3}  phones')
    for i, (word, tag, phones, ids) in enumerate(
            zip(words, tags, per_word_phones, per_word_ids)):
        print(f'  {i:>3}  {word!r:<22} {tag:<5} {len(ids):>3}  {phones}')


def run_text_path(fe: FeatureExtractor, text: str,
                  checks: Checks) -> tuple[str, list[int]]:
    """`--phones_from transcript`: whole-sentence tagging, no audio/ASR/GPU."""
    print_header(f'TEXT PATH (phonemize_text / --phones_from transcript): '
                 f'{text!r}')
    words = text.split()
    tags = fe.hphzr.tag_words(words)
    # `phonemize_sentence` is exactly this per-word loop, just with the tags of
    # the whole utterance (heteronym context) instead of each word in isolation.
    per_word_phones = [fe.hphzr.phonemize(w, tag=t) for w, t in zip(words, tags)]
    per_word_ids = [fe.tokenizer_g2p(p)['input_ids'] for p in per_word_phones]
    phones = fe.hphzr.phonemize_sentence(text, tags=tags)
    ids = fe.phonemize_text(text)

    print_word_table(words, tags, per_word_phones, per_word_ids)
    checks.check(phones == ' '.join(per_word_phones),
                 'text path: phonemize_sentence == per-word phonemize(tag=...)')
    check_ids(fe, ids, phones, 'text path', checks)

    # The reference behaviour for punctuation: g2p_en passes it through, so the
    # phones keep "SPAY1K." -> the `.` is the last id of the word. The audio
    # path has to reproduce this (see `attach_punctuation` in features.py).
    expected = {ch for ch in text if ch in _KEPT_PUNCT}
    got = ''.join(ch for ch in phones if ch in _KEPT_PUNCT)
    checks.check(all(ch in got for ch in expected),
                 'text path: punctuation is phonemized',
                 f'text={sorted(expected)} phones={got!r}')
    return phones, ids


def run_audio_path(fe: FeatureExtractor, audio_path: str,
                   transcript: str | None, language: str,
                   checks: Checks) -> tuple[str, list[int], str]:
    """`--phones_from aligner`: forced alignment / ASR, then per-token g2p.

    Returns ``(phones, ids, transcript_used)``; ``transcript_used`` is the
    caller transcript, or the tokens the ASR produced when none was given.
    """
    source = 'forced alignment' if transcript else 'ASR'
    print_header(f'AUDIO PATH (align + phonemize_items / --phones_from aligner, '
                 f'{source}): {audio_path}')
    wav, sr = librosa.load(win_longpath(audio_path), sr=16000)
    duration = len(wav) / sr
    print(f'  audio: {duration:.2f}s @ {sr} Hz')
    checks.check(sr == 16000, 'audio path: loaded at 16 kHz', f'{sr} Hz')

    t0 = time.time()
    items = fe.align(wav, language=language, transcription=transcript)
    print(f'  align(): {time.time() - t0:.2f}s -> {len(items)} item(s)')
    if not checks.check(bool(items), 'audio path: aligner returned items'):
        return '', [], transcript or ''

    words = [str(it['text']) for it in items]
    tags = fe.hphzr.tag_words(words)
    # phonemize_items() tags the items together and attaches phones/phone_ids;
    # the tags above are recomputed only for the printed table.
    tagged = fe.phonemize_items(items, transcript=transcript)
    per_word_phones = [it['phones'] for it in tagged]
    per_word_ids = [list(it['phone_ids']) for it in tagged]

    print(f'  {"#":>3}  {"word":<22} {"tag":<5} {"n":>3}  '
          f'{"span":<13}  phones')
    for i, (item, tag, phones, ids) in enumerate(
            zip(items, tags, per_word_phones, per_word_ids)):
        print(f'  {i:>3}  {item["text"]!r:<22} {tag:<5} {len(ids):>3}  '
              f'{item["start_time"]:6.2f}-{item["end_time"]:<6.2f}  {phones}')

    phones = ' '.join(per_word_phones)
    ids = list(itertools.chain.from_iterable(per_word_ids))
    check_ids(fe, ids, phones, 'audio path', checks)

    # Punctuation: the aligner's tokenizer strips it, `attach_punctuation` puts
    # it back and `phonemize_items` appends it to the word's phones. Without
    # that the codes silently lose the pause cue the published stage-2 corpora
    # carry in (almost) every clip.
    item_punct = ''.join(str(it.get('punct', '')) for it in items)
    phone_punct = ''.join(ch for ph in per_word_phones
                          for ch in ph if ch in _KEPT_PUNCT)
    if transcript:
        expected = ''.join(ch for ch in transcript if ch in _KEPT_PUNCT)
        checks.check(all(ch in phone_punct for ch in set(expected)),
                     'audio path: transcript punctuation survives into the '
                     'phones', f'transcript={expected!r} phones={phone_punct!r}')
    if item_punct:
        checks.check(phone_punct == item_punct,
                     'audio path: recovered punctuation reaches the phones',
                     f'items={item_punct!r} phones={phone_punct!r}')
    print(f'  punctuation: items={item_punct!r} phones={phone_punct!r}')

    covered = max((float(it['end_time']) for it in items), default=0.0)
    checks.check(covered <= duration + 0.05,
                 'audio path: final word ends inside the clip',
                 f'{covered:.2f}s of {duration:.2f}s')
    return phones, ids, transcript or ' '.join(words)


def compare_paths(text_phones: str, text_ids: list[int],
                  audio_phones: str, audio_ids: list[int],
                  transcript: str, checks: 'Checks',
                  strict: bool = True) -> None:
    """Compare the two paths' ids.

    With a caller-supplied transcript both paths phonemize the same words and
    tokenize them as one sentence, so the ids must be **identical** (`strict`).
    In ASR mode the reference text is the ASR output, while the aligner reports
    its own token list -- which can drop a token it cannot place -- so a
    difference there is real data, and only reported.
    """
    print_header('CROSS-CHECK: transcript path vs aligner path'
                 + ('' if strict else ' (informational)'))
    print(f'  text used:    {transcript!r}')
    print(f'  text phones:  {text_phones!r}')
    print(f'  audio phones: {audio_phones!r}')
    same = text_ids == audio_ids
    if strict:
        checks.check(same, 'ids identical across the two paths',
                     '' if same else f'{len(text_ids)} vs {len(audio_ids)} ids')
    else:
        print(f'  ids identical: {same}  ({len(text_ids)} vs '
              f'{len(audio_ids)} ids; the ASR text and the aligner token list '
              f'may differ)')
    if not same:
        first_diff = next((i for i, (a, b) in enumerate(zip(text_ids, audio_ids))
                           if a != b), None)
        if first_diff is None:
            first_diff = min(len(text_ids), len(audio_ids))
        print(f'  first id difference at index: {first_diff}')

    norm_text = normalise_phones(text_phones).split()
    norm_audio = normalise_phones(audio_phones).split()
    if norm_text == norm_audio:
        print('  phone tokens match after punctuation normalisation.')
        return

    print('  phone tokens differ (expected for punctuation / split contractions);'
          ' diff (- text, + audio):')
    for line in difflib.unified_diff(norm_text, norm_audio, lineterm='',
                                     fromfile='text', tofile='audio', n=3):
        if line.startswith(('---', '+++')):
            continue
        print(f'    {line}')


def build_extractor(config_path: str, device: str,
                    need_qwen: bool) -> tuple[FeatureExtractor, float]:
    t0 = time.time()
    hp = OmegaConf.load(config_path)
    fe = FeatureExtractor(hp=hp, device=device, load_qwen=need_qwen)
    return fe, time.time() - t0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Smoke test for stage-2 phonemization (text and audio paths).')
    parser.add_argument('--text', action='append', default=None,
        help='transcript to phonemize via the text path (repeatable; default: '
             'a small heteronym-rich set)')
    parser.add_argument('--audio', action='append', default=None,
        help='audio file(s) for the audio path (default: test.wav if present)')
    parser.add_argument('--transcript', type=str, default=None,
        help='known transcript for --audio: forces alignment instead of ASR; '
             'pass a single --audio when using it')
    parser.add_argument('--language', type=str, default='English',
        help='language hint for the ASR / forced aligner (default: English)')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
        help='stage-1 config, only used to build the (otherwise unused) hp')
    parser.add_argument('--device', type=str, default=None,
        help='torch device (default: cuda:0 if available, else cpu)')
    parser.add_argument('--skip_text', action='store_true',
        help='skip the text/transcript path')
    parser.add_argument('--skip_audio', action='store_true',
        help='skip the audio/aligner path (no Qwen weights, CPU only)')
    parser.add_argument('--expect', action='append', default=None,
        help='substring that must appear in the text-path phone string '
             '(repeatable; e.g. --expect REH1D)')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    texts = [] if args.skip_text else (args.text or DEFAULT_TEXTS)
    audios: list[str] = []
    if not args.skip_audio:
        for audio_path in (args.audio or ['test.wav']):
            if os.path.exists(win_longpath(audio_path)):
                audios.append(audio_path)
            else:
                print(f'WARNING: audio not found, skipping: {audio_path}')
        if args.transcript is not None and len(audios) > 1:
            print('WARNING: --transcript is applied to every --audio; pass one.')

    if not texts and not audios:
        print('Nothing to do: both paths are disabled or no input is available.')
        return 2

    checks = Checks()
    print(f'device: {device} | text path: {len(texts)} input(s) | '
          f'audio path: {len(audios)} file(s)')
    fe, load_s = build_extractor(args.config, device, need_qwen=bool(audios))
    print(f'FeatureExtractor loaded in {load_s:.2f}s '
          f'(qwen={"yes" if audios else "no"}) | '
          f'dictionary: {len(fe.hphzr.horsedict)} entries | '
          f'tokenizer vocab: {vocab_size(fe)}')
    print(f'special ids: space={fe.space_id} unk={fe.tokenizer_g2p.unk_token_id} '
          f'pad={fe.tokenizer_g2p.pad_token_id}')

    text_results = []
    for text in texts:
        phones, ids = run_text_path(fe, text, checks)
        text_results.append((text, phones, ids))

    if args.expect:
        print_header('EXPECT CHECKS (text path)')
        if not text_results:
            print('  (no text inputs, nothing to expect)')
        joined = '\n'.join(p for _, p, _ in text_results)
        for needle in args.expect:
            ok = needle.upper() in joined.upper()
            checks.check(ok, f'expected {needle!r} in the text-path phones',
                         '' if ok else f'got {joined!r}')

    for audio_path in audios:
        audio_phones, audio_ids, transcript_used = run_audio_path(
            fe, audio_path, args.transcript, args.language, checks)
        if audio_phones and (text_results or args.transcript):
            # Compare against the caller transcript, or against the text path of
            # the ASR tokens when the clip had to be transcribed first.
            compare_text = args.transcript or transcript_used
            compare_paths(fe.hphzr.phonemize_sentence(compare_text),
                          fe.phonemize_text(compare_text),
                          audio_phones, audio_ids, compare_text, checks,
                          strict=bool(args.transcript))

    return checks.summary()


if __name__ == '__main__':
    sys.exit(main())
