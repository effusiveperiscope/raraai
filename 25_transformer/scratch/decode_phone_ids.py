#!/usr/bin/env python
"""Decode stage-2 ``phone_ids`` tensors back to ARPAbet phone strings.

``2_preprocess.py`` writes the text side of a stage-2 item as a 1-D ``long``
tensor of ids from the ``therealvul/g2pen_tokenizer_clean`` tokenizer (the same
one ``features.FeatureExtractor`` builds). Those ids are opaque, so when a
stage-2 run looks wrong the first question is usually *"what phones does this
clip actually condition on?"* -- this script answers it without loading the
ASR / aligner / RMVPE weights, a stage-1 checkpoint, or any audio.

Accepts either

* a ``.phone_ids`` tensor file (one clip), or
* a stage-2 *filelist* (``train_stage2.txt`` / ``val_stage2.txt``, lines of
  ``phone_ids|spec_codes|pitch_codes|sid``), in which case every clip's
  ``phone_ids`` is decoded (use ``--limit`` to cap it),

and decodes each one through the g2pen tokenizer. Inputs are detected by: a
``.phone_ids`` suffix is always a tensor; anything else is treated as a filelist
when its first line looks like ``a|b|c|d``.

Examples
--------
    # one tensor file (quote it: corpus names contain spaces/commas)
    python scratch/decode_phone_ids.py "data/base2/<clip>.flac.phone_ids"

    # the first 5 clips of a corpus split
    python scratch/decode_phone_ids.py data/base2/val_stage2.txt --limit 5

    # also show the raw ids and the subword tokens
    python scratch/decode_phone_ids.py some.phone_ids --ids --tokens

    # machine-readable output for scripting
    python scratch/decode_phone_ids.py data/base2/val_stage2.txt --json > phones.json

The script warns (on stderr) about ``[UNK]`` ids, ids outside the vocabulary
and empty sequences, but never exits non-zero for them -- it is an inspection
tool, not a validator. It exits non-zero only when an input cannot be read.

Notes
-----
The tokenizer repo / special tokens mirror ``FeatureExtractor.__init__`` in
``features.py``; keep them in sync if the phonemizer changes. They are duplicated
here on purpose so the script does not import ``features`` (which pulls in
``g2p_en``, NLTK downloads and RMVPE just to create a tokenizer) and so it runs
from any working directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Iterator, TextIO

import torch
from transformers import AutoTokenizer

# Kept identical to features.py (``FeatureExtractor.__init__``) and the shipped
# ``configs/transformer.yaml`` (``train.phone_vocab_size: 1024``).
G2P_TOKENIZER_REPO = 'therealvul/g2pen_tokenizer_clean'
G2P_SPECIAL_TOKENS = {
    'unk_token': "[UNK]",
    'pad_token': "[PAD]",
    'cls_token': "[CLS]",
    'eos_token': "[SEP]",
    'mask_token': "[MASK]",
}
SPACE_ID = 5  # features.FeatureExtractor.space_id; a normal token, not special


def build_tokenizer():
    """The g2pen tokenizer, exactly as ``FeatureExtractor`` builds it."""
    return AutoTokenizer.from_pretrained(G2P_TOKENIZER_REPO, **G2P_SPECIAL_TOKENS)


def _resolve(path: str) -> str:
    """Absolute-ish path that also works when given relative to the repo root.

    Run from a sibling directory, a relative path such as
    ``data/base2/train_stage2.txt`` still resolves against the repository, which
    is where the filelists were written.
    """
    if os.path.exists(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate = os.path.join(repo_root, path)
    return candidate if os.path.exists(candidate) else path


def looks_like_filelist(path: str) -> bool:
    """Distinguish a stage-2 filelist from a bare ``.phone_ids`` tensor file."""
    if path.endswith('.phone_ids'):
        return False
    if path.endswith('.txt'):
        return True
    # Unknown suffix: peek. A filelist line has four pipe-separated fields;
    # a .pt file is binary, so it will not read back as clean UTF-8 text.
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            head = next((ln for ln in f if ln.strip()), '')
    except OSError:
        return False
    return '\0' not in head and head.count('|') >= 3


def iter_items(paths: list[str]) -> Iterator[dict[str, Any]]:
    """Yield ``{'phone_path', 'sid', 'source'}`` for every requested input.

    ``source`` is the path of the file the entry came from (the tensor itself for
    a direct input, the filelist otherwise), which makes the origin of a decoded
    line obvious in multi-filelist runs.
    """
    for raw in paths:
        path = _resolve(raw)
        if looks_like_filelist(path):
            with open(path, 'r', encoding='utf-8') as f:
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or 'longform' in line:
                        # Preprocessing skips empty lines and 'longform' clips;
                        # no tensor exists for them.
                        continue
                    parts = line.split('|')
                    if len(parts) < 1 or not parts[0]:
                        continue
                    yield {
                        'phone_path': parts[0],
                        'sid': int(parts[3]) if len(parts) > 3 else None,
                        'source': f'{path}:{lineno}',
                    }
        else:
            yield {'phone_path': path, 'sid': None, 'source': path}


def load_phone_ids(path: str) -> list[int]:
    """Read one ``.phone_ids`` tensor into a flat list of ints.

    Mirrors ``dataset_codes.CodeDataset.__getitem__``: ``torch.load(...,
    weights_only=True).long().reshape(-1)``. A dict payload is tolerated so a
    checkpoint-style ``{'phone_ids': tensor}`` also works.
    """
    obj = torch.load(_resolve(path), map_location='cpu', weights_only=True)
    if isinstance(obj, dict):
        obj = obj.get('phone_ids', obj)
    if not isinstance(obj, torch.Tensor):
        obj = torch.as_tensor(obj)
    return obj.long().reshape(-1).tolist()


def diagnostics(tok, ids: list[int]) -> dict[str, Any]:
    """Non-fatal findings for one sequence (empty, [UNK], out of vocabulary)."""
    vocab = int(tok.vocab_size)
    unk = tok.unk_token_id
    special = set(tok.all_special_ids or [])
    return {
        'n_ids': len(ids),
        'empty': not ids,
        'unk_ids': [i for i in ids if i == unk],
        'out_of_vocab': [i for i in ids if not 0 <= i < vocab],
        'special_ids': [i for i in ids if i in special],
    }


def fmt_ids(ids: list[int], limit: int = 64) -> str:
    shown = ', '.join(str(i) for i in ids[:limit])
    return f'[{shown}{", ..." if len(ids) > limit else ""}]'


def decode_item(tok, item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Decode one phone_ids tensor into a record (also used for ``--json``)."""
    ids = load_phone_ids(item['phone_path'])
    diag = diagnostics(tok, ids)
    # An out-of-vocabulary id (a corrupt tensor, or a negative value) makes
    # ``decode`` raise; substitute [UNK] for display so the rest still decodes
    # and the warning below reports the damage.
    vocab = int(tok.vocab_size)
    unk = tok.unk_token_id if tok.unk_token_id is not None else 0
    safe_ids = [i if 0 <= i < vocab else unk for i in ids]
    phones = tok.decode(safe_ids,
                        skip_special_tokens=args.skip_special_tokens).strip()
    tokens = tok.convert_ids_to_tokens(safe_ids) if args.tokens else None
    return {
        'phone_path': item['phone_path'],
        'source': item['source'],
        'sid': item['sid'],
        'n_ids': diag['n_ids'],
        'ids': ids,
        'phones': phones,
        'tokens': tokens,
        **{k: diag[k] for k in ('empty', 'unk_ids', 'out_of_vocab', 'special_ids')},
    }


def warn(record: dict[str, Any]) -> None:
    """Print the non-fatal findings for one record to stderr."""
    who = record['phone_path']
    if record['source'] != record['phone_path']:
        who += f' [{record["source"]}]'
    if record['empty']:
        print(f'WARNING: {who}: phone_ids is empty', file=sys.stderr)
    if record['unk_ids']:
        print(f'WARNING: {who}: {len(record["unk_ids"])} [UNK] id(s) '
              f'{fmt_ids(record["unk_ids"], 8)}', file=sys.stderr)
    if record['out_of_vocab']:
        print(f'WARNING: {who}: {len(record["out_of_vocab"])} id(s) outside '
              f'the vocabulary {fmt_ids(record["out_of_vocab"], 8)}',
              file=sys.stderr)
    if record['special_ids']:
        print(f'WARNING: {who}: {len(record["special_ids"])} special id(s) '
              f'{fmt_ids(record["special_ids"], 8)}', file=sys.stderr)


def print_record(record: dict[str, Any], args: argparse.Namespace,
                 out: TextIO) -> None:
    if args.quiet:
        print(record['phones'] or '<empty>', file=out)
        return
    sid = '' if record['sid'] is None else f'  sid={record["sid"]}'
    origin = record['phone_path']
    if record['source'] != record['phone_path']:
        origin += f'   ({record["source"]})'
    print(f'{origin}{sid}  ({record["n_ids"]} ids)', file=out)
    print(f'  phones: {record["phones"]!r}', file=out)
    if args.ids:
        print(f'  ids:    {fmt_ids(record["ids"])}', file=out)
    if args.tokens:
        print(f'  tokens: {record["tokens"]}', file=out)
    print(file=out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Decode stage-2 phone_ids tensors to ARPAbet phone strings.')
    parser.add_argument('inputs', nargs='+',
        help='one or more .phone_ids tensor files and/or stage-2 filelists '
             '(train_stage2.txt / val_stage2.txt)')
    parser.add_argument('--limit', type=int, default=None,
        help='decode at most this many items (default: all)')
    parser.add_argument('--skip_special_tokens', action='store_true',
        help='drop declared special tokens when decoding (default: keep them, '
             'so a corrupt file is visible instead of silently cleaned up)')
    parser.add_argument('--ids', action='store_true',
        help='also print the raw id sequence')
    parser.add_argument('--tokens', action='store_true',
        help='also print the subword tokens (and include them in --json)')
    parser.add_argument('--quiet', action='store_true',
        help='print only the decoded phone string, one line per item')
    parser.add_argument('--json', action='store_true',
        help='emit a JSON array instead of text (nothing else on stdout)')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    tok = build_tokenizer()

    records: list[dict[str, Any]] = []
    for item in iter_items(args.inputs):
        if args.limit is not None and len(records) >= args.limit:
            break
        try:
            records.append(decode_item(tok, item, args))
        except Exception as exc:  # unreadable / not a phone_ids tensor
            print(f'ERROR: {item["phone_path"]} ({item["source"]}): {exc}',
                  file=sys.stderr)
            return 1

    if args.json:
        json.dump(records, sys.stdout, indent=2)
        sys.stdout.write('\n')
        return 0

    for record in records:
        warn(record)
        print_record(record, args, sys.stdout)

    if not records:
        print('No phone_ids found (empty inputs or --limit 0).', file=sys.stderr)
        return 0

    n_ids = sum(r['n_ids'] for r in records)
    n_bad = sum(1 for r in records
                if r['unk_ids'] or r['out_of_vocab'] or r['empty'])
    print(f'Decoded {len(records)} item(s), {n_ids} id(s) total'
          + (f'; {n_bad} item(s) with warnings (see stderr)' if n_bad else '.'),
          file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())