"""Mine high-intensity candidates from the unlabeled `diverse_full_linux.txt`.

Unlike `mine_high_intensity.py` (which ranks already-labeled clips), this scans
the big unlabeled pool, restricted to Genshin/MLP, and scores it with the
classifiers trained on the fixed labeled Genshin/MLP/MareFairData cache.

The pool is 204k rows; after excluding already-annotated stems there are ~143k
Genshin/MLP clips.  Feature extraction is the expensive part, so the script
samples per speaker (default 100 clips/speaker), caches features to
`dsp_features_diverse.csv`, and can be re-run with `--limit 0` to scan all.

Usage:
    python mine_diverse.py \
        --labeled-csv dsp_features_fixed.csv \
        --pool diverse_full_linux.txt \
        --limit 30000 --workers 6 \
        --out-csv report/diverse_high_intensity_candidates.csv \
        --all-out report/diverse_high_intensity_all.csv
"""
import argparse
import os
import random
import re
from collections import Counter, defaultdict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier as HGBClassifier

from dsp_common import parse_corpus, parse_speaker, read_filelist
from dsp_features import TIER1, batch_extract, write_csv
from mine_high_intensity import _rank

LABELED_FILES = [
    'data/v0.5/train_filelist_linux.txt',
    'data/v0.5/val_filelist_linux.txt',
    'data/v0.5_fixed/train_filelist_linux.txt',
    'data/v0.5_fixed/val_filelist_linux.txt',
]


def read_pool(pool_path, corpora, limit, max_per_speaker, seed,
              labeled_files=LABELED_FILES, corpus_limits=None,
              extra_stems=None):
    """Read path|sid rows, filter to corpora, drop labeled stems, sample."""
    labeled_stems = set()
    for lp in labeled_files:
        if not os.path.isfile(lp):
            continue
        for path, _ in read_filelist(lp):
            labeled_stems.add(os.path.splitext(os.path.basename(path))[0])
    if extra_stems:
        labeled_stems.update(extra_stems)

    by_speaker = defaultdict(list)
    stats = Counter()
    with open(pool_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats['total'] += 1
            parts = line.rsplit('|', 1)
            path = parts[0].strip()
            sid = parts[1].strip() if len(parts) > 1 else ''
            corpus = parse_corpus(path)
            if corpus not in corpora:
                continue
            stats[f'corpus_{corpus}'] += 1
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem in labeled_stems:
                stats['labeled'] += 1
                continue
            if not os.path.exists(path):
                stats['missing'] += 1
                continue
            speaker = parse_speaker(path)
            by_speaker[(corpus, speaker)].append((path, sid, corpus, speaker))
            stats['eligible'] += 1

    rng = random.Random(seed)
    sampled = []
    for rows in by_speaker.values():
        rng.shuffle(rows)
        sampled.extend(rows[:max_per_speaker] if max_per_speaker else rows)

    if corpus_limits:
        by_corpus = defaultdict(list)
        for row in sampled:
            by_corpus[row[2]].append(row)
        sampled = []
        for corpus, rows in by_corpus.items():
            rng.shuffle(rows)
            sampled.extend(rows[:corpus_limits.get(corpus, len(rows))])
    rng.shuffle(sampled)
    if limit:
        sampled = sampled[:limit]

    print(f'pool: {stats["total"]} rows | '
          f'genshin={stats["corpus_genshin"]} mlp={stats["corpus_mlp"]} | '
          f'labeled={stats["labeled"]} missing={stats["missing"]} | '
          f'eligible={stats["eligible"]} | sampled={len(sampled)}')
    print(f'  sampled by corpus: '
          f'{dict(Counter(r[2] for r in sampled))}')
    print(f'  sampled speakers: {len({(r[2], r[3]) for r in sampled})}')
    return sampled


def train_classifiers(labeled_csv, corpora=('genshin', 'mlp', 'marefair'),
                      out_path=None):
    """Train clf6/clf7 on the fixed labeled cache (no VCTK/other)."""
    df = pd.read_csv(labeled_csv)
    df = df[df['corpus'].isin(corpora)].reset_index(drop=True)
    missing = [c for c in TIER1 if c not in df.columns]
    if missing:
        raise ValueError(f'{labeled_csv} missing tier-1 columns: {missing}')
    X = df[TIER1].to_numpy(dtype=float)
    y = df['label'].to_numpy(dtype=float)
    models = {}
    for name, threshold in (('clf6', 6), ('clf7', 7)):
        target = (y >= threshold).astype(int)
        model = HGBClassifier(
            max_iter=300, learning_rate=0.05, max_depth=3,
            l2_regularization=1.0, random_state=0, class_weight='balanced')
        model.fit(X, target)
        models[name] = model
        print(f'{name}: trained on {len(df)} clips, '
              f'{int(target.sum())} positives')
    if out_path:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        joblib.dump({'features': TIER1, **models}, out_path)
        print(f'wrote {out_path}')
    return models, df


def extract_features(sample, cache_path, workers, refresh=False):
    """Extract (or load) tier-1 features for the sampled pool."""
    sid_map = {r[0]: r[1] for r in sample}
    if cache_path and os.path.exists(cache_path) and not refresh:
        df = pd.read_csv(cache_path)
        keep = set(sid_map)
        before = len(df)
        df = df[df['path'].isin(keep)].reset_index(drop=True)
        print(f'loaded cache {cache_path}: {before} rows -> {len(df)} '
              f'after excluding newly labeled / out-of-sample clips')
    else:
        items = [(r[0], 0, 'diverse') for r in sample]
        rows = batch_extract(items, tier=1, workers=workers, verbose=True)
        df = pd.DataFrame(rows)
        print(f'extracted {len(df)} / {len(sample)} clips')
        if cache_path:
            write_csv(df.to_dict('records'), cache_path)
            print(f'wrote {cache_path}')
    if 'sid' not in df.columns:
        df['sid'] = df['path'].map(sid_map)
    return df


def score_and_select(df, models, model_path, top, max_per_speaker):
    """Score the pool, rank, and return a diverse listening queue."""
    X = df[TIER1].to_numpy(dtype=float)
    df['clf6'] = models['clf6'].predict_proba(X)[:, 1]
    df['clf7'] = models['clf7'].predict_proba(X)[:, 1]
    # Genshin is ~11 dB louder than MLP; rank loudness within corpus so the
    # queue is not dominated by the louder recording condition.
    rms_rank = df.groupby('corpus')['rms_dbfs'].rank(pct=True).to_numpy()
    df['score'] = (_rank(df['clf6'].to_numpy())
                   + _rank(df['clf7'].to_numpy())
                   + rms_rank)

    if model_path and os.path.exists(model_path):
        from dsp_intensity import DSPIntensityModel
        dsp = DSPIntensityModel.load(model_path)
        df['dsp_pred'] = dsp.predict(df)
    else:
        df['dsp_pred'] = np.nan

    df['reason'] = np.where(
        df['clf7'].to_numpy() >= 0.7, 'A_clf7',
        np.where(df['clf6'].to_numpy() >= 0.8, 'B_clf6', 'C_top'))
    df = df.sort_values('score', ascending=False).reset_index(drop=True)

    picked = []
    per_speaker = Counter()
    for _, row in df.iterrows():
        if max_per_speaker and per_speaker[row['speaker']] >= max_per_speaker:
            continue
        picked.append(row)
        per_speaker[row['speaker']] += 1
        if top and len(picked) >= top:
            break

    cand = pd.DataFrame(picked).reset_index(drop=True)
    cand['current_label'] = np.nan
    cand['labels'] = '[]'
    cand['suggested'] = np.nan
    cand['new_label'] = ''
    cand['notes'] = ''
    cols = ['path', 'current_label', 'labels', 'suggested', 'sid', 'speaker',
            'corpus', 'rms_dbfs', 'clf6', 'clf7', 'dsp_pred', 'score',
            'reason', 'new_label', 'notes']
    cand = cand[cols]
    print(f'{len(cand)} candidates from {len(df)} scored clips')
    print(f'  reasons {cand["reason"].value_counts().to_dict()}')
    print(f'  corpora {cand["corpus"].value_counts().to_dict()}')
    print(f'  speakers {cand["speaker"].nunique()}')
    return cand, df


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--labeled-csv', type=str, default='dsp_features_fixed.csv')
    parser.add_argument('--pool', type=str, default='diverse_full_linux.txt')
    parser.add_argument('--corpora', nargs='+', default=['genshin', 'mlp'])
    parser.add_argument('--limit', type=int, default=30000,
                        help='sample size after per-speaker cap (0 = all)')
    parser.add_argument('--max-per-speaker', type=int, default=100,
                        help='max pool clips per speaker before --limit')
    parser.add_argument('--corpus-limits', type=str, default=None,
                        help='per-corpus quota, e.g. "genshin:18000,mlp:12000"')
    parser.add_argument('--top', type=int, default=200,
                        help='number of candidates in the listening queue')
    parser.add_argument('--max-per-speaker-final', type=int, default=3,
                        help='diversity cap in the final queue (0 = no cap)')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--cache', type=str, default='dsp_features_diverse.csv')
    parser.add_argument('--refresh-cache', action='store_true')
    parser.add_argument('--model', type=str,
                        default='checkpoints/dsp_intensity/tier1_fixed.pkl')
    parser.add_argument('--classifiers-out', type=str,
                        default='checkpoints/dsp_intensity/high_clf_fixed.pkl')
    parser.add_argument('--out-csv', type=str,
                        default='report/diverse_high_intensity_candidates.csv')
    parser.add_argument('--all-out', type=str,
                        default='report/diverse_high_intensity_all.csv')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    models, _ = train_classifiers(args.labeled_csv, args.corpora + ['marefair'],
                                  args.classifiers_out)
    corpus_limits = None
    if args.corpus_limits:
        corpus_limits = {
            k.strip(): int(v)
            for k, v in (part.split(':') for part in
                         args.corpus_limits.split(','))
        }
    extra_stems = set()
    if args.labeled_csv and os.path.exists(args.labeled_csv):
        lab = pd.read_csv(args.labeled_csv, usecols=['path'])
        extra_stems = {os.path.splitext(os.path.basename(p))[0]
                       for p in lab['path'].dropna()}
    sample = read_pool(args.pool, args.corpora, args.limit,
                       args.max_per_speaker, args.seed,
                       corpus_limits=corpus_limits,
                       extra_stems=extra_stems)
    if not sample:
        raise SystemExit('no unlabeled clips found')

    df = extract_features(sample, args.cache, args.workers,
                          refresh=args.refresh_cache)
    cand, scored = score_and_select(df, models, args.model, args.top,
                                    args.max_per_speaker_final)

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    cand.to_csv(args.out_csv, index=False)
    scored.to_csv(args.all_out, index=False)
    print(f'\nwrote {args.out_csv}')
    print(f'wrote {args.all_out}')
    print('\ntop candidates:')
    print(cand[['path', 'sid', 'speaker', 'corpus', 'clf6', 'clf7',
                'dsp_pred', 'reason']].head(12).to_string(index=False))


if __name__ == '__main__':
    main()

