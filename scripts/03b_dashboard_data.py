"""
Paso 3b: Genera dashboard_data.json con todo lo que el dashboard necesita.

Aplica filtros:
  1. Excluye llamadas del 2026-03-26 (bug de plataforma: Retell clasifico
     auto como no_answer llamadas con contenido real, campaign_id=NaN)
  2. Excluye labels SDR (qualified/unqualified) — son de prospeccion, no de cobranza
  3. (REMOVIDO) Antes excluia Claude=no_answer. Ahora se reincluyen para honestidad
     de metricas (ver memoria Tema B).

Nota: las llamadas sin transcript quedan fuera en el pipeline (02_run_labelers.py),
no aparecen en labels_comparison.csv.

Usa Claude Opus 4.6 como ground truth.

Output:
  outputs/dashboard_data.json — unificado, consumido por dashboard.html
"""

import json
import os
import sys
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, cohen_kappa_score,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

BUG_DATE = '2026-03-26'  # Bug de plataforma: contaminacion no_answer
SDR_LABELS = {'qualified', 'unqualified'}  # Labels de SDR, no de cobranza


def fleiss_kappa(ratings_matrix):
    """Compute Fleiss' kappa for multi-rater agreement.
    ratings_matrix: 2D array [n_items, n_raters] with category labels.
    """
    n_items, n_raters = ratings_matrix.shape
    categories = sorted(set(ratings_matrix.flatten()))
    n_categories = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    # Build count matrix [n_items, n_categories]
    counts = np.zeros((n_items, n_categories))
    for i in range(n_items):
        for j in range(n_raters):
            cat = ratings_matrix[i, j]
            counts[i, cat_to_idx[cat]] += 1

    # Per-item agreement
    p_i = (counts ** 2).sum(axis=1) - n_raters
    p_i = p_i / (n_raters * (n_raters - 1))
    P_bar = p_i.mean()

    # Category marginals
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    P_e = (p_j ** 2).sum()

    if P_e == 1:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def main():
    # Cargar datos
    calls_path = os.path.join(DATA_DIR, 'calls.csv')
    labels_path = os.path.join(DATA_DIR, 'labels_comparison.csv')

    if not os.path.exists(calls_path) or not os.path.exists(labels_path):
        print("ERROR: Faltan CSVs. Ejecuta 01_extract_data.py y 02_run_labelers.py primero.")
        sys.exit(1)

    calls = pd.read_csv(calls_path, usecols=['call_id', 'start_timestamp', 'call_duration'])
    labels = pd.read_csv(labels_path)

    # Merge por call_id
    df = labels.merge(calls, on='call_id', how='left')
    print(f"Total llamadas en labels_comparison: {len(labels)}")
    print(f"Llamadas con match en calls.csv: {df['start_timestamp'].notna().sum()}")

    # Extraer fecha (YYYY-MM-DD)
    df['date'] = df['start_timestamp'].str[:10]

    # === FILTRO 1: Bug 2026-03-26 ===
    # El filtro puede haber sido aplicado upstream en 02_run_labelers.py
    # (via --exclude-date). Contamos las originales del calls.csv para
    # reportar un numero honesto en el dashboard.
    n_before = len(df)
    df = df[df['date'] != BUG_DATE].copy()
    n_bug_here = n_before - len(df)
    try:
        calls_all = pd.read_csv(calls_path, usecols=['start_timestamp'])
        calls_all['_d'] = calls_all['start_timestamp'].astype(str).str[:10]
        n_bug_original = int((calls_all['_d'] == BUG_DATE).sum())
    except Exception:
        n_bug_original = n_bug_here
    n_bug = max(n_bug_here, n_bug_original)
    print(f"Excluidas por bug {BUG_DATE}: {n_bug} (en este run: {n_bug_here}, upstream: {n_bug_original - n_bug_here})")

    # Ground truth = Claude Opus 4.6
    df['ground_truth'] = df['label_claude']

    # === FILTRO 2: Labels SDR (qualified/unqualified) — no son de cobranza ===
    # Excluir si CUALQUIER labeler o ground_truth tiene esas labels
    n_before = len(df)
    sdr_mask = df['ground_truth'].isin(SDR_LABELS)
    for labeler_col in ['label_retell', 'label_llama', 'label_claude']:
        if labeler_col in df.columns:
            sdr_mask |= df[labeler_col].isin(SDR_LABELS)
    df = df[~sdr_mask].copy()
    n_sdr = n_before - len(df)
    print(f"Excluidas por labels SDR (qualified/unqualified): {n_sdr}")

    # Filtrar invalidos del ground truth
    n_before = len(df)
    df = df[df['ground_truth'].notna() & ~df['ground_truth'].isin(['parse_error', 'api_error'])]
    n_invalid = n_before - len(df)
    if n_invalid:
        print(f"Excluidas por ground_truth invalido: {n_invalid}")
    print(f"Dataset final: {len(df)} llamadas")

    labels_list = sorted(df['ground_truth'].unique().tolist())
    print(f"Labels activos: {labels_list}")

    # Date range
    date_min = df['date'].min()
    date_max = df['date'].max()

    # === COMPUTE METRICS ===
    labelers_of_interest = ['retell', 'llama']  # gemini tiene parse_errors, skipped

    results = {
        'meta': {
            'total_calls': int(len(df)),
            'date_range': {'start': date_min, 'end': date_max},
            'filters': [
                f'Excluidas llamadas del {BUG_DATE} (bug de plataforma: {n_bug} calls)',
                f'Excluidas labels SDR qualified/unqualified ({n_sdr} calls)',
                'Incluye todos los labels incluyendo no_answer (honestidad metricas)',
            ],
            'judge': 'Claude Opus 4.6',
            'labels': labels_list,
            'excluded_bug_date_count': int(n_bug),
            'excluded_sdr_labels_count': int(n_sdr),
        },
        'kpis': {},
        'resumen_by_label': [],
        'per_label_metrics': {},
        'confusion_matrices': {},
        'errores': [],
        'kappa': {},
    }

    # Accuracy global por labeler (vs Claude)
    for labeler in labelers_of_interest:
        col = f'label_{labeler}'
        if col not in df.columns:
            continue
        mask = df[col].notna() & ~df[col].isin(['parse_error', 'api_error'])
        sub = df[mask]
        correct = (sub[col] == sub['ground_truth']).sum()
        total = len(sub)
        accuracy = float(correct / total) if total > 0 else 0.0
        results['kpis'][f'{labeler}_accuracy'] = {
            'accuracy_pct': round(accuracy * 100, 1),
            'correct': int(correct),
            'total': int(total),
        }

    # Fleiss kappa (multi-rater: retell + llama + claude)
    kappa_cols = ['label_retell', 'label_llama', 'label_claude']
    kappa_mask = df[kappa_cols].notna().all(axis=1)
    for c in kappa_cols:
        kappa_mask &= ~df[c].isin(['parse_error', 'api_error'])
    kappa_df = df.loc[kappa_mask, kappa_cols]
    if len(kappa_df) > 0:
        ratings = kappa_df.values
        fk = fleiss_kappa(ratings)
        results['kpis']['fleiss_kappa'] = {
            'value': round(fk, 3),
            'n_items': int(len(kappa_df)),
            'n_raters': 3,
            'interpretation': (
                'casi perfecto' if fk >= 0.81 else
                'substancial' if fk >= 0.61 else
                'moderado' if fk >= 0.41 else
                'aceptable' if fk >= 0.21 else 'leve'
            ),
        }

    # Pairwise Cohen's kappa
    pairs = [('retell', 'llama'), ('retell', 'claude'), ('llama', 'claude')]
    for a, b in pairs:
        col_a, col_b = f'label_{a}', f'label_{b}'
        mask = df[col_a].notna() & df[col_b].notna()
        mask &= ~df[col_a].isin(['parse_error', 'api_error'])
        mask &= ~df[col_b].isin(['parse_error', 'api_error'])
        if mask.sum() > 0:
            ck = cohen_kappa_score(df.loc[mask, col_a], df.loc[mask, col_b])
            results['kappa'][f'{a}_vs_{b}'] = round(float(ck), 3)

    # Resumen bar chart: Retell accuracy per label (support desc)
    retell_col = 'label_retell'
    for lbl in labels_list:
        mask_gt = df['ground_truth'] == lbl
        support = int(mask_gt.sum())
        if support == 0:
            continue
        sub = df[mask_gt]
        valid = sub[retell_col].notna() & ~sub[retell_col].isin(['parse_error', 'api_error'])
        if valid.sum() == 0:
            acc = 0.0
        else:
            acc = float((sub.loc[valid, retell_col] == lbl).sum() / valid.sum())
        results['resumen_by_label'].append({
            'label': lbl,
            'support': support,
            'accuracy': round(acc * 100, 1),
        })
    # Sort by support desc
    results['resumen_by_label'].sort(key=lambda r: -r['support'])

    # Per-label precision/recall for Retell and Llama (vs Claude)
    for labeler in labelers_of_interest:
        col = f'label_{labeler}'
        if col not in df.columns:
            continue
        mask = df[col].notna() & ~df[col].isin(['parse_error', 'api_error'])
        y_true = df.loc[mask, 'ground_truth']
        y_pred = df.loc[mask, col]

        prec_arr = precision_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
        rec_arr = recall_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)
        f1_arr = f1_score(y_true, y_pred, labels=labels_list, average=None, zero_division=0)

        results['per_label_metrics'][labeler] = {
            'labels': labels_list,
            'precision': [round(float(p), 3) for p in prec_arr],
            'recall':    [round(float(r), 3) for r in rec_arr],
            'f1':        [round(float(f), 3) for f in f1_arr],
        }

    # Confusion matrices vs Claude
    for labeler in labelers_of_interest:
        col = f'label_{labeler}'
        if col not in df.columns:
            continue
        mask = df[col].notna() & ~df[col].isin(['parse_error', 'api_error'])
        y_true = df.loc[mask, 'ground_truth']
        y_pred = df.loc[mask, col]

        cm = confusion_matrix(y_true, y_pred, labels=labels_list)
        correct = int(np.diag(cm).sum())
        total = int(cm.sum())
        accuracy = float(correct / total) if total > 0 else 0.0

        per_label = {}
        for i, lbl in enumerate(labels_list):
            tp = int(cm[i, i])
            fn = int(cm[i, :].sum() - tp)
            fp = int(cm[:, i].sum() - tp)
            prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            per_label[lbl] = {
                'precision': round(prec, 3),
                'recall': round(rec, 3),
                'f1': round(f1, 3),
                'support': int(cm[i, :].sum()),
            }

        results['confusion_matrices'][f'{labeler}_vs_claude'] = {
            'labeler': labeler,
            'judge': 'claude',
            'labels': labels_list,
            'matrix': cm.tolist(),
            'row_sums': cm.sum(axis=1).astype(int).tolist(),
            'col_sums': cm.sum(axis=0).astype(int).tolist(),
            'total': total,
            'correct': correct,
            'accuracy': round(accuracy, 4),
            'per_label': per_label,
        }

    # Errores table: calls where Retell != Claude
    err_mask = (df['label_retell'] != df['label_claude']) & df['label_retell'].notna() & df['label_claude'].notna()
    err_mask &= ~df['label_retell'].isin(['parse_error', 'api_error'])
    errores_df = df[err_mask].sort_values('date', ascending=False)
    for _, row in errores_df.iterrows():
        # Format date as MM/DD
        date_mmdd = row['date'][5:].replace('-', '/') if isinstance(row['date'], str) else ''
        results['errores'].append({
            'date': date_mmdd,
            'call_id': row['call_id'],
            'retell': row['label_retell'],
            'claude': row['label_claude'],
            'llama': row['label_llama'] if pd.notna(row['label_llama']) else '—',
            'dur': int(row['call_duration']) if pd.notna(row['call_duration']) else 0,
        })

    # Save
    out_path = os.path.join(OUTPUT_DIR, 'dashboard_data.json')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Dashboard data guardado en {out_path}")
    print(f"{'='*60}")
    print(f"KPIs:")
    for k, v in results['kpis'].items():
        print(f"  {k}: {v}")
    print(f"\nErrores (Retell vs Claude): {len(results['errores'])}")


if __name__ == '__main__':
    main()
