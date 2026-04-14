"""
Paso 3: Confusion Matrix y métricas de accuracy.
Genera matrices de confusión, métricas por labeler y agreement entre pares.

Por default:
  - Usa Claude Opus 4.6 como ground truth (juez)
  - Excluye llamadas con label no_answer
  - Genera JSON con matrices (para dashboard HTML nativo) + PNGs (backup)

Flags opcionales:
  --judge {claude,majority}     Fuente del ground truth (default: claude)
  --exclude-labels LABELS...    Labels a excluir del analisis (default: no_answer)
"""

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, precision_score, recall_score, cohen_kappa_score,
)
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def majority_vote(row, labeler_cols):
    """Ground truth por majority vote. Empate = 'ambiguous'."""
    votes = [row[col] for col in labeler_cols if pd.notna(row.get(col)) and row[col] not in ('parse_error', 'api_error')]
    if not votes:
        return 'no_votes'
    counter = Counter(votes)
    top = counter.most_common(2)
    if len(top) > 1 and top[0][1] == top[1][1]:
        return 'ambiguous'
    return top[0][0]


def plot_confusion_matrix(y_true, y_pred, labeler_name, labels, output_dir, judge_name):
    """Genera y guarda heatmap de confusion matrix (PNG backup)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(max(8, len(labels)), max(6, len(labels) * 0.8)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {labeler_name} vs {judge_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    path = os.path.join(output_dir, f'confusion_matrix_{labeler_name}_vs_{judge_name}.png')
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def compute_fnr_fpr(y_true, y_pred, labels):
    """Calcula FNR y FPR por label."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    results = {}
    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        results[label] = {'FNR': fnr, 'FPR': fpr}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--judge', default='claude', choices=['claude', 'majority'],
                        help='Fuente del ground truth (default: claude)')
    parser.add_argument('--exclude-labels', nargs='*', default=['no_answer'],
                        help='Labels a excluir (default: no_answer). Pasa vacio para no excluir nada.')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = os.path.join(DATA_DIR, 'labels_comparison.csv')
    if not os.path.exists(input_path):
        print("ERROR: Primero ejecuta 02_run_labelers.py")
        sys.exit(1)

    df = pd.read_csv(input_path)

    # Detectar columnas de labels
    label_cols = [c for c in df.columns if c.startswith('label_')]
    print(f"Labelers encontrados: {label_cols}")

    # Ground truth segun juez seleccionado
    if args.judge == 'claude':
        if 'label_claude' not in df.columns:
            print("ERROR: columna label_claude no encontrada en labels_comparison.csv")
            sys.exit(1)
        df['ground_truth'] = df['label_claude']
        judge_name = 'claude'
        # Al usar Claude como juez, comparamos todos los otros labelers contra el
        candidate_cols = [c for c in label_cols if c != 'label_claude']
        print(f"Ground truth: Claude Opus 4.6 (juez)")
    else:
        df['ground_truth'] = df.apply(lambda row: majority_vote(row, label_cols), axis=1)
        judge_name = 'majority'
        candidate_cols = label_cols
        print(f"Ground truth: majority vote de {len(label_cols)} labelers")

    # Filtrar invalidos
    invalid_truth = ['ambiguous', 'no_votes', 'parse_error', 'api_error']
    valid = df[~df['ground_truth'].isin(invalid_truth)].copy()
    valid = valid[valid['ground_truth'].notna()]

    # Excluir labels seleccionados
    n_before_exclude = len(valid)
    for excluded in args.exclude_labels:
        valid = valid[valid['ground_truth'] != excluded]
    n_excluded = n_before_exclude - len(valid)

    print(f"Llamadas con ground truth valido: {len(valid)} (excluidas {n_excluded} por filtro de labels: {args.exclude_labels})")

    if args.judge == 'majority':
        ambiguous_count = (df['ground_truth'] == 'ambiguous').sum()
        print(f"Llamadas ambiguas (majority vote): {ambiguous_count}")
        if ambiguous_count > 0:
            ambiguous_df = df[df['ground_truth'] == 'ambiguous']
            ambiguous_df.to_csv(os.path.join(DATA_DIR, 'ambiguous_calls.csv'), index=False)
            print(f"  -> Guardadas en data/ambiguous_calls.csv")

    labels = sorted(valid['ground_truth'].unique().tolist())
    print(f"Labels activos: {labels}")

    # JSON output container (para dashboard HTML nativo)
    json_output = {
        'judge': judge_name,
        'excluded_labels': args.exclude_labels,
        'total_calls': len(valid),
        'labels': labels,
        'matrices': {},
    }

    # Métricas por labeler
    all_metrics = []
    for col in candidate_cols:
        labeler_name = col.replace('label_', '')
        y_true = valid['ground_truth']
        y_pred = valid[col]

        # Filtrar predicciones inválidas del labeler
        mask = ~y_pred.isin(['parse_error', 'api_error']) & y_pred.notna()
        # Tambien excluir predicciones del labeler que sean labels excluidos (ej: no_answer)
        for excluded in args.exclude_labels:
            mask = mask & (y_pred != excluded)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            print(f"\n{labeler_name}: sin predicciones validas, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"LABELER: {labeler_name} vs {judge_name}")
        print(f"{'='*50}")
        print(f"Predicciones validas: {len(y_pred_clean)}/{len(valid)}")

        # Classification report
        report = classification_report(y_true_clean, y_pred_clean, labels=labels, zero_division=0)
        print(report)

        # FNR / FPR
        fnr_fpr = compute_fnr_fpr(y_true_clean, y_pred_clean, labels)
        print("FNR / FPR por label:")
        for label, vals in fnr_fpr.items():
            fnr_status = "OK" if vals['FNR'] < 0.025 else "MEJORAR"
            fpr_status = "OK" if vals['FPR'] < 0.01 else "MEJORAR"
            print(f"  {label}: FNR={vals['FNR']:.4f} [{fnr_status}] | FPR={vals['FPR']:.4f} [{fpr_status}]")

        # Confusion matrix PNG (backup)
        plot_confusion_matrix(y_true_clean, y_pred_clean, labeler_name, labels, OUTPUT_DIR, judge_name)

        # Confusion matrix para JSON (dashboard)
        cm = confusion_matrix(y_true_clean, y_pred_clean, labels=labels)
        correct = int(np.diag(cm).sum())
        total = int(cm.sum())
        accuracy = float(correct / total) if total > 0 else 0.0

        # Per-label precision/recall/f1 para tooltip enriquecido
        per_label = {}
        for i, lbl in enumerate(labels):
            tp = int(cm[i, i])
            fn = int(cm[i, :].sum() - tp)
            fp = int(cm[:, i].sum() - tp)
            precision_lbl = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall_lbl = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1_lbl = float(2 * precision_lbl * recall_lbl / (precision_lbl + recall_lbl)) if (precision_lbl + recall_lbl) > 0 else 0.0
            per_label[lbl] = {
                'precision': precision_lbl,
                'recall': recall_lbl,
                'f1': f1_lbl,
                'support': int(cm[i, :].sum()),
            }

        json_output['matrices'][f'{labeler_name}_vs_{judge_name}'] = {
            'labeler': labeler_name,
            'judge': judge_name,
            'labels': labels,
            'matrix': cm.tolist(),
            'row_sums': cm.sum(axis=1).astype(int).tolist(),
            'col_sums': cm.sum(axis=0).astype(int).tolist(),
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
            'per_label': per_label,
        }

        # Métricas globales
        f1_macro = f1_score(y_true_clean, y_pred_clean, labels=labels, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true_clean, y_pred_clean, labels=labels, average='weighted', zero_division=0)
        precision_macro = precision_score(y_true_clean, y_pred_clean, labels=labels, average='macro', zero_division=0)
        recall_macro = recall_score(y_true_clean, y_pred_clean, labels=labels, average='macro', zero_division=0)

        all_metrics.append({
            'labeler': labeler_name,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'valid_predictions': len(y_pred_clean),
        })

    # Guardar JSON
    json_path = os.path.join(OUTPUT_DIR, 'confusion_matrices.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print(f"\nJSON guardado en {json_path}")

    # Cohen's Kappa entre pares
    print(f"\n{'='*50}")
    print("COHEN'S KAPPA (inter-annotator agreement)")
    print(f"{'='*50}")

    kappa_results = []
    for i, col_a in enumerate(label_cols):
        for col_b in label_cols[i+1:]:
            mask = (valid[col_a].notna() & valid[col_b].notna() &
                    ~valid[col_a].isin(['parse_error', 'api_error']) &
                    ~valid[col_b].isin(['parse_error', 'api_error']))
            if mask.sum() > 0:
                kappa = cohen_kappa_score(valid.loc[mask, col_a], valid.loc[mask, col_b])
                name_a = col_a.replace('label_', '')
                name_b = col_b.replace('label_', '')
                kappa_results.append({'labeler_a': name_a, 'labeler_b': name_b, 'kappa': kappa})
                print(f"  {name_a} vs {name_b}: {kappa:.4f}")

    # Guardar métricas
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics_summary.csv'), index=False)
    print(f"\nMetricas guardadas en outputs/metrics_summary.csv")

    if kappa_results:
        kappa_df = pd.DataFrame(kappa_results)
        kappa_df.to_csv(os.path.join(OUTPUT_DIR, 'kappa_agreement.csv'), index=False)

    # Gráfica comparativa de F1
    if all_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(all_metrics))
        ax.bar(x, [m['f1_macro'] for m in all_metrics], alpha=0.7, label='F1 Macro')
        ax.bar(x, [m['f1_weighted'] for m in all_metrics], alpha=0.5, label='F1 Weighted')
        ax.set_xticks(x)
        ax.set_xticklabels([m['labeler'] for m in all_metrics], rotation=45)
        ax.set_ylabel('Score')
        ax.set_title(f'F1 Score por Labeler (vs {judge_name}, excl. {args.exclude_labels})')
        ax.legend()
        ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='Threshold excelente')
        ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Threshold bueno')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'f1_comparison.png'), dpi=150)
        plt.close()
        print("Grafica F1 guardada en outputs/f1_comparison.png")


if __name__ == '__main__':
    main()
