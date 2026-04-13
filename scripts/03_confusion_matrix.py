"""
Paso 3: Confusion Matrix y métricas de accuracy.
Genera matrices de confusión, métricas por labeler y agreement entre pares.
"""

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


def plot_confusion_matrix(y_true, y_pred, labeler_name, labels, output_dir):
    """Genera y guarda heatmap de confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(max(8, len(labels)), max(6, len(labels) * 0.8)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {labeler_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    path = os.path.join(output_dir, f'confusion_matrix_{labeler_name}.png')
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_path = os.path.join(DATA_DIR, 'labels_comparison.csv')
    if not os.path.exists(input_path):
        print("ERROR: Primero ejecuta 02_run_labelers.py")
        sys.exit(1)

    df = pd.read_csv(input_path)

    # Detectar columnas de labels
    label_cols = [c for c in df.columns if c.startswith('label_')]
    print(f"Labelers encontrados: {label_cols}")

    # Ground truth por majority vote
    df['ground_truth'] = df.apply(lambda row: majority_vote(row, label_cols), axis=1)

    # Filtrar ambiguos y errores
    valid = df[~df['ground_truth'].isin(['ambiguous', 'no_votes'])].copy()
    ambiguous_count = (df['ground_truth'] == 'ambiguous').sum()
    print(f"Llamadas con ground truth valido: {len(valid)}")
    print(f"Llamadas ambiguas (para revision manual): {ambiguous_count}")

    if ambiguous_count > 0:
        ambiguous_df = df[df['ground_truth'] == 'ambiguous']
        ambiguous_df.to_csv(os.path.join(DATA_DIR, 'ambiguous_calls.csv'), index=False)
        print(f"  -> Guardadas en data/ambiguous_calls.csv")

    labels = sorted(valid['ground_truth'].unique().tolist())
    print(f"Labels: {labels}")

    # Métricas por labeler
    all_metrics = []
    for col in label_cols:
        labeler_name = col.replace('label_', '')
        y_true = valid['ground_truth']
        y_pred = valid[col]

        # Filtrar predicciones inválidas
        mask = ~y_pred.isin(['parse_error', 'api_error']) & y_pred.notna()
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]

        if len(y_true_clean) == 0:
            print(f"\n{labeler_name}: sin predicciones validas, skipping")
            continue

        print(f"\n{'='*50}")
        print(f"LABELER: {labeler_name}")
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

        # Confusion matrix plot
        plot_confusion_matrix(y_true_clean, y_pred_clean, labeler_name, labels, OUTPUT_DIR)

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
        ax.set_title('F1 Score por Labeler')
        ax.legend()
        ax.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='Threshold excelente')
        ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.5, label='Threshold bueno')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'f1_comparison.png'), dpi=150)
        plt.close()
        print("Grafica F1 guardada en outputs/f1_comparison.png")


if __name__ == '__main__':
    main()
