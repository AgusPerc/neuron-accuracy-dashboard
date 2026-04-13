"""
Paso 5: Genera reporte final en Markdown con todos los resultados.
"""

import os
import sys
from datetime import datetime
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_lines = []

    def add(text=''):
        report_lines.append(text)

    add("# Accuracy Report — Neuron AI")
    add(f"**Generado:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    add()

    # --- Resumen ejecutivo ---
    add("## Resumen Ejecutivo")
    add()

    # Dataset info
    dataset_path = os.path.join(DATA_DIR, 'dataset_completo.csv')
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        add("## Dataset")
        add(f"- Total llamadas: {len(df)}")
        if 'call_outcome' in df.columns:
            add(f"- Labels únicos: {df['call_outcome'].nunique()}")
        if 'campaign_id' in df.columns:
            add(f"- Campañas: {df['campaign_id'].nunique()}")
        add()

        # Distribución de labels
        if 'call_outcome' in df.columns:
            add("### Distribución de Labels")
            add("| Label | Count | % |")
            add("|-------|-------|---|")
            counts = df['call_outcome'].value_counts()
            for label, count in counts.items():
                pct = count / len(df) * 100
                add(f"| {label} | {count} | {pct:.1f}% |")
            add()

    # --- Métricas por labeler ---
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics_summary.csv')
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)
        add("## Métricas por Labeler")
        add("| Labeler | F1 Macro | F1 Weighted | Precision | Recall | Predicciones |")
        add("|---------|----------|-------------|-----------|--------|-------------|")
        for _, row in metrics.iterrows():
            f1m = row['f1_macro']
            status = "excellent" if f1m > 0.9 else ("good" if f1m > 0.8 else "needs improvement")
            add(f"| {row['labeler']} | {f1m:.4f} | {row['f1_weighted']:.4f} | "
                f"{row['precision_macro']:.4f} | {row['recall_macro']:.4f} | "
                f"{row['valid_predictions']} | <!-- {status} -->")
        add()

        # Mejor labeler
        best = metrics.loc[metrics['f1_macro'].idxmax()]
        add(f"**Mejor labeler (F1 Macro):** {best['labeler']} ({best['f1_macro']:.4f})")
        add()

    # --- Agreement ---
    kappa_path = os.path.join(OUTPUT_DIR, 'kappa_agreement.csv')
    if os.path.exists(kappa_path):
        kappa = pd.read_csv(kappa_path)
        add("## Inter-Annotator Agreement (Cohen's Kappa)")
        add("| Labeler A | Labeler B | Kappa |")
        add("|-----------|-----------|-------|")
        for _, row in kappa.iterrows():
            k = row['kappa']
            add(f"| {row['labeler_a']} | {row['labeler_b']} | {k:.4f} |")
        add()
        add("Interpretación: >0.8 excelente, 0.6-0.8 bueno, 0.4-0.6 moderado, <0.4 pobre")
        add()

    # --- Confusion matrices (referencias a imágenes) ---
    cm_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('confusion_matrix_') and f.endswith('.png')]
    if cm_files:
        add("## Confusion Matrices")
        for f in sorted(cm_files):
            labeler = f.replace('confusion_matrix_', '').replace('.png', '')
            add(f"### {labeler}")
            add(f"![Confusion Matrix - {labeler}]({f})")
            add()

    # --- Gráfica F1 ---
    if os.path.exists(os.path.join(OUTPUT_DIR, 'f1_comparison.png')):
        add("## Comparación F1")
        add("![F1 Comparison](f1_comparison.png)")
        add()

    # --- Speech Precision ---
    speech_path = os.path.join(DATA_DIR, 'speech_precision.csv')
    if os.path.exists(speech_path):
        sp = pd.read_csv(speech_path)
        add("## Speech Precision")
        add(f"- Promedio adherencia: {sp['adherence_pct'].mean():.1f}%")
        add(f"- Min: {sp['adherence_pct'].min():.1f}%")
        add(f"- Max: {sp['adherence_pct'].max():.1f}%")
        add(f"- Llamadas evaluadas: {len(sp)}")
        add()
        if 'agent_id' in sp.columns:
            add("### Por Agente")
            add("| Agent ID | Adherencia Promedio | Llamadas |")
            add("|----------|-------------------|----------|")
            grouped = sp.groupby('agent_id')['adherence_pct'].agg(['mean', 'count'])
            for agent_id, row in grouped.iterrows():
                add(f"| {agent_id} | {row['mean']:.1f}% | {int(row['count'])} |")
            add()
    else:
        add("## Speech Precision")
        add("*Pendiente — requiere scripts de agentes en `scripts_agentes/`*")
        add()

    # --- Ambiguous calls ---
    ambiguous_path = os.path.join(DATA_DIR, 'ambiguous_calls.csv')
    if os.path.exists(ambiguous_path):
        amb = pd.read_csv(ambiguous_path)
        add("## Llamadas Ambiguas (requieren revisión manual)")
        add(f"- Total: {len(amb)}")
        add(f"- Archivo: `data/ambiguous_calls.csv`")
        add()

    # --- Thresholds ---
    add("## Thresholds de Referencia")
    add("| Métrica | Aceptable | Mejorar |")
    add("|---------|-----------|---------|")
    add("| FNR | <2.5% | >2.5% |")
    add("| FPR | <1% | >1% |")
    add("| F1 | >0.90 excelente, >0.80 bueno | <0.80 |")
    add()

    # --- Próximos pasos ---
    add("## Próximos Pasos")
    add("1. Revisar llamadas ambiguas manualmente")
    add("2. Ajustar prompts de labelers con bajo F1")
    add("3. Correr con dataset completo si sample muestra resultados prometedores")
    add("4. Implementar Speech Precision con scripts de agentes")
    add("5. Considerar weighted vote o Dawid-Skene para ground truth más robusto")
    add()

    # Guardar
    report = '\n'.join(report_lines)
    report_path = os.path.join(OUTPUT_DIR, 'accuracy_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"Reporte generado: {report_path}")
    print(f"Longitud: {len(report_lines)} líneas")


if __name__ == '__main__':
    main()
