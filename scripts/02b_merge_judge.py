"""
Paso 2b: Consolida los resultados del juez Opus (8 batch CSVs) con las
predicciones de los labelers (labels_comparison.csv).

Inputs:
  data/judge_batches/batch_01_result.csv ... batch_08_result.csv
  data/labels_comparison.csv (output de 02_run_labelers.py — tiene label_retell, label_llama)

Output:
  data/labels_comparison.csv (in-place: agrega/sobreescribe label_claude + confidence_claude)
  data/judge_reasoning.csv (archivo separado con call_id + reasoning del juez, para auditoria)
"""

import glob
import os
import sys
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
BATCH_DIR = os.path.join(DATA_DIR, 'judge_batches')
LABELS_CSV = os.path.join(DATA_DIR, 'labels_comparison.csv')
REASONING_CSV = os.path.join(DATA_DIR, 'judge_reasoning.csv')


def main():
    batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, 'batch_*_result.csv')))
    if not batch_files:
        print("ERROR: No hay batch_*_result.csv en", BATCH_DIR)
        sys.exit(1)

    print(f"Consolidando {len(batch_files)} batches del juez:")
    dfs = []
    for f in batch_files:
        # Some batches wrote unescaped commas in reasoning.
        # on_bad_lines='skip' drops broken rows; then we re-parse manually.
        try:
            d = pd.read_csv(f)
        except pd.errors.ParserError:
            # Fallback: parse with python engine + quoting tolerant
            d = pd.read_csv(f, engine='python', on_bad_lines='warn', quoting=1)
            # If still missing rows, do a manual line-by-line parse
            if len(d) < 29:
                print(f"  WARN: {os.path.basename(f)} has broken rows, parsing manually")
                rows = []
                with open(f, 'r', encoding='utf-8') as fh:
                    header = fh.readline().strip().split(',')
                    for line in fh:
                        parts = line.rstrip('\n').split(',')
                        if len(parts) < 4:
                            continue
                        # First 3 cols are single-valued; everything else is reasoning (join)
                        rows.append({
                            'call_id': parts[0],
                            'label_claude': parts[1],
                            'confidence_claude': parts[2],
                            'reasoning': ','.join(parts[3:]).replace('"', ''),
                        })
                d = pd.DataFrame(rows)
        print(f"  {os.path.basename(f)}: {len(d)} rows")
        dfs.append(d)
    judge = pd.concat(dfs, ignore_index=True)
    print(f"Total juez: {len(judge)} rows")

    # Sanity: no dup call_ids
    dups = judge['call_id'].duplicated().sum()
    if dups:
        print(f"WARN: {dups} call_ids duplicados en batches del juez, quedando con el primero")
        judge = judge.drop_duplicates(subset='call_id', keep='first')

    # Labels distrib from judge
    print(f"\nDistribucion del juez Opus (ground truth):")
    print(judge['label_claude'].value_counts().to_string())

    # Save reasoning separately for auditability
    judge[['call_id', 'label_claude', 'confidence_claude', 'reasoning']].to_csv(
        REASONING_CSV, index=False
    )
    print(f"\nReasoning guardado: {REASONING_CSV}")

    # Merge into labels_comparison.csv
    if not os.path.exists(LABELS_CSV):
        print(f"ERROR: {LABELS_CSV} no existe. Corre 02_run_labelers.py primero.")
        sys.exit(1)

    labels = pd.read_csv(LABELS_CSV)
    print(f"\nlabels_comparison.csv actual: {len(labels)} rows")
    print(f"Columnas: {list(labels.columns)}")

    # Drop old claude cols if present
    for col in ['label_claude', 'confidence_claude', 'label_claude_sonnet4', 'confidence_claude_sonnet4']:
        if col in labels.columns:
            labels = labels.drop(columns=[col])

    # Merge by call_id
    merged = labels.merge(
        judge[['call_id', 'label_claude', 'confidence_claude']],
        on='call_id',
        how='left',
    )
    matched = merged['label_claude'].notna().sum()
    print(f"\nMatched: {matched}/{len(merged)} rows tienen juicio del juez")

    if matched == 0:
        print("ERROR: Ningun match. Los call_ids del juez no coinciden con labels_comparison.csv.")
        sys.exit(1)

    merged.to_csv(LABELS_CSV, index=False)
    print(f"\nGuardado: {LABELS_CSV}")
    print(f"Columnas finales: {list(merged.columns)}")


if __name__ == '__main__':
    main()
