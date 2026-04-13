"""
Paso 1: Extracción de datos de Supabase.
Extrae calls + analysis + campaigns, hace JOIN y guarda CSVs en data/.
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def fetch_all_rows(table, select='*', page_size=1000):
    """Pagina automáticamente si hay más de 1000 rows."""
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    all_rows = []
    offset = 0
    while True:
        response = supabase.table(table).select(select).range(offset, offset + page_size - 1).execute()
        rows = response.data
        if not rows:
            break
        all_rows.extend(rows)
        if len(rows) < page_size:
            break
        offset += page_size
    return all_rows


def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: SUPABASE_URL y SUPABASE_KEY son requeridos en .env")
        sys.exit(1)

    os.makedirs(DATA_DIR, exist_ok=True)

    print("Extrayendo datos de Supabase...")

    # Extraer tablas
    print("  -> calls...")
    calls = pd.DataFrame(fetch_all_rows('calls'))
    print(f"     {len(calls)} rows")

    print("  -> analysis...")
    analysis = pd.DataFrame(fetch_all_rows('analysis'))
    print(f"     {len(analysis)} rows")

    print("  -> campaigns...")
    campaigns = pd.DataFrame(fetch_all_rows('campaigns'))
    print(f"     {len(campaigns)} rows")

    # Guardar CSVs individuales
    calls.to_csv(os.path.join(DATA_DIR, 'calls.csv'), index=False)
    analysis.to_csv(os.path.join(DATA_DIR, 'analysis.csv'), index=False)
    campaigns.to_csv(os.path.join(DATA_DIR, 'campaigns.csv'), index=False)

    # JOIN: calls + analysis + campaigns
    dataset = calls.merge(analysis, on='call_id', how='left', suffixes=('', '_analysis'))
    if 'campaign_id' in dataset.columns and not campaigns.empty:
        campaigns_renamed = campaigns.rename(columns={'id': 'campaign_id'})
        dataset = dataset.merge(campaigns_renamed, on='campaign_id', how='left', suffixes=('', '_campaign'))

    dataset.to_csv(os.path.join(DATA_DIR, 'dataset_completo.csv'), index=False)

    # Resumen
    print("\n=== RESUMEN ===")
    print(f"Total llamadas: {len(dataset)}")

    if 'call_outcome' in dataset.columns:
        print(f"\nLabels unicos ({dataset['call_outcome'].nunique()}):")
        print(dataset['call_outcome'].value_counts().to_string())

    if 'outcome_confidence' in dataset.columns:
        print(f"\nDistribucion de outcome_confidence:")
        print(dataset['outcome_confidence'].describe().to_string())

    print(f"\nArchivos guardados en {DATA_DIR}/")


if __name__ == '__main__':
    main()
