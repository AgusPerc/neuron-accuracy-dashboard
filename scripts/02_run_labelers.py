"""
Paso 2: Correr labelers de evaluación (Gemini, Llama, Claude Sonnet 4).
Clasifica cada transcripción y guarda resultados en data/labels_comparison.csv.
"""

import argparse
import os
import sys
import json
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'prompts')

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Precios aproximados por 1M tokens (input/output)
PRICING = {
    'gemini': {'input': 0.075, 'output': 0.30},       # Gemini 2.5 Flash
    'llama': {'input': 0.0, 'output': 0.0},            # Groq free tier
    'claude_sonnet4': {'input': 3.0, 'output': 15.0},  # Claude Sonnet 4
}

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds, doubles each retry


def load_prompt(labels):
    with open(os.path.join(PROMPTS_DIR, 'labeler_prompt.txt'), 'r') as f:
        template = f.read()
    return template.replace('{labels}', '\n'.join(f'- {l}' for l in labels))


def parse_json_response(text):
    """Extrae JSON de la respuesta del LLM, tolerando markdown fences."""
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1] if '\n' in text else text[3:]
        text = text.rsplit('```', 1)[0]
    try:
        result = json.loads(text.strip())
        return result.get('label', 'error'), result.get('confidence', 0.0)
    except json.JSONDecodeError:
        return 'parse_error', 0.0


def call_gemini(prompt, transcript):
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
    full_prompt = prompt.replace('{transcript}', transcript)
    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(full_prompt)
            return parse_json_response(response.text)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                print(f"    Gemini error: {e}")
                return 'api_error', 0.0


def call_llama(prompt, transcript):
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)
    full_prompt = prompt.replace('{transcript}', transcript)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model='meta-llama/llama-4-scout-17b-16e-instruct',
                messages=[{'role': 'user', 'content': full_prompt}],
                temperature=0.0,
                max_tokens=200,
            )
            return parse_json_response(response.choices[0].message.content)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                print(f"    Llama error: {e}")
                return 'api_error', 0.0


def call_claude_sonnet4(prompt, transcript):
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    full_prompt = prompt.replace('{transcript}', transcript)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.messages.create(
                model='claude-sonnet-4-20250514',
                max_tokens=200,
                messages=[{'role': 'user', 'content': full_prompt}],
            )
            return parse_json_response(response.content[0].text)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
            else:
                print(f"    Claude Sonnet 4 error: {e}")
                return 'api_error', 0.0


def estimate_cost(df, avg_tokens_per_transcript=1500):
    """Estima costo antes de correr."""
    n = len(df)
    total_input_tokens = n * avg_tokens_per_transcript
    total_output_tokens = n * 50  # ~50 tokens por respuesta JSON

    print("\n=== ESTIMACION DE COSTO ===")
    total = 0.0
    for name, prices in PRICING.items():
        cost_in = (total_input_tokens / 1_000_000) * prices['input']
        cost_out = (total_output_tokens / 1_000_000) * prices['output']
        cost = cost_in + cost_out
        total += cost
        print(f"  {name}: ${cost:.4f}")
    print(f"  TOTAL estimado: ${total:.4f}")
    print(f"  ({n} llamadas, ~{avg_tokens_per_transcript} tokens/transcripcion)")
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yes', action='store_true', help='Skip interactive confirmation')
    parser.add_argument('--all', action='store_true', help='Process ALL eligible calls (not sample of 100)')
    parser.add_argument('--limit', type=int, default=100, help='Sample size when --all is not set (default 100)')
    parser.add_argument('--skip-gemini', action='store_true', help='Skip Gemini labeler')
    parser.add_argument('--skip-llama', action='store_true', help='Skip Llama labeler')
    parser.add_argument('--skip-claude', action='store_true', help='Skip Claude Sonnet 4 labeler')
    parser.add_argument('--exclude-date', action='append', default=[], help='Date to exclude (YYYY-MM-DD, can repeat)')
    args = parser.parse_args()

    dataset_path = os.path.join(DATA_DIR, 'dataset_completo.csv')
    if not os.path.exists(dataset_path):
        print("ERROR: Primero ejecuta 01_extract_data.py")
        sys.exit(1)

    df = pd.read_csv(dataset_path)

    # Filtrar: transcript no vacío + call completada
    df = df[df['transcript'].notna() & (df['transcript'].str.strip() != '')]
    if 'call_status' in df.columns:
        df = df[df['call_status'].str.lower().isin(['completed', 'ended'])]

    # Exclude specific dates (e.g., platform bugs)
    if args.exclude_date and 'start_timestamp' in df.columns:
        df = df.copy()
        df['_date'] = df['start_timestamp'].astype(str).str[:10]
        before = len(df)
        df = df[~df['_date'].isin(args.exclude_date)]
        df = df.drop(columns=['_date'])
        print(f"Excluidas por --exclude-date {args.exclude_date}: {before - len(df)}")

    print(f"Llamadas validas para clasificar: {len(df)}")

    if len(df) == 0:
        print("No hay llamadas validas. Revisa los datos.")
        sys.exit(1)

    # Labels posibles (de los datos existentes)
    labels = df['call_outcome'].dropna().unique().tolist()
    print(f"Labels encontrados: {labels}")

    prompt = load_prompt(labels)

    # Sample: todas (--all) o las N mas recientes
    if 'start_timestamp' in df.columns:
        df = df.sort_values('start_timestamp', ascending=False, na_position='last')
    sample_size = len(df) if args.all else min(args.limit, len(df))
    df_sample = df.head(sample_size).copy()
    print(f"\nUsando sample de {sample_size} llamadas{' (TODAS)' if args.all else ' (mas recientes)'}")

    # Estimar costo
    estimate_cost(df_sample)
    if not args.yes:
        confirm = input("\nContinuar? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelado.")
            sys.exit(0)

    # Determinar qué labelers correr
    labelers = {}
    if GOOGLE_API_KEY and not args.skip_gemini:
        labelers['gemini'] = call_gemini
    else:
        reason = '--skip-gemini' if args.skip_gemini else 'GOOGLE_API_KEY no encontrada'
        print(f"WARN: skipping Gemini ({reason})")

    if GROQ_API_KEY and not args.skip_llama:
        labelers['llama'] = call_llama
    else:
        reason = '--skip-llama' if args.skip_llama else 'GROQ_API_KEY no encontrada'
        print(f"WARN: skipping Llama ({reason})")

    if ANTHROPIC_API_KEY and not args.skip_claude:
        labelers['claude_sonnet4'] = call_claude_sonnet4
    else:
        reason = '--skip-claude' if args.skip_claude else 'ANTHROPIC_API_KEY no encontrada'
        print(f"WARN: skipping Claude Sonnet 4 ({reason})")

    if not labelers:
        print("ERROR: No hay API keys configuradas para ningún labeler.")
        sys.exit(1)

    # Correr labelers
    results = []
    for idx, (_, row) in enumerate(df_sample.iterrows()):
        print(f"\n[{idx+1}/{sample_size}] call_id: {row['call_id']}")
        result = {
            'call_id': row['call_id'],
            'label_retell': row.get('call_outcome', ''),
            'confidence_retell': row.get('outcome_confidence', 0.0),
        }

        for name, func in labelers.items():
            print(f"  -> {name}...", end=' ')
            label, confidence = func(prompt, row['transcript'])
            result[f'label_{name}'] = label
            result[f'confidence_{name}'] = confidence
            print(f"{label} ({confidence:.2f})")

        results.append(result)

    # Guardar
    results_df = pd.DataFrame(results)
    output_path = os.path.join(DATA_DIR, 'labels_comparison.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResultados guardados en {output_path}")
    print(f"Total procesadas: {len(results_df)}")


if __name__ == '__main__':
    main()
