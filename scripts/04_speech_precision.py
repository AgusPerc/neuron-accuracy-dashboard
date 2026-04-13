"""
Paso 4: Speech Precision — Evalúa adherencia del agente al script esperado.
Usa Claude para comparar transcripción vs script y medir desviaciones.
"""

import os
import sys
import json
import time
import pandas as pd
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'prompts')

ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

SPEECH_EVAL_PROMPT = """You are evaluating how well an AI voice agent followed its expected script during a debt collection call.

## Expected Script
{script}

## Actual Transcript
{transcript}

## Instructions
Compare the transcript against the expected script and evaluate:
1. Overall adherence percentage (0-100%)
2. Which script sections were followed correctly
3. Which sections were skipped or deviated from
4. Type of each deviation (skipped, modified, improvised, off-topic)

Respond ONLY with valid JSON:
```json
{{
  "adherence_pct": <0-100>,
  "sections_followed": ["section1", "section2"],
  "deviations": [
    {{
      "section": "section name",
      "type": "skipped|modified|improvised|off-topic",
      "detail": "brief description"
    }}
  ],
  "overall_assessment": "brief assessment"
}}
```"""


def evaluate_speech(transcript, script):
    """Evalúa adherencia de una transcripción al script."""
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = SPEECH_EVAL_PROMPT.replace('{script}', script).replace('{transcript}', transcript)

    for attempt in range(3):
        try:
            response = client.messages.create(
                model='claude-sonnet-4-20250514',
                max_tokens=500,
                messages=[{'role': 'user', 'content': prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith('```'):
                text = text.split('\n', 1)[1] if '\n' in text else text[3:]
                text = text.rsplit('```', 1)[0]
            return json.loads(text.strip())
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (2 ** attempt))
            else:
                print(f"    Error: {e}")
                return None


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Buscar scripts de agentes
    scripts_dir = os.path.join(os.path.dirname(__file__), '..', 'scripts_agentes')
    if not os.path.exists(scripts_dir):
        print("=" * 50)
        print("SPEECH PRECISION - PENDIENTE")
        print("=" * 50)
        print()
        print("No se encontró la carpeta 'scripts_agentes/'.")
        print("Para ejecutar este análisis, crea la carpeta y agrega")
        print("los scripts esperados de cada agente en formato .txt")
        print()
        print("Estructura esperada:")
        print("  neuron-accuracy-metrics/")
        print("    scripts_agentes/")
        print("      agent_<agent_id>.txt    # Script del agente")
        print()
        print("Cada archivo debe contener el speech/script que el")
        print("agente debería seguir durante las llamadas.")
        print()
        os.makedirs(scripts_dir, exist_ok=True)

        # Crear template
        template_path = os.path.join(scripts_dir, 'TEMPLATE.txt')
        with open(template_path, 'w') as f:
            f.write("# Script del Agente: [nombre]\n")
            f.write("# Agent ID: [id]\n\n")
            f.write("## Saludo\n[script de saludo]\n\n")
            f.write("## Identificación\n[script de identificación del deudor]\n\n")
            f.write("## Motivo de llamada\n[script del motivo]\n\n")
            f.write("## Negociación\n[script de negociación]\n\n")
            f.write("## Cierre\n[script de cierre]\n")
        print(f"Template creado en: {template_path}")
        return

    if not ANTHROPIC_API_KEY:
        print("ERROR: ANTHROPIC_API_KEY requerida para Speech Precision")
        sys.exit(1)

    # Cargar dataset
    dataset_path = os.path.join(DATA_DIR, 'dataset_completo.csv')
    if not os.path.exists(dataset_path):
        print("ERROR: Primero ejecuta 01_extract_data.py")
        sys.exit(1)

    df = pd.read_csv(dataset_path)
    df = df[df['transcript'].notna() & (df['transcript'].str.strip() != '')]

    # Cargar scripts disponibles
    scripts = {}
    for fname in os.listdir(scripts_dir):
        if fname.endswith('.txt') and fname != 'TEMPLATE.txt':
            agent_id = fname.replace('.txt', '').replace('agent_', '')
            with open(os.path.join(scripts_dir, fname), 'r') as f:
                scripts[agent_id] = f.read()

    if not scripts:
        print("No hay scripts de agentes en scripts_agentes/ (solo el template).")
        print("Agrega archivos .txt con los scripts para cada agente.")
        return

    print(f"Scripts de agentes cargados: {list(scripts.keys())}")
    print(f"Llamadas a evaluar: {len(df)}")

    # Sample
    sample_size = min(50, len(df))
    df_sample = df.head(sample_size).copy()

    cost_est = sample_size * 2000 * 3.0 / 1_000_000  # ~input tokens * price
    print(f"\nCosto estimado: ${cost_est:.2f}")
    confirm = input("Continuar? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelado.")
        return

    results = []
    for idx, (_, row) in enumerate(df_sample.iterrows()):
        agent_id = str(row.get('agent_id', ''))
        script = scripts.get(agent_id)
        if not script:
            # Usar el primer script disponible como fallback
            script = list(scripts.values())[0]

        print(f"[{idx+1}/{sample_size}] call_id: {row['call_id']}...", end=' ')
        result = evaluate_speech(row['transcript'], script)
        if result:
            results.append({
                'call_id': row['call_id'],
                'agent_id': agent_id,
                'adherence_pct': result.get('adherence_pct', 0),
                'num_deviations': len(result.get('deviations', [])),
                'deviation_types': ','.join(d['type'] for d in result.get('deviations', [])),
                'assessment': result.get('overall_assessment', ''),
            })
            print(f"{result.get('adherence_pct', 0)}%")
        else:
            print("error")

    # Guardar
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(DATA_DIR, 'speech_precision.csv'), index=False)

    # Resumen
    print(f"\n=== RESUMEN SPEECH PRECISION ===")
    print(f"Promedio adherencia: {results_df['adherence_pct'].mean():.1f}%")
    print(f"Min: {results_df['adherence_pct'].min():.1f}%")
    print(f"Max: {results_df['adherence_pct'].max():.1f}%")

    if 'agent_id' in results_df.columns:
        print(f"\nPor agente:")
        print(results_df.groupby('agent_id')['adherence_pct'].agg(['mean', 'count']).to_string())

    print(f"\nResultados guardados en data/speech_precision.csv")


if __name__ == '__main__':
    main()
