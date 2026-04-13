# Neuron Accuracy Metrics

## Que es
Benchmark de labelers (LLMs) que clasifican llamadas de cobranza en Neuron AI.
Mide Category Accuracy (labels) y Speech Precision (adherencia a script).

## Stack
Python + pandas + sklearn + seaborn. Datos en Supabase (PostgreSQL).

## Labelers evaluados
- **Actuales** (via Retell AI): GPT-4, Claude Haiku/Sonnet
- **Nuevos**: Gemini 2.5 Flash, Llama 4 Scout (Groq), Claude Sonnet 4

## Reglas
- API keys en `.env`, NUNCA hardcodeadas
- Siempre empezar con sample de 50-100 llamadas
- Paginar Supabase con `.range()` si >1000 rows
- Calcular e imprimir costo estimado ANTES de correr labelers
- CSVs en `data/` (gitignored), reportes en `outputs/`
- Si falta una API key, skip ese labeler y continuar

## Thresholds
- FNR <2.5% | FPR <1% | F1 >0.90 excelente, 0.80-0.90 bueno, <0.80 mejorar

## Scripts (ejecutar en orden)
1. `01_extract_data.py` — Extrae datos de Supabase
2. `02_run_labelers.py` — Corre los 3 labelers nuevos
3. `03_confusion_matrix.py` — Confusion matrix + metricas
4. `04_speech_precision.py` — Adherencia a script
5. `05_generate_report.py` — Reporte final en markdown
