# XAi-PROMPTSPRINT

# XAI-COMPASS

A governance layer for existing AI/ML models — makes their decisions explainable, multilingual, and audit-ready.

## What it does

Point XAI-COMPASS at an existing model (currently demoed on **AetherDx**, an InceptionV3 nail-image healthcare screener) and it wraps every prediction with:

- **Explain** — Grad-CAM heatmaps showing which parts of the image drove the decision
- **Translate** — plain-language explanations in English, Hindi, and Marathi
- **Verify** — a compliance checklist scoring evidence readiness for each decision
- **Audit** — downloadable, timestamped audit reports per decision

```
EXISTING MODEL → MODEL CONNECTOR → XAI-COMPASS → EXPLAIN → TRANSLATE → VERIFY → AUDIT
```

## Stack

Streamlit (UI) + FastAPI (model connector API) + TensorFlow/Keras + OpenCV (Grad-CAM).

## Running it

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Add your `.keras` / `.h5` model file to a `models/` folder.
3. Start the model connector API:
   ```bash
   uvicorn backend.api:app --reload --port 8000
   ```
4. In a separate terminal, launch the app:
   ```bash
   streamlit run app.py
   ```
5. Upload an image, hit **Analyze**, and review the explanation, translations, compliance evidence, and audit report.

## Note

This is a demonstration platform. AI outputs are not medical, legal, or regulatory certifications.
