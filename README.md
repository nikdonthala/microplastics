# MicroScan AI — Microplastic Screening

Educational prototype for **image-based screening and morphological classification
of suspected microplastics** in water samples.

> ⚠️ **Science disclaimer** — image analysis alone does not confirm polymer identity.
> Confirmatory analysis (FTIR or Raman spectroscopy) is required. All metrics shown by
> this app come from real computation on the data you provide; nothing is hard-coded.

## Stack

| Layer     | Technology                                                        |
| --------- | ----------------------------------------------------------------- |
| Frontend  | React 19 + TypeScript + Vite, glassmorphism UI (green/white)      |
| Charts    | ECharts                                                           |
| Backend   | FastAPI (Python), OpenCV, scikit-learn (Random Forest)            |
| Deploy    | Vercel — Vite static build + FastAPI as a serverless function     |
| Tests     | pytest (35 tests)                                                 |

## Repository layout

```
├── api/
│   ├── index.py            # Vercel serverless entrypoint (exports `app`)
│   └── requirements.txt    # Python deps for the function
├── backend/
│   ├── app/                # FastAPI app (routes, services, schemas, utils)
│   ├── data/               # training/ and sample/ data dirs
│   ├── training/           # train_model.py, evaluate_model.py, demo data gen
│   └── tests/              # pytest suite
├── frontend/
│   └── src/                # React UI (App.tsx, charts.tsx, api.ts, styles.css)
├── vercel.json             # Vercel build + routing config
└── .github/workflows/ci.yml
```

## Run locally

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload        # http://localhost:8000 (docs at /docs)
```

### Frontend

```bash
cd frontend
npm install
npm run dev                          # http://localhost:5173 (proxies /api to :8000)
```

### Train the demo model (optional — enables classification)

```bash
cd backend
python -m training.generate_demo_data --dataset          # synthetic demo CSV
python -m training.train_model --dataset demo_particles.csv
```

Without a trained model the app runs in **demo mode**: detection + morphology
features work, classification returns `unclassified`.

## API

| Endpoint               | Method | Purpose                                     |
| ---------------------- | ------ | ------------------------------------------- |
| `/api/health`          | GET    | Liveness probe                              |
| `/api/analyze`         | POST   | Full pipeline (multipart image upload)      |
| `/api/model/status`    | GET    | Whether a trained model is available        |
| `/api/model/metrics`   | GET    | Evaluation metrics from the training run    |
| `/api/train`           | POST   | Train from a CSV in `backend/data/training/`|

## Tests

```bash
cd backend
python -m pytest -q
```

## Deployment (Vercel)

1. Push this repository to GitHub (private repo `microplastics`).
2. Import it at [vercel.com/new](https://vercel.com/new) — `vercel.json` supplies
   the build command (`frontend` Vite build) and routes `/api/*` (plus the SPA
   catch-all) to `api/index.py`.
3. Every push to `main` auto-deploys production; PRs get preview deployments.

The function runs on Fluid compute; heavy deps (OpenCV, scikit-learn) are
supported (function size limit is 5 GB on Fluid).

## CI

GitHub Actions runs the backend pytest suite and the frontend TypeScript build
on every push/PR — broken commits never reach Vercel auto-deploy.
