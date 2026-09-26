# MicroScan AI — Microplastic Screening

**Live demo:** https://microplastic-stuff.vercel.app
**[Github repo](https://github.com/nikdonthala/microplastics)**

## What is this project about?

Microplastics are tiny plastic particles that end up in water and are hard to
spot with the naked eye. This project is a web app that helps **screen water
samples for suspected microplastics** from a simple photo, and gives a first
guess at what kind of particles are there.

## What are we doing?

1. **Upload** a photo of a water sample (or use a demo image).
2. **Detect** particles in the image using OpenCV (color/edge-based detection).
3. **Measure** each particle's shape — size, roundness, aspect ratio, texture, etc.
4. **Classify** particles into categories (fragment, fiber, pellet) with a
   Random Forest model trained with scikit-learn.
5. **Visualize** the results as interactive charts (ECharts) in a React UI.

## Stack

- **Frontend:** React + TypeScript + Vite
- **Backend:** FastAPI (Python) + OpenCV + scikit-learn
- **Deploy:** Vercel (static frontend + serverless API)

> ⚠️ This is an educational prototype — image analysis alone cannot confirm
> polymer identity. Confirmatory analysis (FTIR/Raman spectroscopy) is required.

## Setup & API Reference

### Run locally

#### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload        # http://localhost:8000 (docs at /docs)
```

#### Frontend

```bash
cd frontend
npm install
npm run dev                          # http://localhost:5173 (proxies /api to :8000)
```

#### Train the demo model (optional — enables classification)

```bash
cd backend
python -m training.generate_demo_data --dataset          # synthetic demo CSV
python -m training.train_model --dataset demo_particles.csv
```

Without a trained model the app runs in **demo mode**: detection + morphology
features work, classification returns `unclassified`.

### API endpoints

| Endpoint             | Method | Purpose                                      |
| -------------------- | ------ | -------------------------------------------- |
| `/api/health`        | GET    | Liveness probe                               |
| `/api/analyze`       | POST   | Full pipeline (multipart image upload)       |
| `/api/model/status`  | GET    | Whether a trained model is available         |
| `/api/model/metrics` | GET    | Evaluation metrics from the training run     |
| `/api/train`         | POST   | Train from a CSV in `backend/data/training/` |

### Tests

```bash
cd backend
python -m pytest -q
```

### Deployment (Vercel)

1. Push this repository to GitHub (private repo `microplastics`).
2. Import it at [vercel.com/new](https://vercel.com/new) — `vercel.json` supplies
   the build command (`frontend` Vite build) and routes `/api/*` (plus the SPA
   catch-all) to `api/index.py`.
3. Every push to `main` auto-deploys production; PRs get preview deployments.

The function runs on Fluid compute; heavy deps (OpenCV, scikit-learn) are
supported (function size limit is 5 GB on Fluid).

### CI

GitHub Actions runs the backend pytest suite and the frontend TypeScript build
on every push/PR — broken commits never reach Vercel auto-deploy.
