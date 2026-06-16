# Deployment Quick Guide — Vercel (frontend) + Render (backend)

This file summarizes the minimal, time-saving deployment steps I recommend.

1) Vercel — Frontend (fast, zero-config)

- Project root: set to `frontend` when importing the repo in Vercel.
- Framework Preset: Vite
- Build Command: `npm ci && npm run build` (or `pnpm install && pnpm build`)
- Output Directory: `dist`
- Environment Variable (Vercel → Project → Environment Variables):
  - `VITE_API_URL` = `https://<your-backend-url>`

Quick steps:

```bash
# 1. Push repo to GitHub (if not already)
# 2. In Vercel: New Project → Import Repo → Root Directory = frontend
# 3. Set Build Command and Output Directory as above
# 4. Add `VITE_API_URL` env var to point to your Render backend URL
# 5. Deploy
```

2) Render — Backend (recommended for FastAPI)

- Service Type: Web Service (Environment: "Python") OR Docker (if you prefer Docker builds)
- If using Python service (no Docker):
  - Build Command: `pip install -r requirements.txt`
  - Start Command: `bash render_start.sh`

- Environment / Secrets to set in Render (Environment tab):
  - `PINECONE_API_KEY` (if using Pinecone)
  - `GOOGLE_CREDENTIALS_JSON` — *full contents* of your Google service account JSON (will be written to `/tmp/google_credentials.json` by `render_start.sh`)
  - `OPENAI_API_KEY` or other model keys if required
  - `ALLOWED_ORIGINS` — comma-separated frontend origins (e.g., `https://your-frontend.vercel.app,https://localhost:3000`)

Notes:
- `render_start.sh` (included in repo) will write `GOOGLE_CREDENTIALS_JSON` to disk and set `GOOGLE_APPLICATION_CREDENTIALS` before launching the app.
- The FastAPI app entrypoint is `src.api:app`. If your app entry changes, update the `uvicorn` command accordingly.

3) CORS and frontend config

- Ensure `ALLOWED_ORIGINS` contains the Vercel URL (e.g., `https://your-app.vercel.app`) so the front-end can call the API.
- Set `VITE_API_URL` in Vercel to the Render service URL (e.g., `https://rc-gpt-backend.onrender.com`).

4) Quick smoke-test

```bash
# From local machine or using curl
curl https://<your-backend>/health
curl https://<your-frontend>/ (open in browser)
```

5) Single-provider quick alternative (both on Render)

- Frontend: Create a "Static Site" on Render, set the Build Command `npm ci && npm run build` and Publish Directory `frontend/dist`.
- Backend: Create a "Web Service" as above.

---

If you want, I can now:
- Add a tiny `vercel.json` for the `frontend/` folder, or
- Create a Render service template (YAML) for easy import.
