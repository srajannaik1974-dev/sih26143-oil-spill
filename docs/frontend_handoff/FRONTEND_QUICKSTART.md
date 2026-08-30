# Frontend Quickstart Guide (Member 5)

This document provides the fastest path for Member 5 to connect the React + Vite frontend to the Member 6 FastAPI backend.

---

## 1. Backend Server URL
- **Base URL**: `http://127.0.0.1:8000/api`
- **Swagger Documentation**: `http://127.0.0.1:8000/docs`
- **OpenAPI Schema**: `http://127.0.0.1:8000/api/openapi.json`

---

## 2. How to Start the Backend
From the repository root on branch `integration-backend`:
```bash
# 1. Activate Python virtual environment (if used)
# 2. Run backend via Uvicorn
python -m uvicorn backend.main:app --reload --port 8000
```

Verify backend health:
```bash
curl http://127.0.0.1:8000/api/health
```
Expected output:
```json
{"status":"ok","service":"SIH 2026 - Oil Spill & Vessel Attribution API","version":"1.0.0"}
```

---

## 3. How to Start the React Frontend
In a separate terminal inside `frontend/`:
```bash
cd frontend
npm install
npm run dev
```

The frontend dev server will launch at `http://localhost:5173`.

---

## 4. Setting the Base URL Environment Variable
Do NOT hardcode `http://127.0.0.1:8000` across `App.jsx`. Use `import.meta.env.VITE_API_BASE_URL`:

Create `frontend/.env.development`:
```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

In React code:
```javascript
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";
```

---

## 5. Summary of Active API Endpoints

1. `GET /api/health` — Service health check
2. `POST /api/spill/detect` — Satellite detection & polygon
3. `POST /api/spill/backtrack` — Ocean drift backtracking & origin
4. `POST /api/ais/candidates` — Query candidate vessels near origin
5. `POST /api/vessels/rank` — Rank suspect vessels with attribution scores
