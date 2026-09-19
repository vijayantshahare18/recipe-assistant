# AI-Powered Recipe & Nutrition Assistant

**Your Ingredients, Our Intelligence, Better Meals!**

RecipeMind is a mobile-first React + FastAPI application that turns available ingredients and dietary preferences into recipe suggestions, nutrition-aware meal plans, substitutions, grocery lists and cooking Q&A.

## Architecture

```text
React + Vite + Tailwind
        |
        | JSON/HTTP
        v
FastAPI
  |-- ingredient normalization
  |-- server-side diet/allergy/calorie safety filters
  |-- vector retrieval (FAISS when installed; NumPy fallback offline)
  |-- optional OpenAI recipe embeddings + RAG personalization
  |-- USDA FoodData Central adapter with local nutrition fallback
  |-- seven-day meal planning + grocery derivation
  v
Curated recipe dataset
```

## Six-screen flow

1. **Home / fridge input** — add and remove ingredients.
2. **Preferences** — diet, goal, cuisine, calorie limit and allergies.
3. **Recipe suggestions** — personalized cards with nutrition and filters.
4. **Recipe details** — ingredients, instructions, nutrition, favorites and sharing.
5. **Substitutions** — diet/allergy-aware replacement suggestions.
6. **Weekly meal plan** — Monday–Sunday breakfast/lunch/dinner with daily calories.

Supporting tools include a grocery list and cooking/nutrition chat.

## Tech stack

- Frontend: React, Vite, Tailwind CSS, Lucide React
- Backend: Python, FastAPI, Pydantic
- Retrieval: FAISS `IndexFlatIP` when available; deterministic NumPy vector fallback for offline development
- LLM: OpenAI Responses API for optional recipe personalization and chat
- Embeddings: OpenAI `text-embedding-3-small` when configured; local ingredient vectors otherwise
- Nutrition: USDA FoodData Central adapter when `USDA_API_KEY` is configured; curated local nutrition values otherwise

## Folder structure

```text
recipe-assistant/
├── backend/
│   ├── app/main.py
│   ├── app/nutrition.py
│   └── requirements.txt
├── data/recipes.json
├── frontend/
│   ├── src/main.jsx
│   ├── src/index.css
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
├── tests/test_api.py
├── .env.example
└── README.md
```

## Environment setup

Copy `.env.example` to `.env`.

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
USDA_API_KEY=
VITE_API_URL=http://localhost:8000
```

API keys remain backend-only. Only `VITE_API_URL` is intended for the frontend.

## Backend startup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r backend/requirements.txt
python -m uvicorn backend.app.main:app --reload --port 8000
```

Health check: `GET /health`

## Frontend startup

```bash
cd frontend
npm install
npm run dev
```

Set `VITE_API_URL` if FastAPI runs somewhere other than `http://localhost:8000`.

## API

- `POST /recipes/suggest` — normalize ingredients, perform vector retrieval, enforce server-side safety filters and optionally personalize results with OpenAI.
- `GET /recipes/{id}` — recipe details plus nutrition adapter status.
- `POST /substitutions` — diet/allergy-aware substitutions.
- `POST /meal-plan/generate` — seven-day plan with calorie targeting, rotation and ingredient reuse.
- `POST /grocery-list` — derive missing ingredients from selected recipe IDs or current ingredient-based matches.
- `POST /chat` — cooking/nutrition Q&A with OpenAI or an explicit prototype fallback.
- `GET /health` — service, retrieval, embedding and nutrition integration status.

## RAG and nutrition behavior

Recipe ingredient vectors are indexed and queried through normalized inner-product similarity. FAISS is used when installed; NumPy provides the deterministic offline fallback.

When `OPENAI_API_KEY` is configured, recipe text is embedded with the configured OpenAI embedding model and the retrieved recipes are supplied to the personalization layer. Server-side allergy, diet and calorie filtering happens before LLM personalization.

When `USDA_API_KEY` is configured, recipe detail performs a server-side FoodData Central lookup for a representative ingredient. The current curated dataset does not contain ingredient quantities, so its recipe-level calorie/macronutrient totals remain the local curated values rather than pretending to calculate a full USDA recipe total. Without a USDA key or if the lookup fails, the documented local adapter is used.

## Safety

Allergy, diet and calorie restrictions are enforced by FastAPI. Substitution candidates are independently checked against the same constraints. The LLM is not trusted as the sole safety filter, and no secret is shipped to the browser.

## Testing and verification

Run:

```bash
PYTHONPATH=. pytest -q
python -m compileall -q backend
```

Current source verification:

- **14/14 pytest tests passed**.
- Required API endpoints were smoke-tested successfully.
- HTTP 422 validation and HTTP 404 recipe-not-found behavior were verified.
- Allergy, diet and calorie filtering were verified.
- Seven-day meal-plan generation and three meal slots per day were verified.
- Grocery-list generation was verified with recipe IDs.
- Chat fallback behavior was verified without an API key.
- No API keys were found hard-coded in the source.
- FastAPI CORS for the Vite development origin was verified.
- Frontend static audit confirmed the six required screens and all seven required API routes are wired in `main.jsx`.
- `npm install --no-audit --no-fund` was attempted again and timed out; an offline package-lock attempt also confirmed the required packages are not cached in this environment. Therefore a Vite browser build is **not claimed as verified**.

## Current prototype limitations

- The bundled dataset contains 20 curated recipes rather than a production-scale corpus.
- Full recipe-level USDA nutrition calculation requires ingredient quantities, serving sizes and a larger normalized food mapping; the current adapter deliberately reports its lookup scope.
- OpenAI personalization, embeddings and chat require an API key.
- FAISS requires installation in the target environment; NumPy fallback remains available.
- Fridge image scanning is labeled as an optional unfinished feature and is not part of the six required screens.
