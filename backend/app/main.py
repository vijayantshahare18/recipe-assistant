import json, os, re
from pathlib import Path
from typing import List, Optional
import numpy as np
import httpx
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from .nutrition import enrich_recipe

load_dotenv()
ROOT = Path(__file__).resolve().parents[2]
RECIPES = json.loads((ROOT / 'data/recipes.json').read_text())
STOP = {'and','the','with','of','in','a','an','fresh','to'}
ALIASES = {'onions':'onion','tomatoes':'tomato','eggs':'egg','chickpeas':'chickpea','potatoes':'potato','carrots':'carrot','peppers':'pepper','beans':'bean'}
ALLERGIES = {'Nuts','Dairy','Gluten'}
DIETS = {'Vegetarian','Vegan','Non-Veg'}
CUISINES = sorted({r['cuisine'] for r in RECIPES})


def normalize(x: str) -> str:
    s = re.sub(r'[^a-z0-9 ]', ' ', str(x).lower()).strip()
    return ' '.join(ALIASES.get(w, w) for w in s.split() if w not in STOP)


def tokens(items):
    return {normalize(x) for x in items if normalize(x)}


def ingredient_blockers(r):
    return set(r.get('allergens', []))

VOCAB = sorted({w for r in RECIPES for w in tokens(r['ingredients'])})

# Retrieval uses real vector embeddings when OpenAI embeddings are configured.
# In offline/prototype mode it uses deterministic TF-style binary ingredient vectors,
# keeping the same vector-search contract without pretending an external model ran.
class NumpyIndex:
    def __init__(self, dim): self.vectors = np.empty((0, dim), dtype='float32')
    def add(self, arr): self.vectors = np.vstack([self.vectors, arr])
    def search(self, q, k):
        sims = np.dot(self.vectors, q[0])
        order = np.argsort(-sims)[:k]
        return sims[order].reshape(1,-1), order.reshape(1,-1)

index = faiss.IndexFlatIP(len(VOCAB)) if HAS_FAISS else NumpyIndex(len(VOCAB))

def local_vec(ings):
    v = np.array([1.0 if w in tokens(ings) else 0.0 for w in VOCAB], dtype='float32')
    n = np.linalg.norm(v)
    return v / n if n else v

def embed_texts(texts):
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.embeddings.create(model=os.getenv('OPENAI_EMBEDDING_MODEL','text-embedding-3-small'), input=texts)
        arr = np.array([x.embedding for x in resp.data], dtype='float32')
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-12)
    except Exception:
        return None

RECIPE_TEXTS = [f"{r['name']}. Cuisine: {r['cuisine']}. Ingredients: {', '.join(r['ingredients'])}." for r in RECIPES]
OPENAI_EMBEDDINGS = embed_texts(RECIPE_TEXTS)
if OPENAI_EMBEDDINGS is not None:
    index = faiss.IndexFlatIP(OPENAI_EMBEDDINGS.shape[1]) if HAS_FAISS else NumpyIndex(OPENAI_EMBEDDINGS.shape[1])
    index.add(OPENAI_EMBEDDINGS)
    EMBEDDING_MODE = 'openai-recipe-embeddings'
else:
    M = np.stack([local_vec(r['ingredients']) for r in RECIPES]).astype('float32')
    index.add(M)
    EMBEDDING_MODE = 'local-ingredient-vector-prototype'

def vec(ings):
    if OPENAI_EMBEDDINGS is not None:
        arr = embed_texts([f"User ingredients: {', '.join(ings)}"] )
        if arr is not None:
            return arr[0]
    return local_vec(ings)

app = FastAPI(title='RecipeMind API', version='0.4.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:5173',
        'http://127.0.0.1:5173',
        'https://recipe-assistant-1.onrender.com'
    ],
    allow_methods=['*'],
    allow_headers=['*']
)

class Prefs(BaseModel):
    ingredients: List[str] = Field(min_length=1)
    diet: str = 'Vegetarian'
    cuisine: str = ''
    calorie_limit: int = Field(default=600, ge=100, le=5000)
    allergies: List[str] = []
    goal: str = 'Maintenance'

    @field_validator('ingredients')
    @classmethod
    def valid_ingredients(cls, v):
        clean = [x.strip() for x in v if str(x).strip()]
        if not clean: raise ValueError('At least one ingredient is required')
        return clean

    @field_validator('diet')
    @classmethod
    def valid_diet(cls, v):
        if v not in DIETS: raise ValueError('Invalid diet')
        return v

    @field_validator('allergies')
    @classmethod
    def valid_allergies(cls, v):
        bad = set(v) - ALLERGIES
        if bad: raise ValueError(f'Unsupported allergies: {sorted(bad)}')
        return v

class SubReq(BaseModel):
    missing_ingredients: List[str] = Field(min_length=1)
    diet: str = 'Vegetarian'
    allergies: List[str] = []

    @field_validator('diet')
    @classmethod
    def valid_diet(cls, v):
        if v not in DIETS: raise ValueError('Invalid diet')
        return v

    @field_validator('allergies')
    @classmethod
    def valid_allergies(cls, v):
        bad = set(v) - ALLERGIES
        if bad: raise ValueError(f'Unsupported allergies: {sorted(bad)}')
        return v

ALLERGY_BLOCK = {
    'Nuts': {'almond','cashew','peanut','walnut','almond milk'},
    'Dairy': {'milk','paneer','cheese','yogurt','butter','ghee'},
    'Gluten': {'wheat','bread','flour','wheat flour','semolina'}
}
SUBS = {
    'spinach':['kale','cabbage','fenugreek'],
    'milk':['oat milk','soy milk'],
    'paneer':['tofu'],
    'butter':['olive oil','coconut oil'],
    'wheat flour':['oat flour','rice flour'],
    'cheese':['nutritional yeast','tofu'],
    'yogurt':['coconut yogurt','soy yogurt'],
    'egg':['tofu','flax egg'],
    'chicken':['tofu','mushroom'],
}


def safe(r, p):
    if p.diet == 'Vegan' and r['diet'] != 'Vegan': return False
    if p.diet == 'Vegetarian' and r['diet'] == 'Non-Veg': return False
    if p.cuisine and r['cuisine'] != p.cuisine: return False
    if ingredient_blockers(r) & set(p.allergies): return False
    if r['calories'] > p.calorie_limit: return False
    return True


def public(r, have):
    d = dict(r)
    d['missing_ingredients'] = sorted(set(r['ingredients']) - tokens(have))
    return d


def retrieval_context(recipes):
    return '\n'.join(f"{r['id']} | {r['name']} | {r['calories']} kcal | {r['protein']}g protein | ingredients: {', '.join(r['ingredients'])}" for r in recipes)


def personalize_with_llm(recipes, p):
    key = os.getenv('OPENAI_API_KEY')
    if not key or not recipes:
        return recipes, 'prototype-rag-context-only'
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        context = retrieval_context(recipes[:6])
        prompt = f"""You are a recipe personalization layer. Using ONLY these retrieved recipes, return JSON with keys selected_ids and reason. User diet={p.diet}; allergies={p.allergies}; calorie_limit={p.calorie_limit}; goal={p.goal}; ingredients={p.ingredients}. Retrieved recipes:\n{context}"""
        resp = client.responses.create(model=os.getenv('OPENAI_MODEL','gpt-4o-mini'), input=prompt)
        raw = json.loads(resp.output_text)
        selected = {str(x) for x in raw.get('selected_ids', [])}
        ordered = [r for r in recipes if r['id'] in selected] or recipes
        reason = str(raw.get('reason','LLM personalization applied to retrieved recipes.'))
        for r in ordered: r['why_match'] = reason
        return ordered, 'openai-rag-personalization'
    except Exception:
        return recipes, 'prototype-rag-context-only (LLM unavailable)'

@app.get('/health')
def health():
    return {
        'status':'ok','recipes':len(RECIPES),
        'rag':'faiss' if HAS_FAISS else 'numpy-fallback (FAISS unavailable in current environment)',
        'embedding_provider':EMBEDDING_MODE,
        'nutrition':'usda-configured' if os.getenv('USDA_API_KEY') else 'local-prototype-adapter'
    }

@app.post('/recipes/suggest')
def suggest(p: Prefs):
    q = vec(p.ingredients)
    scores, ids = index.search(q.reshape(1,-1), len(RECIPES))
    candidates = []
    for score, i in zip(scores[0], ids[0]):
        if i < 0: continue
        r = RECIPES[int(i)]
        if safe(r, p):
            d = public(r, p.ingredients)
            d['match_score'] = round(float(score), 3)
            d['why_match'] = f"Matches {len(tokens(p.ingredients) & tokens(r['ingredients']))} available ingredients and your {p.diet} preference."
            candidates.append(d)
    candidates, mode = personalize_with_llm(candidates[:10], p)
    return {'recipes': candidates, 'normalized_ingredients': sorted(tokens(p.ingredients)), 'retrieval':'FAISS inner-product vector search' if HAS_FAISS else 'NumPy inner-product vector search fallback', 'personalization':mode, 'embedding_provider':EMBEDDING_MODE}

@app.get('/recipes/{rid}')
def detail(rid: str):
    r = next((x for x in RECIPES if x['id'] == rid), None)
    if not r: raise HTTPException(404, 'Recipe not found')
    d = dict(r)
    d['nutrition_source'] = enrich_recipe(r)
    return d

@app.post('/substitutions')
def substitutions(p: SubReq):
    blocked = set().union(*(ALLERGY_BLOCK.get(a,set()) for a in p.allergies))
    out = []
    for ing in p.missing_ingredients:
        alts = []
        for a in SUBS.get(normalize(ing), []):
            na = normalize(a)
            if na in blocked: continue
            if p.diet == 'Vegan' and na in {'paneer','milk','butter','cheese','yogurt'}: continue
            if p.diet == 'Vegetarian' and na in {'chicken','fish','egg'}: continue
            if any(na in ALLERGY_BLOCK.get(allergy,set()) for allergy in p.allergies): continue
            alts.append(a)
        out.append({'ingredient':ing, 'alternatives':alts, 'explanation':'Chosen for a similar cooking role while respecting the selected diet and allergy constraints.'})
    return {'substitutions':out}

@app.post('/meal-plan/generate')
def meal_plan(p: Prefs):
    safe_rs = [r for r in RECIPES if safe(r,p)]
    if not safe_rs: return {'days':[], 'message':'No recipes satisfy the current preferences.'}
    daily_target = p.calorie_limit * 3
    days=[]
    names=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    used_counts={r['id']:0 for r in safe_rs}
    recent_ingredients=set()
    for i, day in enumerate(names):
        def day_score(r):
            reuse=used_counts[r['id']]
            overlap=len(tokens(r['ingredients']) & recent_ingredients)
            rotation=(i + len(r['ingredients'])) % max(1, len(safe_rs))
            return (abs((r['calories']*3)-daily_target), -overlap, reuse, rotation)
        pool=sorted(safe_rs, key=day_score)
        chosen=[]
        for r in pool:
            if r['id'] not in {x['id'] for x in chosen}:
                chosen.append(r)
            if len(chosen)==3: break
        if not chosen:
            chosen=[safe_rs[i % len(safe_rs)]]
        for r in chosen: used_counts[r['id']]+=1
        recent_ingredients=set().union(*(tokens(r['ingredients']) for r in chosen))
        meals=[{'type':t,'name':r['name'],'calories':r['calories'],'recipe_id':r['id']} for t,r in zip(['Breakfast','Lunch','Dinner'],chosen)]
        days.append({'day':day,'meals':meals,'total_calories':sum(m['calories'] for m in meals)})
    return {'days':days, 'strategy':'preference-safe calorie targeting with recipe rotation and ingredient reuse across the week', 'target_calories':daily_target}

class GroceryReq(BaseModel):
    ingredients: List[str] = []
    diet: str = 'Vegetarian'
    cuisine: str = ''
    calorie_limit: int = Field(default=600, ge=100, le=5000)
    allergies: List[str] = []
    goal: str = 'Maintenance'
    recipe_ids: List[str] = []

    @field_validator('diet')
    @classmethod
    def valid_grocery_diet(cls, v):
        if v not in DIETS: raise ValueError('Invalid diet')
        return v

    @field_validator('allergies')
    @classmethod
    def valid_grocery_allergies(cls, v):
        bad = set(v) - ALLERGIES
        if bad: raise ValueError(f'Unsupported allergies: {sorted(bad)}')
        return v

@app.post('/grocery-list')
def grocery(p: GroceryReq):
    if p.recipe_ids:
        result = [r for r in RECIPES if r['id'] in set(p.recipe_ids)]
    else:
        if not p.ingredients:
            raise HTTPException(422, 'ingredients or recipe_ids is required')
        prefs = Prefs(ingredients=p.ingredients, diet=p.diet, cuisine=p.cuisine, calorie_limit=p.calorie_limit, allergies=p.allergies, goal=p.goal)
        result = suggest(prefs)['recipes'][:3]
    have = tokens(p.ingredients)
    counts={}
    for r in result:
        for x in set(r['ingredients']) - have:
            counts[x]=counts.get(x,0)+1
    return {'items':[{'ingredient':k,'recipe_count':v} for k,v in sorted(counts.items(),key=lambda x:(-x[1],x[0]))], 'recipes_used':[r['id'] for r in result]}

@app.post('/chat')
def chat(payload: dict):
    q=str(payload.get('message','')).strip()
    if not q: raise HTTPException(422,'message is required')
    if os.getenv('OPENAI_API_KEY'):
        from openai import OpenAI
        client=OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        resp=client.responses.create(model=os.getenv('OPENAI_MODEL','gpt-4o-mini'), input=f"Answer this cooking/nutrition question concisely and safely: {q}")
        return {'answer':resp.output_text,'source':'openai'}
    return {'answer':'Prototype mode: add OPENAI_API_KEY to enable LLM Q&A. Recipe retrieval, nutrition values, allergy filtering, substitutions and meal planning remain server-side.','source':'prototype-fallback'}
