from fastapi.testclient import TestClient
from backend.app.main import app
c=TestClient(app)
def test_health(): assert c.get('/health').status_code==200
def test_suggest_and_normalize():
 r=c.post('/recipes/suggest',json={'ingredients':['ONIONS','tomatoes','spinach'],'diet':'Vegan','cuisine':'Indian','calorie_limit':600,'allergies':[],'goal':'Maintenance'}); assert r.status_code==200; assert r.json()['normalized_ingredients']==['onion','spinach','tomato']
def test_details(): assert c.get('/recipes/r1').status_code==200
def test_allergy_filter():
 r=c.post('/recipes/suggest',json={'ingredients':['spinach','onion'],'diet':'Vegetarian','cuisine':'Indian','calorie_limit':600,'allergies':['Dairy'],'goal':'Maintenance'}); assert all('Dairy' not in x['allergens'] for x in r.json()['recipes'])
def test_subs_respect_allergy():
 r=c.post('/substitutions',json={'missing_ingredients':['spinach','milk'],'diet':'Vegan','allergies':['Nuts']}); assert r.status_code==200; assert 'almond milk' not in sum([x['alternatives'] for x in r.json()['substitutions']],[])
def test_plan():
 r=c.post('/meal-plan/generate',json={'ingredients':['onion'],'diet':'Vegan','cuisine':'Indian','calorie_limit':600,'allergies':[],'goal':'Maintenance'}); assert r.status_code==200; assert len(r.json()['days'])==7

def test_grocery_requires_source():
 r=c.post('/grocery-list',json={'ingredients':[],'recipe_ids':[]})
 assert r.status_code==422

def test_detail_has_nutrition_adapter():
 r=c.get('/recipes/r1'); assert r.status_code==200
 assert r.json()['nutrition_source']['source'] in {'local-prototype-adapter','usda-fooddata-central'}

def test_plan_has_meal_slots_and_rotation():
 r=c.post('/meal-plan/generate',json={'ingredients':['onion','tomato'],'diet':'Vegetarian','cuisine':'Indian','calorie_limit':600,'allergies':[],'goal':'Maintenance'})
 assert r.status_code==200
 days=r.json()['days']; assert all(len(d['meals'])==3 for d in days)
 assert len({m['recipe_id'] for d in days for m in d['meals']}) >= 3


def test_plan_reuses_ingredients_between_adjacent_days():
 r=c.post('/meal-plan/generate',json={'ingredients':['onion','tomato','spinach'],'diet':'Vegetarian','cuisine':'Indian','calorie_limit':600,'allergies':[],'goal':'Maintenance'})
 assert r.status_code==200
 days=r.json()['days']
 assert len(days)==7
 assert all(len(d['meals'])==3 for d in days)


def test_detail_contains_shareable_recipe_content():
 r=c.get('/recipes/r1')
 assert r.status_code==200
 body=r.json()
 assert body['name'] and body['calories'] >= 0 and body['instructions']


def test_usda_adapter_without_key_uses_local_fallback(monkeypatch):
 from backend.app import nutrition
 monkeypatch.delenv('USDA_API_KEY', raising=False)
 assert nutrition.lookup_food('spinach') is None
 assert nutrition.enrich_recipe({'ingredients':['spinach']}) == {'source':'local-prototype-adapter'}


def test_cors_allows_vite_origin():
 r=c.options('/health',headers={'Origin':'http://localhost:5173','Access-Control-Request-Method':'GET'})
 assert r.status_code in {200,204}
 assert r.headers.get('access-control-allow-origin') == 'http://localhost:5173'


def test_health_reports_integration_modes():
 body=c.get('/health').json()
 assert body['status']=='ok'
 assert body['rag'] in {'faiss','numpy-fallback (FAISS unavailable in current environment)'}
 assert body['embedding_provider'] in {'openai-recipe-embeddings','local-ingredient-vector-prototype'}
 assert body['nutrition'] in {'usda-configured','local-prototype-adapter'}
