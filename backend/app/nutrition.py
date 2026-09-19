"""USDA FoodData Central adapter with a deterministic local fallback.
The app never exposes USDA credentials to the browser.
"""
import os
from functools import lru_cache
import httpx

NUTRIENT_KEYS = {
    'Energy': ('calories', 'kcal'),
    'Protein': ('protein', 'g'),
    'Carbohydrate, by difference': ('carbs', 'g'),
    'Total lipid (fat)': ('fat', 'g'),
}

@lru_cache(maxsize=128)
def lookup_food(query: str):
    key = os.getenv('USDA_API_KEY')
    if not key:
        return None
    try:
        r = httpx.get(
            'https://api.nal.usda.gov/fdc/v1/foods/search',
            params={'api_key': key, 'query': query, 'pageSize': 1},
            timeout=5.0,
        )
        r.raise_for_status()
        foods = r.json().get('foods') or []
        return foods[0] if foods else None
    except Exception:
        return None

def nutrition_for_ingredient(query: str):
    food = lookup_food(query)
    if not food:
        return None
    values = {}
    for n in food.get('foodNutrients', []):
        name = n.get('nutrientName') or n.get('nutrient', {}).get('name')
        if name in NUTRIENT_KEYS:
            field, unit = NUTRIENT_KEYS[name]
            values[field] = {'value': n.get('value'), 'unit': n.get('unit') or unit}
    return {'fdc_id': food.get('fdcId'), 'description': food.get('description'), 'nutrients': values}

def enrich_recipe(recipe):
    """Return USDA nutrition for the recipe's first ingredient when configured.
    Recipe-level totals remain the curated prototype values unless a full
    ingredient-quantity calculation is supplied in a future dataset version.
    """
    if not os.getenv('USDA_API_KEY'):
        return {'source': 'local-prototype-adapter'}
    sample = recipe.get('ingredients', [None])[0]
    result = nutrition_for_ingredient(sample) if sample else None
    if not result:
        return {'source': 'local-prototype-adapter', 'usda_status': 'lookup-unavailable'}
    return {'source': 'usda-fooddata-central', 'sample_ingredient': sample, 'sample': result}
