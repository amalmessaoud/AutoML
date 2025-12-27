import ollama
from src.schemas.recipe_schema import RecipePlan


def generate_recipe(available_ingredients: str, meal_type: str, model: str = "llama3.1:8b") -> RecipePlan:
    prompt = f"""
You are a precise chef. Your task is to output a recipe in EXACTLY the required JSON format.

Available ingredients (use ONLY these): {available_ingredients}
Meal type: {meal_type}

Here is the EXACT JSON format you MUST follow:

EXAMPLE (do not copy this recipe, just the structure):
{{
  "title": "Spicy Garlic Chicken Stir-Fry",
  "meal_type": "dinner",
  "servings": 4,
  "ingredients_used": [
    {{"name": "chicken", "quantity": "500g"}},
    {{"name": "garlic", "quantity": "6 cloves"}},
    {{"name": "olive oil", "quantity": "3 tablespoons"}},
    {{"name": "paprika", "quantity": "1 teaspoon"}}
  ],
  "steps": [
    {{"number": 1, "description": "Heat olive oil in a large pan over medium heat."}},
    {{"number": 2, "description": "Mince garlic and add to the pan, cook for 1 minute."}},
    {{"number": 3, "description": "Add diced chicken and paprika, stir-fry until cooked through."}},
    {{"number": 4, "description": "Serve hot with rice or vegetables."}}
  ],
  "prep_time_minutes": 25,
  "notes": "Adjust paprika for desired spice level."
}}

Rules:
- Use ONLY these field names: title, meal_type, servings, ingredients_used, steps, prep_time_minutes, notes
- servings is a single integer
- ingredients_used: list of objects with "name" and "quantity" (no "unit")
- steps: list of objects with "number" (starting at 1) and "description"
- Return ONLY the JSON. No explanations, no markdown, no extra text before or after.

Now create a new recipe using the available ingredients:
"""

    print("🤖 Thinking... (streaming output)\n")

    stream = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        format="json",
        temperature=0.3,
    )

    raw_output = ""
    for chunk in stream:
        content = chunk["message"]["content"]
        print(content, end="", flush=True)
        raw_output += content

    print("\n\n✅ Done thinking!\n")

    try:
        recipe = RecipePlan.model_validate_json(raw_output)
        print("\n🎉 SUCCESS! Valid structured recipe:")
        print(recipe.model_dump_json(indent=2))
        return recipe
    except Exception as e:
        print("\n❌ JSON validation failed:", e)
        print("Raw output was:")
        print(raw_output)
        raise e

# Quick test at the bottom
if __name__ == "__main__":
    ingredients = "chicken, rice, tomatoes, onions, garlic, olive oil, salt, pepper, paprika"
    meal = "dinner"

    try:
        recipe = generate_recipe(ingredients, meal)
        print("\n✅ Valid structured recipe:")
        print(recipe.model_dump_json(indent=2))
    except Exception as e:
        print("\n❌ Validation failed:", e)