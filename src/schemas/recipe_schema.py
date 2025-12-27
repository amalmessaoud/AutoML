from pydantic import BaseModel, Field
from typing import List

class Ingredient(BaseModel):
    name: str = Field(..., description="Name of the ingredient")
    quantity: str = Field(..., description="Amount, e.g., '2 cups' or '200g'")

class Step(BaseModel):
    number: int = Field(..., description="Step number starting from 1")
    description: str = Field(..., description="Clear instruction")

class RecipePlan(BaseModel):
    title: str = Field(..., description="Creative name for the dish")
    meal_type: str = Field(..., description="e.g., breakfast, lunch, dinner")
    servings: int = Field(..., ge=1, description="Number of people")
    ingredients_used: List[Ingredient] = Field(..., description="Only ingredients from the user's list")
    steps: List[Step] = Field(..., min_items=3, description="Step-by-step cooking instructions")
    prep_time_minutes: int = Field(..., ge=1, description="Estimated preparation time")
    notes: str = Field("", description="Optional tips or warnings")