"""
schemas.py
----------
Pydantic models define the shape of API requests and responses.
This keeps input validation clear for beginners.
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field


class StudentHealthInput(BaseModel):
    """Features collected from the Angular form."""

    sleep_duration: Optional[float] = Field(
        default=None, description="Hours of sleep per night"
    )
    heart_rate: Optional[float] = Field(default=None, description="Beats per minute")
    bmi: Optional[float] = Field(default=None, description="Body Mass Index")
    calorie_expenditure: Optional[float] = Field(
        default=None, description="Daily calorie expenditure"
    )
    step_count: Optional[float] = Field(default=None, description="Daily step count")
    exercise_duration: Optional[float] = Field(
        default=None, description="Exercise minutes"
    )
    water_intake: Optional[float] = Field(
        default=None, description="Litres of water per day"
    )
    diet_type: Optional[str] = Field(
        default=None, description="veg | non-veg | balanced"
    )
    stress_level: Optional[str] = Field(
        default=None, description="low | medium | high"
    )
    sleep_quality: Optional[str] = Field(
        default=None, description="poor | average | good"
    )
    physical_activity_level: Optional[str] = Field(
        default=None, description="sedentary | moderate | active"
    )
    smoking_alcohol: Optional[str] = Field(
        default=None, description="no | occasional | yes"
    )
    gender: Optional[str] = Field(default=None, description="male | female | other")


class PredictionResponse(BaseModel):
    health_condition: str
    probabilities: Dict[str, float]
    message: str
