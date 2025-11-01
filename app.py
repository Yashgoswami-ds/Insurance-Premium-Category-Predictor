from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, computed_field
from typing import Literal, Annotated
import pickle
import pandas as pd
from fastapi.responses import JSONResponse

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

tier_1_cities = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", "Hyderabad", "Pune"]
tier_2_cities = ["Jaipur", "Chandigarh", "Indore", "Lucknow", "Patna", "Ranchi", "Visakhapatnam", "Coimbatore", "Bhopal"]

class UserIp(BaseModel):
    id: Annotated[str, Field(..., description="Unique ID of the user")]
    age: Annotated[int, Field(..., description="Age of the user", gt=0, lt=120)]
    weight: Annotated[float, Field(..., description="Weight of the user in kilograms", gt=0)]
    height: Annotated[float, Field(..., description="Height of the user in meters", gt=0)]
    smoker: Annotated[Literal['yes', 'no'], Field(..., description="Whether the user smokes")]
    city: Annotated[str, Field(..., description="City where the user resides")]
    occupation: Annotated[Literal['retired', 'freelancer', 'student', 'government_job', 'business_owner', 'unemployed', 'private_job'], Field(..., description="Occupation of the user")]
    income_lpa: Annotated[float, Field(..., description="Income of the user in Lakhs per annum")]

    @computed_field()
    @property
    def bmi(self) -> float:
        return round(self.weight / (self.height ** 2), 2)

    @computed_field()
    @property
    def lifestyle_risk(self) -> str:
        if self.smoker == 'yes' and self.bmi > 30:
            return "high"
        elif self.smoker == 'yes' or self.bmi > 27:
            return "medium"
        else:
            return "low"

    @computed_field()
    @property
    def age_group(self) -> str:
        if self.age < 25:
            return "young"
        elif self.age < 45:
            return "adult"
        elif self.age < 60:
            return "middle_aged"
        return "senior"

    @computed_field()
    @property
    def city_tier(self) -> int:
        if self.city in tier_1_cities:
            return 1
        elif self.city in tier_2_cities:
            return 2
        else:
            return 3

@app.post("/predict")
def predict_premium(data: UserIp):
    try:
        input_df = pd.DataFrame([{
            'bmi': data.bmi,
            'age_group': data.age_group,
            'lifestyle_risk': data.lifestyle_risk,
            'city_tier': data.city_tier,
            'income_lpa': data.income_lpa,
            'occupation': data.occupation
        }])

        prediction = model.predict(input_df)[0]

        return JSONResponse(status_code=200, content={
            "response": {
                "predicted_category": prediction,
                "confidence": {},  # <-- changed from None
                "class_probabilities": {}  # <-- changed from None
            }
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")
