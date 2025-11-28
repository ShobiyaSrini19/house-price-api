from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import gradio as gr

app = FastAPI()

# Load ML model
model = joblib.load("house_model.pkl")


# ----------------------------
# FastAPI endpoint
# ----------------------------
class Input(BaseModel):
    data: Optional[list] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

@app.post("/predict")
def predict(input: Input):
    pred = model.predict([input.data])[0]
    price_in_inr = pred * 100000 * 85
    return {"prediction_in_inr": int(price_in_inr)}


# ----------------------------
# Gradio UI Function
# ----------------------------
def gradio_predict(MedInc, HouseAge, AveRooms, AveBedrms,
                   Population, AveOccup, Latitude, Longitude):
    data = [
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ]
    
    pred = model.predict([data])[0]
    price_in_usd = pred * 100000
    price_in_inr = price_in_usd * 85

    return f"Estimated House Price: ₹{price_in_inr:,.0f}"


# ----------------------------
# Gradio UI (v4 compatible)
# ----------------------------
ui = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Number(label="Median Income (×10k USD)", value=8.3252),
        gr.Number(label="House Age", value=41),
        gr.Number(label="Average Rooms", value=6.98),
        gr.Number(label="Average Bedrooms", value=1.02),
        gr.Number(label="Population", value=322),
        gr.Number(label="Average Occupancy", value=2.55),
        gr.Number(label="Latitude", value=37.88),
        gr.Number(label="Longitude", value=-122.23),
    ],
    outputs=gr.Textbox(label="Predicted Price (INR)"),
    title="🏡 House Price Predictor (INR)",
    description="Enter housing details to predict selling price (converted to Indian Rupees ₹).",
    theme=gr.themes.Soft()   # 🌈 Beautiful theme that works in Gradio v4
)

# Mount inside FastAPI
app = gr.mount_gradio_app(app, ui, path="/gradio")


# Run locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
