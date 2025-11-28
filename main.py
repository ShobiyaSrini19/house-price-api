from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import gradio as gr

# FastAPI app
app = FastAPI()

# Load your model
model = joblib.load("house_model.pkl")


# -------------------------
# FASTAPI PREDICT ENDPOINT
# -------------------------
class Input(BaseModel):
    data: Optional[list] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

@app.post("/predict")
def predict(input: Input):
    pred = model.predict([input.data])[0]
    return {"prediction": float(pred)}


# -------------------------
# GRADIO UI FUNCTION
# -------------------------
def gradio_predict(
    MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude
):
    data = [
        MedInc,
        HouseAge,
        AveRooms,
        AveBedrms,
        Population,
        AveOccup,
        Latitude,
        Longitude,
    ]
    pred = model.predict([data])[0]
    return f"Predicted Price: {pred:.2f}"


# Gradio Interface
ui = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Number(label="Median Income"),
        gr.Number(label="House Age"),
        gr.Number(label="Average Rooms"),
        gr.Number(label="Average Bedrooms"),
        gr.Number(label="Population"),
        gr.Number(label="Average Occupancy"),
        gr.Number(label="Latitude"),
        gr.Number(label="Longitude"),
    ],
    outputs="text",
    title="California House Price Predictor",
    description="Enter values to predict house prices.",
)

# IMPORTANT: New mount API for Gradio v4+
app = gr.mount_gradio_app(app, ui, path="/gradio")


# Run locally (ignored on Render)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
