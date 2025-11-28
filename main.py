from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import gradio as gr

app = FastAPI()

# Load model
model = joblib.load("house_model.pkl")

# FastAPI input model
class Input(BaseModel):
    data: Optional[list] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

@app.post("/predict")
def predict(input: Input):
    pred = model.predict([input.data])
    return {"prediction": float(pred[0])}


# ============================
# 🎨  Gradio UI
# ============================

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
    return f"Predicted House Price: {pred:.2f}"

# Gradio interface
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
    description="Enter the details and get the predicted house price",
)

# Mount Gradio on FastAPI
app = gr.mount_gradio_app(app, ui, path="/gradio")


# Run server (local use only)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
