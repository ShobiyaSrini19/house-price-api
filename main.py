from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import gradio as gr

app = FastAPI()

# Load model
model = joblib.load("house_model.pkl")


# ----------------------------
# FastAPI endpoint (USD)
# ----------------------------
class Input(BaseModel):
    data: Optional[list] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

@app.post("/predict")
def predict(input: Input):
    pred = model.predict([input.data])[0]
    # Model output is in $100,000 units
    price_usd = pred * 100000
    return {"prediction_usd": price_usd}


# ----------------------------
# Gradio Predict Function
# ----------------------------
def gradio_predict(MedInc, HouseAge, AveRooms, AveBedrms,
                   Population, AveOccup, Latitude, Longitude):

    data = [
        MedInc, HouseAge, AveRooms, AveBedrms,
        Population, AveOccup, Latitude, Longitude
    ]

    pred = model.predict([data])[0]     # model output ($100k units)
    price_usd = pred * 100000           # convert to actual dollars

    return f"💰 Estimated House Price: ${price_usd:,.2f}"


# ----------------------------
# 🌈 Custom Colourful Theme
# ----------------------------
custom_theme = gr.themes.Grape(
    primary_hue="violet",
    secondary_hue="orange",
    neutral_hue="slate"
).set(
    button_primary_background="linear-gradient(90deg, #ff7e5f, #feb47b)",
    button_primary_text_color="white",
    button_primary_hover_background="linear-gradient(90deg, #ff9566, #ffc37f)",
    input_background_fill="white",
    body_background_fill="linear-gradient(135deg, #667eea, #764ba2)",
    body_text_color="white",
    block_title_text_color="white"
)


# ----------------------------
# 🌟 Gradio UI
# ----------------------------
ui = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Number(label="🏠 Median Income (×10k USD)", value=8.3252),
        gr.Number(label="⏳ House Age", value=41),
        gr.Number(label="🛏 Average Rooms", value=6.98),
        gr.Number(label="🛌 Average Bedrooms", value=1.02),
        gr.Number(label="👨‍👩‍👧 Population", value=322),
        gr.Number(label="👥 Average Occupancy", value=2.55),
        gr.Number(label="📍 Latitude", value=37.88),
        gr.Number(label="📍 Longitude", value=-122.23),
    ],
    outputs=gr.Textbox(label="Predicted Price (USD)"),
    title="🏡 House Price Predictor (USD)",
    description="Enter house details to estimate its value in U.S. Dollars ($).",
    theme=custom_theme,
)

# Mount inside FastAPI
app = gr.mount_gradio_app(app, ui, path="/gradio")


# Run Locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
