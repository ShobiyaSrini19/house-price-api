from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import joblib
import gradio as gr

app = FastAPI()

# Load Model
model = joblib.load("house_model.pkl")


# ----------------------------
# FASTAPI PREDICT ENDPOINT
# ----------------------------
class Input(BaseModel):
    data: Optional[list] = [8.3252, 41.0, 6.98, 1.02, 322, 2.55, 37.88, -122.23]

@app.post("/predict")
def predict(input: Input):
    pred = model.predict([input.data])[0]
    inr_price = pred * 100000 * 85  # USD → INR
    return {"prediction_in_inr": int(inr_price)}


# ----------------------------
# GRADIO UI FUNCTION
# ----------------------------
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

    pred = model.predict([data])[0]          # model output (in $100k)
    price_in_usd = pred * 100000             # convert to USD
    price_in_inr = price_in_usd * 85         # convert to INR

    return f"🏡 Estimated House Price: ₹{price_in_inr:,.0f} INR"


# ----------------------------
# BEAUTIFUL UI
# ----------------------------
css = """
#root {background: linear-gradient(135deg, #6a11cb, #2575fc); height: 100vh;}
h1 {color: white !important; text-align: center !important;}
p {color: #f0f0f0 !important; text-align: center;}
"""

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
    title="🏡 House Price Predictor (INR Version)",
    description="Enter the values below to get the house price in Indian Rupees (₹).",
    css=css,
)

app = gr.mount_gradio_app(app, ui, path="/gradio")



# Run Locally
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=10000)
