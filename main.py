from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from fastapi import Request

app = FastAPI()

# Load the saved model
overall_model = tf.keras.models.load_model("electricity_usage_prediction_overall_model.h5")
device_model = tf.keras.models.load_model("electricity_usage_prediction_model.h5")

# For demonstration, using a mock scaled sequence. Replace with your actual last sequence.
last_sequence_overall = np.random.rand(10, 2)  # Replace with your actual scaled data

# Create the images directory if it doesn't exist
if not os.path.exists("images"):
    os.mkdir("images")

@app.get("/")
def read_root():
    return {"message": "Welcome to Electricity usage prediction API"}

@app.get("/predict")
def predict():
    global last_sequence_overall
    predicted_overall = []
    
    # Making predictions
    for i in range(10):
        seq_reshaped = last_sequence_overall[-10:].reshape(1, 10, 2)
        pred = overall_model.predict(seq_reshaped)
        predicted_overall.append(pred[0])
        
        # Append the new prediction to last_sequence_overall for future predictions
        last_sequence_overall = np.vstack((last_sequence_overall, pred))
    
    # Mock rescaling, replace with your actual scaler
    predicted_overall_rescaled = np.array(predicted_overall) * 100  # Replace with actual rescaling
    
    return JSONResponse(content={"predicted_values": predicted_overall_rescaled.tolist()})

@app.get("/predict/image")
def predict_image(request: Request):
    global last_sequence_overall
    
    # Generate a mock actual series for demonstration
    actual_series = np.random.rand(30) * 100  # Replace with your actual data
    
    # Generate the prediction plot
    plt.figure(figsize=(12, 6))
    plt.plot(actual_series, label='Actual', color='blue')
    plt.plot(np.arange(30, 40), last_sequence_overall[-10:, 0] * 100, label='Predicted', color='red')  # Replace with actual rescaling
    plt.title("Electricity Usage for Overall_Usage")
    plt.xlabel("Days")
    plt.ylabel("Electricity Usage")
    plt.legend()
    
    # Save the plot as an image
    image_path = "images/predicted_plot.png"
    plt.savefig(image_path)

    # Construct the full URL
    full_url = str(request.base_url) + f"files/{image_path}"
    
    return JSONResponse(content={"image_link": full_url})

@app.get("/predict/devices/images")
async def predict_all_images(request: Request):
    global last_sequence_overall
    
    # Replace with your actual data preparation for all devices
    # For demonstration, generate a mock actual series for all devices
    all_device_actual_series = {
        "Device1": np.random.rand(30) * 100,
        "Device2": np.random.rand(30) * 100,
        # Add more devices as needed
    }
    
    # Generate prediction plots for all devices
    all_device_plots = {}
    
    for device, actual_series in all_device_actual_series.items():
        plt.figure(figsize=(12, 6))
        plt.plot(actual_series, label='Actual', color='blue')
        plt.plot(np.arange(30, 40), last_sequence_overall[-10:, 0] * 100, label='Predicted', color='red')  # Replace with actual rescaling
        plt.title(f"Electricity Usage for {device}")
        plt.xlabel("Days")
        plt.ylabel("Electricity Usage")
        plt.legend()
        
        # Save the plot as an image for each device
        image_path = f"images/{device}_predicted_plot.png"
        plt.savefig(image_path)
        
        # Construct the full URL for each device's image
        full_url = str(request.base_url) + f"/files/{image_path}"
        all_device_plots[device] = full_url
    
    return JSONResponse(content={"device_images": all_device_plots})

@app.get("/recommend")
def get_recommendation():
    return None

@app.get("/recommend/images")
def get_recommend_images():
    return None

# Mount the images directory to serve image files
app.mount("/files", StaticFiles(directory="."), name="files")
    