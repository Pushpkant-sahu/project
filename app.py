# app.py

import torch
from flask import Flask, request, render_template
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# Load Hugging Face model and tokenizer
model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract single input value from form
    input_val = float(request.form['input_value'])  # Adjust type if needed (int/float)
    final_features = [[input_val]]  # 2D array, shape (1, 1)

    # Convert input to PyTorch tensor
    input_tensor = torch.tensor(final_features, dtype=torch.float32)

    # Make prediction
    with torch.no_grad():
        prediction = model(input_tensor)
        predicted_class = torch.argmax(prediction, dim=1).item()

    # Interpret prediction
    output = 'Placed' if predicted_class == 1 else 'Not Placed'

    return render_template('index.html', prediction_text=f'Prediction: {output}')

if __name__ == "__main__":
    app.run(debug=True)
