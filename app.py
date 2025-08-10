import os
import torch
from flask import Flask, request, render_template
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 🔹 Model & Tokenizer Load
# Tum apna custom model path/model name yaha dal sakte ho
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()  # Inference mode

app = Flask(__name__)

# Home Route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['input_text']  # HTML form ka name="input_text"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()

    return render_template('index.html', prediction_text=f"Predicted Severity: {predicted_class}")

# Main entry point
if __name__ == "__main__":
    # Render ke PORT environment variable ko use karo
    port = int(os.environ.get("PORT", 10000))  # Default 10000 agar PORT set na ho
    app.run(host="0.0.0.0", port=port)
