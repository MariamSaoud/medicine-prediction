import os
import requests

url = "https://github.com/MariamSaoud/medicine-prediction/raw/main/knn_model.pkl"
filename = "knn_model.pkl"

if not os.path.exists(filename):
    print("📥 Downloading model file...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print("✅ Model downloaded successfully.")
    else:
        print(f"❌ Failed to download model. Status code: {response.status_code}")
else:
    print("🟢 Model already exists. Skipping download.")