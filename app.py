import os
import gdown
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ✅ Updated model link (new .keras file)
model_url = 'https://drive.google.com/uc?id=1T8_HZT8tjXabaZlBo2UNC5vScSHxUZRQ'
model_path = 'cnn_model.keras'

# ✅ Download model if not already present
if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# ✅ Load model with Streamlit caching
@st.cache_resource
def load_cnn_model():
    return load_model(model_path)

model = load_cnn_model()

# ✅ Label map
label_map = {
    0: "benign",
    1: "malignant",
    2: "glioma",
    3: "meningioma",
    4: "notumor",
    5: "pituitary",
    6: "pneumonia"
}

# ✅ Sidebar
st.sidebar.title("🧠 Multi-Disease Classifier")
st.sidebar.markdown("""
Upload a medical image (JPG or PNG).  
The model will predict the disease class and show confidence score.
""")

# ✅ Main UI
st.title("📷 Medical Disease Detection")
st.write("Upload a medical image to get a prediction.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0]
    predicted_class = np.argmax(pred)
    confidence = np.max(pred)

    # Layout: image + result side by side
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="Uploaded Image", use_column_width=True)
    with col2:
        st.markdown("### 🧠 Predicted Disease")
        st.markdown(f"**{label_map[predicted_class].capitalize()}**")
        st.markdown("### 📈 Confidence Score")
        st.markdown(f"**{confidence * 100:.2f}%**")
        st.progress(float(confidence))
