
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model (make sure 'model.h5' exists in the same directory)
model = tf.keras.models.load_model("mobileNet_model.h5")

# Class labels (edit this if you have different ones)
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

# Color mapping for classes
colors = {
    "CaS": "#3498db",
    "CoS": "#2ecc71",
    "Gum": "#e67e22",
    "MC": "#9b59b6",
    "OC": "#e74c3c",
    "OLP": "#1abc9c",
    "OT": "#f1c40f",
}

# Configure Streamlit page
st.set_page_config(
    page_title="Teeth Classification App",
    page_icon="🦷",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Optional custom CSS
st.markdown("""
<style>
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-size: 16px;
        border-radius: 8px;
        padding: 10px 24px;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Sidebar info
st.sidebar.title("🧪 About")
st.sidebar.info("""
This app uses a deep learning model to classify teeth images into 7 categories:

- CaS (Caries Superficial)
- CoS (Caries Occlusal)
- Gum Disease
- MC (Missing Crown)
- OC (Other Condition)
- OLP (Oral Lichen Planus)
- OT (Other Tooth Problem)

**Built with**: TensorFlow, Streamlit
                We still recommend consulting a dental professional for accurate diagnosis and treatment.
**Note**: This app is for educational purposes only and should not replace professional medical advice.
""")

# Main title
st.markdown("<h1 style='text-align: center; color: #0066cc;'>🦷 Teeth Disease Classification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Upload a teeth image to get an instant diagnosis from the AI model.</p>", unsafe_allow_html=True)
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.markdown("### 📸 Uploaded Image")
    st.image(image, caption="Teeth Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Make prediction
    predictions = model.predict(img_array)
    pred_class_idx = np.argmax(predictions)
    pred_class = class_names[pred_class_idx]
    confidence = predictions[0][pred_class_idx] * 100

    color = colors.get(pred_class, "#7f8c8d")
    st.markdown(
        f"<h3 style='text-align:center; color:{color}'>✅ Prediction: {pred_class} ({confidence:.2f}%)</h3>",
        unsafe_allow_html=True
    )

    # Show raw probabilities (optional)
    with st.expander("🔎 See prediction confidence for all classes"):
        for i, cls in enumerate(class_names):
            st.write(f"**{cls}**: {predictions[0][i]*100:.2f}%")

st.markdown("---")
st.markdown("<p style='text-align: center; font-size:14px;'>Made with ❤️ by Noureldeen • Powered by TensorFlow & Streamlit</p>", unsafe_allow_html=True)



