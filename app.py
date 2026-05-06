import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np

from gradcam_utils import generate_gradcam

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Industrial Defect AI",
    page_icon="🔍",
    layout="wide"
)

# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 1200px;
}

.title-text {
    font-size: 44px;
    font-weight: 700;
    color: white;
    margin-bottom: 0px;
    line-height: 1.2;
}

.subtitle-text {
    font-size: 18px;
    color: #BBBBBB;
    margin-top: 5px;
    margin-bottom: 25px;
}

.section-title {
    font-size: 28px;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 20px;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    font-size: 16px;
    font-weight: 600;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
    padding: 10px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# DEVICE
# ------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        2
    )

    model.load_state_dict(
        torch.load(
            "models/defect_model.pth",
            map_location=DEVICE
        )
    )

    model = model.to(DEVICE)

    model.eval()

    return model

model = load_model()

# ------------------------------------------------
# IMAGE TRANSFORM
# ------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ------------------------------------------------
# CLASS LABELS
# ------------------------------------------------

classes = {
    0: "Good",
    1: "Defective"
}

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

with st.sidebar:

    st.title("⚙️ Model Info")

    st.markdown("---")

    st.write("### Architecture")
    st.write("ResNet18")

    st.write("### Explainability")
    st.write("GradCAM")

    st.write("### Framework")
    st.write("PyTorch")

    st.write("### Device")
    st.write(str(DEVICE).upper())

    st.markdown("---")

    st.write("### Demo Samples")

    st.write(
        "Use built-in sample images "
        "to test the model instantly."
    )

# ------------------------------------------------
# HERO SECTION
# ------------------------------------------------

st.markdown(
    """
    <div class="title-text">
    🔍 Explainable Industrial <br>
    Defect Detection
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle-text">
    AI-powered industrial visual inspection system using
    Deep Learning and GradCAM Explainability
    </div>
    """,
    unsafe_allow_html=True
)

# ------------------------------------------------
# IMAGE INPUT SECTION
# ------------------------------------------------

st.markdown(
    '<div class="section-title">📂 Choose Image</div>',
    unsafe_allow_html=True
)

option = st.radio(
    "Select Input Method",
    [
        "Use Demo Sample",
        "Upload Your Own Image"
    ]
)

image = None

# ------------------------------------------------
# UPLOAD OPTION
# ------------------------------------------------

if option == "Upload Your Own Image":

    uploaded_file = st.file_uploader(
        "Upload Industrial Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

# ------------------------------------------------
# DEMO SAMPLE OPTION
# ------------------------------------------------

else:

    st.write("### Select Demo Sample")

    demo_col1, demo_col2 = st.columns(
        [1, 1],
        gap="large"
    )

    with demo_col1:

        if st.button("Use Defective Sample"):

            st.session_state["selected_image"] = (
                "sample_images/defective_sample.png"
            )

    with demo_col2:

        if st.button("Use Good Sample"):

            st.session_state["selected_image"] = (
                "sample_images/good_sample.png"
            )

    if "selected_image" in st.session_state:

        image = Image.open(
            st.session_state["selected_image"]
        ).convert("RGB")

# ------------------------------------------------
# MAIN FLOW
# ------------------------------------------------

if image is not None:

    st.markdown(
        '<div class="section-title">Selected Image</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Original Image")

        st.image(
            image,
            width='stretch'
        )

    analyze = st.button("Analyze Image")

    if analyze:

        input_tensor = transform(image)

        input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(DEVICE)

        with torch.no_grad():

            outputs = model(input_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                1
            )

        prediction = classes[predicted.item()]

        confidence_score = confidence.item() * 100

        image_np = np.array(image)

        gradcam_result = generate_gradcam(
            model,
            input_tensor,
            image_np
        )

        with col2:

            st.subheader("GradCAM Heatmap")

            st.image(
                gradcam_result,
                width='stretch'
            )

        st.markdown("## 📊 Inspection Results")

        metric1, metric2, metric3 = st.columns(3)

        with metric1:

            st.metric(
                label="Prediction",
                value=prediction
            )

        with metric2:

            st.metric(
                label="Confidence",
                value=f"{confidence_score:.2f}%"
            )

        with metric3:

            st.metric(
                label="Model Status",
                value="Active"
            )

        st.write("### Confidence Score")

        st.progress(
            float(confidence_score / 100)
        )

        if prediction == "Good":

            st.success(
                "✅ No visible defects detected."
            )

        else:

            st.error(
                "⚠️ Defect detected in product."
            )

        st.info(
            "GradCAM highlights the regions "
            "that influenced the AI prediction."
        )

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.markdown(
    """
    <div class="footer">
    Built using PyTorch, Streamlit, ResNet18 and GradCAM
    </div>
    """,
    unsafe_allow_html=True
)