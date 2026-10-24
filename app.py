
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from PIL import Image
from streamlit_drawable_canvas import st_canvas


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DigitAI | Number Recognition",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "simple_cnn_mnist.pth"
CSS_PATH = BASE_DIR / "style.css"


# --------------------------------------------------
# 2. LOAD CSS
# --------------------------------------------------

def load_css():
    if CSS_PATH.exists():
        css = CSS_PATH.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


load_css()


# --------------------------------------------------
# 3. CNN MODEL
# Architecture must match the model used for training
# --------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------------------------------------
# 4. LOAD TRAINED MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    model = SimpleCNN()

    try:
        state_dict = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(
            MODEL_PATH,
            map_location="cpu",
        )

    model.load_state_dict(state_dict)
    model.eval()

    return model


model = load_model()


# --------------------------------------------------
# 5. IMAGE PREPROCESSING
# --------------------------------------------------

def make_foreground(image):
    """
    Convert an image into a binary image where the digit
    is white and the background is black.
    """

    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGBA"))

    image = np.asarray(image)

    # Handle transparent canvas images
    if image.ndim == 3 and image.shape[2] == 4:
        rgb = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3].astype(np.float32) / 255.0

        # Composite transparent pixels onto a black background
        image = (rgb * alpha[:, :, None]).astype(np.uint8)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Detect background brightness from the image borders
    border_pixels = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1],
    ])

    background_is_bright = np.median(border_pixels) > 127

    if background_is_bright:
        gray = 255 - gray

    # Threshold the digit
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    return binary


def prepare_single_digit(binary):
    """
    Resize a digit to 28 x 28 and center it,
    similar to the MNIST input format.
    """

    coords = cv2.findNonZero(binary)

    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    digit = binary[y:y + h, x:x + w]

    # Leave space around the digit
    target_size = 20
    scale = target_size / max(w, h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w,
    ] = resized

    return canvas.astype(np.float32) / 255.0


# --------------------------------------------------
# 6. SEGMENT MULTIPLE DIGITS
# --------------------------------------------------

def segment_digits(binary):
    contours, _ = cv2.findContours(
        binary.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    boxes = []

    image_height, image_width = binary.shape
    minimum_area = max(8, int(image_height * image_width * 0.0001))

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < minimum_area:
            continue

        if h < 5 or w < 2:
            continue

        boxes.append((x, y, w, h))

    # Read digits from left to right
    boxes.sort(key=lambda box: box[0])

    return boxes


# --------------------------------------------------
# 7. PREDICT ONE DIGIT
# --------------------------------------------------

def predict_one_digit(digit_image):
    if model is None:
        return None

    tensor = torch.tensor(
        digit_image,
        dtype=torch.float32,
    ).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    predicted_digit = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_digit].item())

    top_values, top_indices = torch.topk(probabilities, 3)

    top_predictions = [
        {
            "Digit": int(index.item()),
            "Confidence": float(value.item()),
        }
        for value, index in zip(top_values, top_indices)
    ]

    return {
        "digit": predicted_digit,
        "confidence": confidence,
        "probabilities": probabilities.numpy(),
        "top_predictions": top_predictions,
        "processed_image": digit_image,
    }


# --------------------------------------------------
# 8. PREDICT A NUMBER
# --------------------------------------------------

def predict_number(image):
    binary = make_foreground(image)
    boxes = segment_digits(binary)

    predictions = []

    for x, y, w, h in boxes:
        # Crop each digit and add a small border
        padding = max(3, int(max(w, h) * 0.10))

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(binary.shape[1], x + w + padding)
        y2 = min(binary.shape[0], y + h + padding)

        crop = binary[y1:y2, x1:x2]
        prepared = prepare_single_digit(crop)

        if prepared is None:
            continue

        result = predict_one_digit(prepared)

        if result is not None:
            predictions.append(result)

    predicted_number = "".join(
        str(item["digit"]) for item in predictions
    )

    if predictions:
        average_confidence = float(
            np.mean([item["confidence"] for item in predictions])
        )
    else:
        average_confidence = 0.0

    return {
        "number": predicted_number,
        "confidence": average_confidence,
        "digits": predictions,
        "binary_image": binary,
    }


# --------------------------------------------------
# 9. SESSION HISTORY
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []


def add_to_history(source, result):
    if result["number"]:
        st.session_state.history.append({
            "Source": source,
            "Prediction": result["number"],
            "Confidence": round(result["confidence"] * 100, 2),
        })


# --------------------------------------------------
# 10. DISPLAY PREDICTION
# --------------------------------------------------

def display_result(result):
    if not result or not result["number"]:
        st.warning(
            "No digit detected. Draw or upload a clearer image "
            "with dark digits on a light background, or vice versa."
        )
        return

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Number",
            result["number"],
        )

    with col2:
        st.metric(
            "Average Confidence",
            f"{result['confidence'] * 100:.2f}%",
        )

    st.progress(
        min(max(result["confidence"], 0.0), 1.0)
    )

    st.subheader("Individual Digit Predictions")

    digit_columns = st.columns(
        min(len(result["digits"]), 5)
    )

    for index, digit_result in enumerate(result["digits"]):
        with digit_columns[index % len(digit_columns)]:
            st.image(
                digit_result["processed_image"],
                caption=f"Digit {index + 1}",
                width=100,
            )

            st.write(
                f"**Prediction:** {digit_result['digit']}"
            )

            st.write(
                f"**Confidence:** "
                f"{digit_result['confidence'] * 100:.2f}%"
            )

    if result["digits"]:
        selected_digit = st.selectbox(
            "View probability distribution for digit",
            options=list(range(len(result["digits"]))),
            format_func=lambda i: (
                f"Digit {i + 1}: "
                f"{result['digits'][i]['digit']}"
            ),
            key="probability_digit_select",
        )

        selected_result = result["digits"][selected_digit]

        probabilities = selected_result["probabilities"]

        probability_df = pd.DataFrame({
            "Digit": list(range(10)),
            "Probability (%)": probabilities * 100,
        })

        st.subheader("Probability Distribution")
        st.bar_chart(
            probability_df.set_index("Digit")
        )

        with st.expander("Show top 3 predictions"):
            top_df = pd.DataFrame(
                selected_result["top_predictions"]
            )
            top_df["Confidence"] = (
                top_df["Confidence"] * 100
            ).round(2).astype(str) + "%"

            st.dataframe(
                top_df,
                use_container_width=True,
                hide_index=True,
            )

    with st.expander("View preprocessed image"):
        st.image(
            result["binary_image"],
            caption="Binary image used for digit detection",
            width=350,
        )


# --------------------------------------------------
# 11. SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.title("🔮 DigitAI")
    st.caption("Handwritten Digit Recognition")

    st.divider()

    st.subheader("Model Information")
    st.write("Architecture: Convolutional Neural Network")
    st.write("Dataset: MNIST")
    st.write("Input size: 28 × 28")
    st.write("Classes: 0–9")
    st.write("Device: CPU")

    st.divider()

    st.subheader("History")
    st.write(
        f"Predictions made: {len(st.session_state.history)}"
    )

    if st.button("Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# --------------------------------------------------
# 12. MAIN PAGE
# --------------------------------------------------

st.title("DigitAI")
st.write("Draw a digit or upload an image to predict handwritten numbers.")

if model is None:
    st.error(
        "Trained model not found. Please check that this file exists:\n\n"
        "`models/simple_cnn_mnist.pth`"
    )
    st.stop()

tab_draw, tab_upload, tab_analytics, tab_about = st.tabs([
    "✍️ Draw a Digit",
    "📁 Upload an Image",
    "📊 Analytics",
    "ℹ️ About",
])


# --------------------------------------------------
# 13. DRAW TAB
# IMPORTANT: return_image_data=True fixes your error
# --------------------------------------------------

with tab_draw:
    st.subheader("Draw any digit from 0 to 9")

    canvas_col, button_col = st.columns([3, 1])

    with canvas_col:
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=14,
            stroke_color="#FFFFFF",
            background_color="#10172A",
            height=180,
            width=420,
            drawing_mode="freedraw",
            key="digitai_canvas",
            update_streamlit=True,
            point_display_radius=0,
            return_image_data=True,
        )

    with button_col:
        st.write("")
        st.write("")

        predict_button = st.button(
            "🔍 Predict Digit",
            use_container_width=True,
            type="primary",
        )

        clear_canvas = st.button(
            "🧹 Clear Canvas",
            use_container_width=True,
        )

    if clear_canvas:
        st.rerun()

    if predict_button:
        if canvas_result.image_data is None:
            st.warning("Please draw a digit first.")
        else:
            canvas_array = canvas_result.image_data.astype(np.uint8)

            if not np.any(canvas_array[:, :, 3] > 0):
                st.warning("Please draw a digit first.")
            else:
                with st.spinner("Recognizing your digit..."):
                    result = predict_number(canvas_array)

                display_result(result)

                if result["number"]:
                    add_to_history("Canvas", result)


# --------------------------------------------------
# 14. UPLOAD TAB
# --------------------------------------------------

with tab_upload:
    st.subheader("Upload a handwritten digit image")

    uploaded_file = st.file_uploader(
        "Choose a PNG, JPG, or JPEG image",
        type=["png", "jpg", "jpeg"],
    )

    if uploaded_file is not None:
        try:
            uploaded_image = Image.open(uploaded_file).convert("RGB")

            st.image(
                uploaded_image,
                caption="Uploaded Image",
                use_container_width=True,
            )

            if st.button(
                "🔍 Recognize Uploaded Image",
                type="primary",
            ):
                with st.spinner("Recognizing digits..."):
                    result = predict_number(
                        np.array(uploaded_image)
                    )

                display_result(result)

                if result["number"]:
                    add_to_history("Upload", result)

        except Exception as error:
            st.error(f"Could not process this image: {error}")


# --------------------------------------------------
# 15. ANALYTICS TAB
# --------------------------------------------------

with tab_analytics:
    st.subheader("Prediction Analytics")

    history = st.session_state.history

    if not history:
        st.info("Your prediction history will appear here.")
    else:
        history_df = pd.DataFrame(history)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Predictions", len(history_df))

        with col2:
            st.metric(
                "Average Confidence",
                f"{history_df['Confidence'].mean():.2f}%",
            )

        st.subheader("Prediction History")

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Predicted Numbers")
        st.bar_chart(
            history_df["Prediction"].value_counts()
        )

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download History as CSV",
            data=csv_data,
            file_name="digitai_prediction_history.csv",
            mime="text/csv",
        )


# --------------------------------------------------
# 16. ABOUT TAB
# --------------------------------------------------

with tab_about:
    st.subheader("About DigitAI")

    st.write(
        """
        DigitAI is a handwritten digit recognition application
        powered by a Convolutional Neural Network (CNN).

        **How it works:**

        1. Draw a digit or upload an image.
        2. The image is converted into a binary image.
        3. Individual digits are detected and resized to 28 × 28.
        4. The trained CNN predicts each digit.
        5. The app displays predictions and confidence scores.

        **Important:** The confidence score represents the model's
        estimated probability for its prediction. It does not
        guarantee that the prediction is correct.
        """
    )


# --------------------------------------------------
# 17. FOOTER
# --------------------------------------------------

st.divider()
st.caption("DigitAI | Handwritten Digit Recognition using CNN")
