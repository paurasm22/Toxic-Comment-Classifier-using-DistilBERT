import streamlit as st
from transformers import pipeline
import re
import pandas as pd

# ----------------------------
# Load model once from Hugging Face
# ----------------------------
@st.cache_resource
def load_model():
    model_id = "Pau22/distilbert-toxic-model"
    return pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        top_k=1
    )

classifier = load_model()

# ----------------------------
# Clean incoming text
# ----------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", str(text))  # remove links
    text = text.encode("ascii", "ignore").decode()   # remove emojis
    return re.sub(r"\s+", " ", text).strip()

# ----------------------------
# Label Mapping
# ----------------------------
LABEL_MAP = {
    "LABEL_0": "Not Toxic",
    "LABEL_1": "Toxic"
}

# ----------------------------
# Sample Inputs
# ----------------------------
toxic_samples = [
    "You are the worst person ever.",
    "Shut up you idiot.",
    "You f*cking clown.",
    "Nobody likes you, go away."
]

non_toxic_samples = [
    "Have a lovely day!",
    "Thank you for your help!",
    "I appreciate your effort.",
    "This was very helpful, thanks!"
]

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Toxic Comment Classifier", layout="centered")

st.title("🧠 DistilBERT Toxic Comment Classifier")
st.write(
    "Detects whether a comment is **Toxic** or **Not Toxic** using a fine-tuned DistilBERT model.\n"
    "[🔗 View Deployed Hugging Face Space](https://huggingface.co/spaces/Pau22/Toxic_Comment_Classifier_using_DistilBERT)"
)

# Dropdowns
col1, col2 = st.columns(2)

with col1:
    toxic_choice = st.selectbox("Pick a toxic example (optional)", ["-- None --"] + toxic_samples)

with col2:
    non_toxic_choice = st.selectbox("Pick a non-toxic example (optional)", ["-- None --"] + non_toxic_samples)

# Auto-fill logic
user_text = ""
if toxic_choice != "-- None --":
    user_text = toxic_choice
elif non_toxic_choice != "-- None --":
    user_text = non_toxic_choice

# Text input
user_text = st.text_area("Enter your text for analysis", value=user_text, height=120)

# ----------------------------
# Prediction button
# ----------------------------
if st.button("Predict"):
    if not user_text.strip():
        st.warning("Please enter a comment.")
    else:
        cleaned = clean_text(user_text)
        raw = classifier(cleaned)

        # Normalize different HF output formats
        if isinstance(raw, list):
            if len(raw) > 0 and isinstance(raw[0], list):
                raw = raw[0][0]
            else:
                raw = raw[0]

        label = LABEL_MAP.get(raw["label"], raw["label"])
        score = float(raw["score"])

        st.subheader("Prediction Result")
        st.markdown(f"### **{label}**")
        st.write(f"🔥 Confidence Score: **{score:.3f}**")
        st.progress(score)

        with st.expander("Raw Model Output"):
            st.json(raw)

# ----------------------------
# Metrics Table
# ----------------------------
st.markdown("---")
st.subheader("📊 Model Evaluation Metrics")

metrics = {
    "Metric": ["Loss", "Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [0.1062, 0.9685, 0.8337, 0.8292, 0.8314],
}

df = pd.DataFrame(metrics)
st.table(df)

st.caption("Trained for 2 epochs on the Jigsaw Toxic Comment dataset using DistilBERT.")
