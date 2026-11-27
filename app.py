import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Page configuration
st.set_page_config(page_title="Phishing & Spam Triage", page_icon="🛡️")

MODEL_NAME = "isom5160/phishing_finetuned_toy"

DEMO_TEXTS = [
    "URGENT! You have won a $1000 prize. Call 09061701461 now to claim.",
    "Ok lar... Joking wif u oni...",
    "FreeMsg: Reply YES to subscribe to our weekly ringtone service. Std rates apply.",
    "Nah I don't think he goes to usf, he lives around here though",
]


@st.cache_resource
def load_model():
    """Load and cache the tokenizer and model on CPU."""
    device = torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.to(device)
    model.eval()
    return tokenizer, model


def predict(text, tokenizer, model):
    """Return (label, confidence) using CPU inference."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())

    label = "Phishing" if pred_id == 1 else "Legit"
    confidence = float(probs[pred_id].item())
    return label, confidence


# --- UI Layout ---
st.title("🛡️ Phishing / Spam Triage")
st.warning("⚠️ **Safety Notice:** Human review is required. Confidence scores are NOT proof of phishing. Fine-tuned on an `sms_spam` toy slice — not for production detection.")

# Load model with spinner
with st.spinner("Loading model..."):
    tokenizer, model = load_model()

# Optional: Quick-fill demo examples
selected_demo = st.selectbox(
    "Choose a sample message (or type your own below):",
    options=["Custom message..."] + DEMO_TEXTS,
)

default_text = "" if selected_demo == "Custom message..." else selected_demo

user_text = st.text_area(
    "Message to classify:",
    value=default_text,
    height=120,
    placeholder="Paste an SMS or short email snippet here...",
)

if st.button("Classify Message", type="primary"):
    if not user_text.strip():
        st.error("Please enter a message to classify.")
    else:
        label, conf = predict(user_text, tokenizer, model)
        
        st.subheader("Result")
        if label == "Phishing":
            st.error(f"🚨 **{label}** (Confidence: {conf:.2%})")
        else:
            st.success(f"✅ **{label}** (Confidence: {conf:.2%})")

