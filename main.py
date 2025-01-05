import os
import re
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from PyPDF2 import PdfReader
import streamlit as st

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    pdf_reader = PdfReader(pdf_file)
    text = " ".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
    return text

# Function to extract key patient information from referral text
def extract_key_info(text: str) -> str:
    """Extract key patient information from referral text."""
    age_match = re.search(r'Age[:\s]+(\d+)|(\d+)\s*years', text)
    dob_match = re.search(r'DOB:?\s*(\d{2}/\d{2}/\d{4})', text)
    symptoms = []
    key_conditions = [
        r'pain', r'bleeding', r'cyst', r'mass', r'endometriosis',
        r'dysmenorr?hoea', r'dyspareunia', r'bloating', r'lethargy'
    ]
    for condition in key_conditions:
        if re.search(condition, text, re.IGNORECASE):
            symptoms.append(condition)
    ca125_match = re.search(r'CA125:?\s*(\d+(?:\.\d+)?)', text)
    cyst_size_match = re.search(r'(\d+(?:\.\d+)?)\s*cm', text)

    info_parts = []
    if age_match:
        age = age_match.group(1) or age_match.group(2)
        info_parts.append(f"{age}yo")
    elif dob_match:
        info_parts.append(f"DOB: {dob_match.group(1)}")
    if symptoms:
        info_parts.append(", ".join(symptoms))
    if ca125_match:
        info_parts.append(f"CA125: {ca125_match.group(1)} U/mL")
    if cyst_size_match:
        info_parts.append(f"{cyst_size_match.group(1)}cm lesion")
    if 'post-menopaus' in text.lower() or 'postmenopaus' in text.lower():
        info_parts.append("post-menopausal")

    return " | ".join(info_parts)

# Function to generate appropriate actions
def generate_actions(department: str, key_info: str, urgency: str) -> str:
    """Generate appropriate actions based on department, key info, and urgency."""
    actions = []
    dept_actions = {
        'Endometriosis': [
            "Complete pelvic ultrasound if not done",
            "Check CA125 levels",
            "Consider pain management plan"
        ],
        'Gynaecology_General Gynecology': [
            "Arrange ultrasound if not completed",
            "Check routine blood tests"
        ],
        'Gynaecology': [
            "Review imaging results",
            "Check tumor markers"
        ],
        'Fertility': [
            "Complete hormonal profile",
            "Arrange ultrasound scan",
            "Partner semen analysis if applicable"
        ]
    }

    if department in dept_actions:
        actions.extend(dept_actions[department])
    if urgency.lower() == 'urgent':
        actions = ["**URGENT:** Book 2WW appointment"] + actions
        if 'ca125' in key_info.lower():
            actions.append("**Expedite CA125** if elevated")
        if 'cyst' in key_info.lower() or 'mass' in key_info.lower():
            actions.append("**Fast-track imaging review**")
    else:
        actions.append("Schedule routine appointment within **6 weeks**")
    if 'post-menopausal' in key_info.lower():
        actions.append("Consider **risk assessment** for gynecologic malignancy")
    if 'pain' in key_info.lower():
        actions.append("Review **analgesia requirements**")
    if 'endometriosis' in key_info.lower():
        actions.append("Consider **hormonal treatment options**")

    return "\n- " + "\n- ".join(actions)

# Function to load model and predict
def load_model_and_predict(hf_model_name, referral_text):
    """Load the trained model from Hugging Face and predict department, urgency, and key info."""
    tokenizer = DistilBertTokenizer.from_pretrained(hf_model_name)
    model = DistilBertForSequenceClassification.from_pretrained(hf_model_name)

    if not referral_text.strip():
        return None, None, None, "Error: Referral text is empty."

    inputs = tokenizer(
        referral_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}

    with torch.no_grad():
        outputs = model(**inputs)
        dept_probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        dept_pred = torch.argmax(dept_probs, dim=1).item()

        department = "Gynaecology" if dept_pred == 0 else "Endometriosis"  # Example mapping
        urgency_label = 'Urgent' if dept_pred == 1 else 'Routine'  # Modify as per your labels

    key_info = extract_key_info(referral_text)
    suggested_actions = generate_actions(department, key_info, urgency_label)

    return department, urgency_label, key_info, suggested_actions

# Streamlit App
st.title("🔍 Medical Referral Analysis Tool")
st.write("This tool extracts and analyzes key information from medical referral letters to provide actionable insights.")

hf_model_name = "khiziiii/aitriage"  # Replace with your Hugging Face model name
uploaded_file = st.file_uploader("📁 Upload Referral PDF", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting and analyzing referral..."):
        referral_text = extract_text_from_pdf(uploaded_file)
        department, urgency, key_info, actions = load_model_and_predict(hf_model_name, referral_text)

        if department is None:
            st.error(actions)
        else:
            st.success("Analysis completed!")
            st.subheader("**📝 Analysis Results**")
            st.write(f"**Department:** {department}")
            st.write(f"**Urgency:** {urgency}")
            st.write(f"**Key Extracted Information:** {key_info}")

            st.subheader("**📋 Suggested Actions**")
            st.markdown(actions)

            st.download_button(
                label="📥 Download Report",
                data=f"Department: {department}\nUrgency: {urgency}\nKey Info: {key_info}\nActions:\n{actions}",
                file_name="referral_analysis.txt",
                mime="text/plain"
            )
