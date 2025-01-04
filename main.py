import streamlit as st
import torch
from transformers import BertTokenizer
from mymodel import BertForMedicalClassification  # Replace with your actual model import
import os
from PyPDF2 import PdfReader


# Functions for extracting key info and generating actions
def extract_key_info(text):
    return "Age and key symptoms extracted from text"

def generate_actions(department, key_info, urgency):
    if urgency == 'Urgent':
        return "Schedule urgent appointment; Complete necessary tests"
    return "Schedule routine appointment; Follow standard pathway"

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def load_model_and_predict(model_path, referral_text):
    # Load model, tokenizer, and mappings
    tokenizer = BertTokenizer.from_pretrained(model_path)
    mappings = torch.load(os.path.join(model_path, 'label_mappings.pt'))
    model = BertForMedicalClassification.from_pretrained(
        model_path,
        num_departments=len(mappings['department_to_id'])
    )
    
    # Tokenize input
    inputs = tokenizer(
        referral_text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=512
    )
    
    # Remove 'token_type_ids' from inputs
    inputs = {k: v for k, v in inputs.items() if k in ['input_ids', 'attention_mask']}  

    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Get department prediction
        dept_probs = torch.nn.functional.softmax(outputs['department_logits'], dim=-1)
        dept_pred = torch.argmax(dept_probs, dim=1).item()
        department = mappings['id_to_department'][dept_pred]
        
        # Get urgency prediction
        urgency_probs = torch.nn.functional.softmax(outputs['urgency_logits'], dim=-1)
        urgency_pred = torch.argmax(urgency_probs, dim=1).item()
        urgency_label = 'Urgent' if urgency_pred == 1 else 'Routine'

    # Extract key information (you can enhance this based on your needs)
    key_info = extract_key_info(referral_text)
    suggested_actions = generate_actions(department, key_info, urgency_label)

    # Return predictions
    return department, urgency_label, key_info, suggested_actions

# Streamlit UI
st.title("Medical Referral Triage Predictor")

# File upload
uploaded_file = st.file_uploader("Upload a referral PDF", type=["pdf"])

if uploaded_file is not None:
    # Extract text from the uploaded PDF
    referral_text = extract_text_from_pdf(uploaded_file)
    st.write("### Extracted Text from Referral:")
    st.write(referral_text)

    # Load model and predict
    model_save_path = "C://Users//khiza//OneDrive//Desktop//project naeem_ahad//project//modell"  # Adjust your path accordingly
    department, urgency_label, key_info, suggested_actions = load_model_and_predict(model_save_path, referral_text)
    
    # Display results
    st.write("## Prediction Results:")
    st.write(f"**Department:** {department}")
    st.write(f"**Urgency:** {urgency_label}")
    st.write(f"**Key Extracted Info:** {key_info}")
    st.write(f"**Suggested Actions:** {suggested_actions}")
