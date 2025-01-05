import os
import requests
import zipfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

def download_and_extract(url, output_dir, zip_file_name):
    """
    Download and extract a zip file if the folder doesn't already exist.
    """
    if not os.path.exists(output_dir):
        print(f"Downloading and extracting {zip_file_name}...")
        response = requests.get(url, stream=True)
        with open(zip_file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
            zip_ref.extractall("./models")
        os.remove(zip_file_name)
    else:
        print(f"{output_dir} already exists.")

# Google Drive direct download links for the models
model_links = {
    "T5": {
        "url": "https://drive.google.com/uc?id=1EAlQFw9wxi0-9WVEU6t9GlsbAirwms4Z&export=download",
        "output_dir": "./models/t5_fine_tuned",
        "zip_file_name": "t5_model.zip",
    },
    "GPT-2": {
        "url": "https://drive.google.com/uc?id=YOUR_GPT2_MODEL_ID&export=download",
        "output_dir": "./models/gpt2_fine_tuned",
        "zip_file_name": "gpt2_model.zip",
    },
    "BART": {
        "url": "https://drive.google.com/uc?id=YOUR_BART_MODEL_ID&export=download",
        "output_dir": "./models/bart_fine_tuned",
        "zip_file_name": "bart_model.zip",
    },
}

# Download and extract all models
for model_name, model_info in model_links.items():
    download_and_extract(model_info["url"], model_info["output_dir"], model_info["zip_file_name"])

# Load models and tokenizers
models = {
    "T5": AutoModelForSeq2SeqLM.from_pretrained(model_links["T5"]["output_dir"]),
    "GPT-2": AutoModelForCausalLM.from_pretrained(model_links["GPT-2"]["output_dir"]),
    "BART": AutoModelForSeq2SeqLM.from_pretrained(model_links["BART"]["output_dir"]),
}

tokenizers = {
    "T5": AutoTokenizer.from_pretrained(model_links["T5"]["output_dir"]),
    "GPT-2": AutoTokenizer.from_pretrained(model_links["GPT-2"]["output_dir"]),
    "BART": AutoTokenizer.from_pretrained(model_links["BART"]["output_dir"]),
}

# Streamlit Interface
import streamlit as st

st.title("QnA Model Demo")
st.subheader("Choose a model for generating answers:")

# Model selection
model_option = st.selectbox("Select Model", ["T5", "GPT-2", "BART"])

# Input
user_input = st.text_area("Enter your question:")

# Inference
if st.button("Generate Answer"):
    model = models[model_option]
    tokenizer = tokenizers[model_option]
    
    if model_option in ["T5", "BART"]:
        inputs = tokenizer(user_input, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs.input_ids, max_length=128, num_beams=5, early_stopping=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elif model_option == "GPT-2":
        inputs = tokenizer.encode(user_input, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs, max_length=128, num_return_sequences=1, num_beams=5, early_stopping=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    st.success(f"Answer ({model_option}): {answer}")
