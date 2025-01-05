import os
import requests
import zipfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import streamlit as st
from time import sleep

# Function to download and extract models
def download_and_extract(url, output_dir, zip_file_name, retries=3, delay=5):
    """
    Download and extract a zip file if the folder doesn't already exist.
    Includes retry logic for large files.
    """
    if not os.path.exists(output_dir):
        with st.spinner(f"Downloading and extracting {zip_file_name}..."):
            for i in range(retries):
                try:
                    # Download file in chunks
                    response = requests.get(url, stream=True)
                    response.raise_for_status()  # Check if request was successful
                    with open(zip_file_name, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)
                    # Extract the zip file
                    with zipfile.ZipFile(zip_file_name, "r") as zip_ref:
                        zip_ref.extractall(output_dir)
                    # Remove the zip file after extraction
                    os.remove(zip_file_name)
                    print(f"Downloaded and extracted {zip_file_name} successfully.")
                    break
                except requests.exceptions.RequestException as e:
                    print(f"Download failed: {e}")
                    if i < retries - 1:
                        print(f"Retrying in {delay} seconds...")
                        sleep(delay)
                    else:
                        print(f"Failed to download after {retries} attempts.")
                except zipfile.BadZipFile as e:
                    print(f"Error extracting {zip_file_name}: {e}")
                    break

# Google Drive direct download links for the models
model_links = {
    "GPT-2": {
        "url": "https://drive.google.com/uc?id=1EAlQFw9wxi0-9WVEU6t9GlsbAirwms4Z&export=download",
        "output_dir": "./models/gpt2-finetuned",
        "zip_file_name": "gpt2-finetuned.zip",
    },
    # Add other models here (e.g., T5, BART) if needed.
}

# Download and extract all models
for model_name, model_info in model_links.items():
    download_and_extract(model_info["url"], model_info["output_dir"], model_info["zip_file_name"])

# Load models and tokenizers
models = {
    "GPT-2": AutoModelForCausalLM.from_pretrained(model_links["GPT-2"]["output_dir"]),
    # "T5": AutoModelForSeq2SeqLM.from_pretrained(model_links["T5"]["output_dir"]),
    # "BART": AutoModelForSeq2SeqLM.from_pretrained(model_links["BART"]["output_dir"]),
}

tokenizers = {
    "GPT-2": AutoTokenizer.from_pretrained(model_links["GPT-2"]["output_dir"]),
    # "T5": AutoTokenizer.from_pretrained(model_links["T5"]["output_dir"]),
    # "BART": AutoTokenizer.from_pretrained(model_links["BART"]["output_dir"]),
}

# Streamlit Interface
st.title("QnA Model Demo")
st.subheader("Choose a model for generating answers:")

# Model selection
model_option = st.selectbox("Select Model", ["GPT-2"])

# Input
user_input = st.text_area("Enter your question:")

# Inference
if st.button("Generate Answer"):
    model = models[model_option]
    tokenizer = tokenizers[model_option]
    
    inputs = tokenizer.encode(user_input, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=128, num_return_sequences=1, num_beams=5, early_stopping=True)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    st.success(f"Answer ({model_option}): {answer}")
