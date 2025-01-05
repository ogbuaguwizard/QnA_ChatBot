import os
import requests
import zipfile
import streamlit as st
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

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

# Model configuration
model_links = {
    "GPT-2": {
        "url": "https://drive.google.com/uc?id=1EAlQFw9wxi0-9WVEU6t9GlsbAirwms4Z&export=download",
        "output_dir": "gpt2_fine_tuned",
        "zip_file_name": "gpt2-finetuned.zip",
    },
    # "T5": {
    #     "url": "https://drive.google.com/uc?id=1xyzT5model1234&export=download",
    #     "output_dir": "./models/t5_fine_tuned",
    #     "zip_file_name": "t5_model.zip",
    # },
    # "BERT": {
    #     "url": "https://drive.google.com/uc?id=1xyzBertmodel5678&export=download",
    #     "output_dir": "./models/bert_fine_tuned",
    #     "zip_file_name": "bert_model.zip",
    # },
}

# Download and extract all models
for model_name, model_info in model_links.items():
    download_and_extract(model_info["url"], model_info["output_dir"], model_info["zip_file_name"])

# Load the fine-tuned models and tokenizers
models = {
    "GPT-2": {
        "model": GPT2LMHeadModel.from_pretrained(model_links["GPT-2"]["output_dir"]),
        "tokenizer": GPT2Tokenizer.from_pretrained(model_links["GPT-2"]["output_dir"]),
    },
    # "T5": {
    #     "model": AutoModelForSeq2SeqLM.from_pretrained(model_links["T5"]["output_dir"]),
    #     "tokenizer": AutoTokenizer.from_pretrained(model_links["T5"]["output_dir"]),
    # },
    # "BERT": {
    #     "model": AutoModelForSeq2SeqLM.from_pretrained(model_links["BERT"]["output_dir"]),
    #     "tokenizer": AutoTokenizer.from_pretrained(model_links["BERT"]["output_dir"]),
    # },
}

# Chat function
def chat(model_name, input_text):
    model = models[model_name]["model"]
    tokenizer = models[model_name]["tokenizer"]

    if model_name == "GPT-2":
        input_ids = tokenizer(f"User: {input_text} \nAI:", return_tensors="pt").input_ids
        output = model.generate(input_ids, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response.split("AI:")[1].strip()

    elif model_name in ["T5", "BERT"]:
        input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).input_ids
        outputs = model.generate(input_ids, max_length=150, num_return_sequences=1, num_beams=5, early_stopping=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()

# Streamlit Interface
st.title("Fine-Tuned Language Models Demo")
st.subheader("Choose a model and ask your question:")

# Model selection
model_option = st.selectbox("Select Model", ["GPT-2", "T5", "BERT"])

# Input
user_input = st.text_area("Type your question here:")

# Generate Response
if st.button("Generate Response"):
    if user_input.strip():
        with st.spinner(f"Generating response using {model_option}..."):
            answer = chat(model_option, user_input)
        st.success(f"AI ({model_option}): {answer}")
    else:
        st.warning("Please enter a question or message.")
