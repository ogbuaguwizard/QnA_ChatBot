import os
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    BertTokenizer,
    BertForQuestionAnswering,
)
import streamlit as st

# Define model paths
MODEL_PATHS = {
    "GPT-2": "model/gpt2-finetuned",
    "T5": "model/T5-finetuned",
    "BERT": "model/BERT-finetuned",
}

# Load models and tokenizers
@st.cache_resource  # Cache models to avoid reloading on each interaction
def load_model_and_tokenizer(model_name):
    if model_name == "GPT-2":
        model = GPT2LMHeadModel.from_pretrained(MODEL_PATHS[model_name])
        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATHS[model_name])
    elif model_name == "T5":
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATHS[model_name])
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[model_name])
    elif model_name == "BERT":
        model = BertForQuestionAnswering.from_pretrained(MODEL_PATHS[model_name])
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATHS[model_name])
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model, tokenizer

# Streamlit Interface
st.title("QnA Model Demo")
st.subheader("Choose a model for generating answers:")

# Model selection
model_option = st.selectbox("Select Model", ["GPT-2", "T5", "BERT"])

# Load the selected model and tokenizer
model, tokenizer = load_model_and_tokenizer(model_option)

# Input
user_input = st.text_area("Enter your question:")

# Inference
if st.button("Generate Answer"):
    if model_option == "GPT-2":
        input_ids = tokenizer(f"User: {user_input} \nAI:", return_tensors="pt").input_ids
        outputs = model.generate(
            input_ids, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = response.split("AI:")[1].strip()
    elif model_option == "T5":
        inputs = tokenizer(user_input, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs.input_ids, max_length=128, num_beams=5, early_stopping=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    elif model_option == "BERT":
        question = user_input
        context = st.text_area("Enter the context for BERT to find an answer:")
        inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
        outputs = model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
        )
    else:
        answer = "Unsupported model selected."

    st.success(f"Answer ({model_option}): {answer}")
