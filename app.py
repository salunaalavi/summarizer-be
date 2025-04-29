import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "salunaalavi/bert-based-summarize"

model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

def summarize(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    summary_ids = model.generate(**inputs, max_new_tokens=128)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

gr.Interface(fn=summarize, inputs="text", outputs="text").launch()