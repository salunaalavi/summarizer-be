from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = FastAPI()

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained("./summarizer_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("./summarizer_model")

class SummarizeRequest(BaseModel):
    text: str

@app.post("/summarize")
async def summarize(req: SummarizeRequest):
    inputs = tokenizer(req.text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    summary_ids = model.generate(**inputs, max_length=400, num_beams=10)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return {"summary": summary}
