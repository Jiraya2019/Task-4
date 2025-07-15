# Task 4 - Generative Text Model using GPT-2

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

prompt = "The importance of artificial intelligence in modern technology"
inputs = tokenizer.encode(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        top_k=50,
        top_p=0.95,
        temperature=0.7,
        do_sample=True,
        early_stopping=True
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text
