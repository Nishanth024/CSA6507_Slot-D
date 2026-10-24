from transformers import AutoTokenizer, AutoModel
import torch

model_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

text = "Artificial intelligence is changing the world."

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state

print("Input Text:", text)
print("Embedding Shape:", embeddings.shape)
print("Embeddings:")
print(embeddings)
