from transformers import AutoTokenizer

text = "Generative AI is amazing!"

bert = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt = AutoTokenizer.from_pretrained("gpt2")

bert_tokens = bert.tokenize(text)
gpt_tokens = gpt.tokenize(text)

print("Original Text:", text)

print("\nBERT Tokens:")
print(bert_tokens)

print("\nGPT Tokens:")
print(gpt_tokens)
