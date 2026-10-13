
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

text = "Generative AI is very interesting."

result = classifier(text)

print("Input:", text)
print("Output:", result)
