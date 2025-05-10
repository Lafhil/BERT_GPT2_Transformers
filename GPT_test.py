from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2-finetuned")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-finetuned")

model.eval()

prompt = "NASA launched"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Génération
outputs = model.generate(
    **inputs,
    max_length=50,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    early_stopping=True
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nTexte généré :")
print(generated_text)
