from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import torch
import accelerate
import transformers
print("transformers:", transformers.__version__)
print("accelerate:", accelerate.__version__)
# print("datasets:", datasets.__version__)
print("torch:", torch.__version__)
# Vérifier GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Le GPU utilisé est : {torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'CPU'}")

# Charger tokenizer et modèle GPT-2
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 n'a pas de pad token

model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))
model.to(device)

# Charger le dataset AG News (juste pour le texte)
dataset = load_dataset("ag_news", split="train[:5%]")  # 5% pour l'exemple rapide

# Prétraitement des textes
def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])


# Préparer le data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned",
    overwrite_output_dir=True,
    # evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_steps=50,
    fp16=torch.cuda.is_available(),  # entraînement en float16 si GPU
    report_to="none",  # évite erreur avec wandb ou autres
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # ici même dataset pour demo
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Entraînement
trainer.train()

# Sauvegarde
trainer.save_model("gpt2-finetuned")
tokenizer.save_pretrained("gpt2-finetuned")
