from transformers import BertForSequenceClassification, BertTokenizer
import torch
from datasets import load_dataset  # Importer la fonction load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm  # Importer tqdm pour la barre de progression

# Vérification de la disponibilité de CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Le GPU utilisé est : {torch.cuda.get_device_name(torch.cuda.current_device()) if torch.cuda.is_available() else 'CPU'}")

# Charger le modèle et le tokenizer
model_name = "bert-base-uncased"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=4)
model.to(device)

tokenizer = BertTokenizer.from_pretrained(model_name)

# Charger le jeu de données AG News
dataset = load_dataset("ag_news")  # Charger le dataset correctement

# Fonction de tokenisation
def tokenize(example):
    return tokenizer(example['text'], padding='max_length', truncation=True, max_length=128)


# Appliquer la tokenisation et convertir en tensors
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# Diviser le jeu de données en ensembles d'entraînement et de test
train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

# Charger les données dans DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Définir la fonction de perte et l'optimiseur
loss_fn = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=5e-5)

# Fonction d'entraînement
def train_model(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):  # Utiliser tqdm pour la barre de progression
        optimizer.zero_grad()
        
        # Convertir les inputs en tensors et déplacer sur le GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Passer les données dans le modèle
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        # Calculer les gradients et mettre à jour les poids
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

# Entraînement sur le GPU (si disponible)
epochs = 3

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    # Entraîner le modèle
    train_loss = train_model(model, train_loader, optimizer, loss_fn, device)
    print(f"Train Loss: {train_loss:.4f}")
    
    # Sauvegarder après chaque époque
    model.save_pretrained("BERT_model")
    tokenizer.save_pretrained("BERT_model")
    torch.save(optimizer.state_dict(), "optimizer.pt")
    with open("epoch.txt", "w") as f:
        f.write(str(epoch + 1))  # Sauvegarder l'époque suivante

# Code de test après l'entraînement
print("\nTesting the model with a sample text...")

# Charger le modèle et le tokenizer sauvegardés
model = BertForSequenceClassification.from_pretrained("BERT_model").to(device)
tokenizer = BertTokenizer.from_pretrained("BERT_model")

model.eval()  # Mode évaluation
text = "NASA launched a new satellite to explore the solar system."

inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

# Prédiction sans calcul de gradients
with torch.no_grad():
    outputs = model(**inputs)
    probs = outputs.logits.softmax(dim=1)
    predicted_class = probs.argmax().item()

label_names = ["World", "Sports", "Business", "Sci/Tech"]
print(f"Texte : {text}")
print(f"Classe prédite : {label_names[predicted_class]}")
