from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Vérification de la disponibilité de CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle et le tokenizer sauvegardés
model = BertForSequenceClassification.from_pretrained("BERT_model").to(device)
tokenizer = BertTokenizer.from_pretrained("BERT_model")

# Tester une nouvelle phrase
def test_model(text):
    model.eval()  # Mode évaluation
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    # Prédiction sans calcul de gradients
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits.softmax(dim=1)
        predicted_class = probs.argmax().item()
        print(probs)
    label_names = ["World", "Sports", "Business", "Sci/Tech"]
    print(f"Texte : {text}")
    print(f"Classe prédite : {label_names[predicted_class]}")

# Tester avec une nouvelle phrase
test_model("The stock market is showing signs of recovery.")
test_model("The World Cup 2022 was amazing!")
test_model("NASA discovered water on Mars.")
test_model("iam a student in Faculty of Sciences of Rabat")
test_model("The stock market is showing signs of recovery.")
test_model("NASA just launched a new mission to Mars.")
test_model("The latest football match was thrilling!")
test_model("The global economy is recovering after the pandemic.")
test_model("the best football players in the world is Cristiano.")
test_model("Scientists have discovered a new way to produce renewable energy.")
test_model("The United Nations held a meeting to address global climate change.")
test_model("Technologie companies are leading the stock market growth.")
test_model("The Tokyo Olympics brought countries together to compete.")
test_model("Artificial intelligence is transforming industries worldwide.")
test_model("A new peace treaty was signed between two neighboring countries.")



