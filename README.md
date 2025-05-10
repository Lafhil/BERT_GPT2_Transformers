# 🧠 BERT News Classifier - AG News

Ce projet implémente un modèle de classification de texte basé sur **BERT** pour catégoriser des articles d’actualité en quatre classes : `World`, `Sports`, `Business` et `Sci/Tech`. Il utilise le dataset **AG News**.

## 📂 Structure du projet

```
bert_news_classifier/
│
├── BERT_code.py          # Script d'entraînement et de test du modèle
├── BERT_test.py          # Script de test du modèle sauvegardé
├── BERT_model/            # Dossier contenant le modèle entraîné et le tokenizer
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── special_tokens_map.json
├── optimizer.pt           # État de l'optimiseur sauvegardé
├── epoch.txt              # Dernière époque sauvegardée
├── README.md              # Fichier de documentation
└── requirements.txt       # Bibliothèques nécessaires (à créer)
```

## 🛠️ Installation

1. Cloner ce dépôt ou copier les fichiers dans un dossier.
2. Installer les bibliothèques nécessaires :

```bash
pip install transformers datasets torch tqdm
```

## 📦 Données utilisées

Le dataset [AG News](https://huggingface.co/datasets/ag_news) est automatiquement téléchargé avec la bibliothèque `datasets`.

- **Taille** : 120,000 exemples d'entraînement
- **Classes** : `World`, `Sports`, `Business`, `Sci/Tech`

## 🧪 Utilisation

### Entraînement

Le script `BERT_model.py` entraîne le modèle sur 3 époques, puis le sauvegarde :

```bash
python BERT_model.py
```

### Test rapide

À la fin du script, plusieurs prédictions sont effectuées sur des textes personnalisés, avec la classe prédite affichée.

```python
Texte : NASA discovered water on Mars.
Classe prédite : Sci/Tech
```

### Recharger le modèle sauvegardé

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model = BertForSequenceClassification.from_pretrained("BERT_model")
tokenizer = BertTokenizer.from_pretrained("BERT_model")
```

## 📈 Résultats

- **Loss d'entraînement** :
  - Époque 1 : 0.2302
  - Époque 2 : 0.1517
  - Époque 3 : 0.1161
- **Exemples de prédictions correctes** :
  - "The stock market is showing signs of recovery." → `Business`
  - "The World Cup 2022 was amazing!" → `Sports`
  - "NASA discovered water on Mars." → `Sci/Tech`
  - "The United Nations held a meeting..." → `World`

## ✅ Fichiers à inclure

- `BERT_model.py`
- `README.md`
- `BERT_model/` (tout le dossier)
- `optimizer.pt`
- `epoch.txt`
- `requirements.txt` (à générer via `pip freeze > requirements.txt`)

## ✨ Améliorations possibles

- Évaluer le modèle sur le test set (accuracy, f1-score…)
- Ajouter une interface graphique ou une API Flask
- Ajouter le support multi-langues

---



Manssour Youssef 
lafhil ouadie  

