# TODO — Plateforme d'analyse comportementale e-commerce

## Phase 1 — Sentiment Analysis (NLP)

### ✅ 1. Initialisation du projet

**Structure de répertoires à créer :**

```
trust-behaviours/
├── phase1_sentiment/
│   ├── data/           # données brutes et prétraitées
│   ├── notebooks/      # exploration + entraînement
│   ├── models/         # modèles exportés (.pt) + vocab.pkl
│   ├── api/            # FastAPI endpoint
│   └── README.md
├── requirements.txt
├── .gitignore
└── README.md
```

**`requirements.txt` à créer :**

```
torch
numpy
pandas
scikit-learn
mlflow
fastapi
uvicorn
```

**`.gitignore` à créer :**

```
phase1_sentiment/data/
phase1_sentiment/models/
__pycache__/
*.pyc
.ipynb_checkpoints/
mlruns/
```

---

### ❌ 2. Dataset

- [❌] Télécharger le dataset **Amazon Fine Food Reviews** sur Kaggle → [lien](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
  - Colonnes utiles : `Score` (1-5) et `Text`
  - Labels : Score 1-2 → négatif, 3 → neutre, 4-5 → positif
  - Placer le fichier `Reviews.csv` dans `phase1_sentiment/data/`
- [✅] -> Utilisation de kagglehub pour exploiter la donnée directemetn dans le cache de la lib

### 3. Notebook d'exploration (`phase1_sentiment/notebooks/01_eda.ipynb`)

- [✅] Charger et inspecter le dataset (`df.head()`, `df.info()`)
  - Le dataset contient 568 454 entrées
- [✅] Distribution des classes (vérifier déséquilibre entre positif/négatif/neutre)
  - Décision d'utiliser les class weights pour éviter les dataset trop légers
- [✅] Histogramme de la longueur des textes
  - Certains textes sont très long donc à voir s'il ne faut pas les ranger dans les anomalies et les exclure de notre dataset
- [✅] Afficher des exemples par classe
  - Amusant, mais pas de conclusion à tirer de ça

---

### 4. Notebook d'entraînement (`phase1_sentiment/notebooks/02_train.ipynb`)

**Prétraitement :**

- [ ] Nettoyage des textes (lowercase, suppression HTML, ponctuation)
- [ ] Tokenisation avec `torchtext` ou un vocabulaire custom
- [ ] Padding des séquences (`maxlen=200`)
- [ ] Split train/val/test (70/15/15)
- [ ] Créer un `Dataset` et `DataLoader` PyTorch

**Architecture PyTorch :**

```python
import torch
import torch.nn as nn

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, output_dim=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                            bidirectional=True, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        # Concaténer les 2 directions du dernier layer
        out = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(out)
```

**Boucle d'entraînement :**

```python
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
```

**Tracking MLflow :**

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("vocab_size", vocab_size)
    mlflow.log_param("embedding_dim", 128)
    mlflow.log_param("lstm_hidden_dim", 64)

    # après entraînement :
    mlflow.log_metric("val_accuracy", val_accuracy)
    mlflow.pytorch.log_model(model, "sentiment_model")
```

- [ ] Entraîner le modèle (objectif : val_accuracy > 85%)
- [ ] Sauvegarder le modèle dans `phase1_sentiment/models/sentiment_model.pt`

---

### 5. API FastAPI (`phase1_sentiment/api/main.py`)

```python
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
import pickle

app = FastAPI()

LABELS = ["negatif", "neutre", "positif"]

# Charger modèle et vocabulaire
model = torch.load("../models/sentiment_model.pt", map_location="cpu")
model.eval()
with open("../models/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

class TextInput(BaseModel):
    text: str

def preprocess(text: str, maxlen=200):
    tokens = text.lower().split()
    ids = [vocab.get(t, 0) for t in tokens][:maxlen]
    ids += [0] * (maxlen - len(ids))  # padding
    return torch.tensor([ids])

@app.post("/predict")
def predict(input: TextInput):
    with torch.no_grad():
        x = preprocess(input.text)
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0].tolist()
    label = LABELS[probs.index(max(probs))]
    return {
        "sentiment": label,
        "scores": {LABELS[i]: round(probs[i], 4) for i in range(3)}
    }
```

- [ ] Lancer l'API : `uvicorn main:app --reload`
- [ ] Tester : `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "This product is amazing!"}'`

---

### 6. README Phase 1 (`phase1_sentiment/README.md`)

- [ ] Décrire l'architecture du modèle
- [ ] Expliquer la théorie BPTT (Backpropagation Through Time)
- [ ] Documenter les métriques obtenues (accuracy, loss)
- [ ] Inclure un exemple d'appel API

---

## Ordre d'exécution recommandé

1. [ ] Créer la structure de répertoires
2. [ ] Créer `requirements.txt` et `.gitignore` à la racine
3. [ ] Télécharger le dataset et le placer dans `data/`
4. [ ] Écrire et exécuter `01_eda.ipynb`
5. [ ] Écrire et exécuter `02_train.ipynb`
6. [ ] Créer l'API FastAPI et la tester
7. [ ] Écrire `README.md` pour la Phase 1

---

## Vérification finale

- [ ] `mlflow ui` → les runs apparaissent avec les métriques
- [ ] `POST /predict` retourne bien un sentiment avec des scores
- [ ] val_accuracy > 85%
