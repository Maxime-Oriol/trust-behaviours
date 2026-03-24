# plateforme d'analyse comportementale e-commerce

**Pourquoi ce sujet ?**
Il recoupe exactement les cas d'usage de l'offre que tu vises (anomalie prix, sentiment client/vendeur, prédiction délais), il est réalisable solo, et les données sont disponibles publiquement (Amazon Reviews, Olist, Kaggle datasets e-commerce). Tu peux le publier sur GitHub + une démo live.

---

## Architecture évolutive en 4 phases

### Phase 1 — Sentiment Analysis (NLP)

_Modèle custom PyTorch, pas de fine-tuning de BERT au départ_

- Dataset : Amazon Product Reviews ou avis Trustpilot scrapés
- Architecture : Embedding layer → LSTM bidirectionnel → Dense
- Objectif : classifier le sentiment (positif / négatif / neutre) sur des échanges client
- Théorie dominante : **réseaux récurrents, backpropagation through time (BPTT)**
- Output : API FastAPI qui prend un texte et retourne un score de sentiment

### Phase 2 — Détection d'anomalies sur les prix

_Autoencoder PyTorch_

- Dataset : historique de prix produits (Kaggle / données simulées réalistes)
- Architecture : Autoencoder convolutif ou dense — l'anomalie = erreur de reconstruction élevée
- Objectif : détecter des comportements vendeurs suspects (price gouging, flash drop)
- Théorie dominante : **théorie de l'information, reconstruction loss, distribution latente**
- Output : score d'anomalie par produit/vendeur sur une période glissante

### Phase 3 — Prédiction de délais de livraison (Time Series)

_LSTM puis Transformer_

- Dataset : Olist Brazilian E-Commerce (public, riche, réaliste)
- Architecture V1 : LSTM many-to-one → prédit le délai en jours
- Architecture V2 : Temporal Fusion Transformer (TFT) — état de l'art sur les time series
- Théorie dominante : **séquences temporelles, attention mechanism, gating**
- Output : intervalle de confiance sur le délai prévu par commande

### Phase 4 — Agent IA orchestrant les 3 modèles

_Ce que tu sais déjà faire_

- Un agent ReAct qui interroge les 3 modèles selon le contexte
- Exemple : "Ce vendeur est-il à risque ?" → appel anomalie + sentiment en parallèle
- Théorie dominante : **RL / planification, tool use, chain-of-thought**
- Output : dashboard de scoring vendeur avec explainability

---

## Ce que ça te donne concrètement

- **4 modèles custom PyTorch** entraînés sur des données réelles
- **4 théories mathématiques différentes** documentées (BPTT, théorie de l'info, attention, RL)
- Un dépôt GitHub structuré avec notebooks, modèles exportés, métriques
- Une démo déployée (Hugging Face Spaces ou Streamlit Cloud — gratuit)
- Un article Medium / blog par phase → visibilité + preuve de compréhension théorique

---

## Par où commencer demain matin

1. Fork le dataset Olist sur Kaggle (il couvre les 4 cas d'usage)
2. Setup : Python + PyTorch + MLflow pour tracker tes expériences dès le début
3. Commence par la Phase 1 — c'est la plus rapide à montrer et la plus lisible pour un recruteur non technique
