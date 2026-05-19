# Fiche de révision — Bloc 4 : Deep Learning NLP

## Projet : Détecteur de SMS Spam — AT&T

---

## 1. Concepts fondamentaux abordés

### Classification binaire de texte
- Objectif : prédire si un SMS est **spam (1)** ou **ham (0)**
- Entrée : texte brut → sortie : probabilité entre 0 et 1
- Seuil de décision : 0.5 (probabilité ≥ 0.5 → spam)

### Dataset déséquilibré
- 4 825 ham (≈ 87 %) vs 747 spam (≈ 13 %)
- **Conséquence** : l'accuracy seule est trompeuse
- Un modèle qui prédit toujours "ham" aurait déjà 86,6 % d'accuracy sans rien apprendre
- **Solution** : utiliser Precision, Recall et **F1-score** comme métriques principales

---

## 2. Pipeline de prétraitement NLP

### Étapes dans l'ordre

1. **Nettoyage** : suppression des colonnes inutiles, renommage
2. **Encodage des labels** : `ham → 0`, `spam → 1`
3. **Split train/test** : 80/20 avec `stratify=y` pour conserver les proportions
4. **Tokenization** : `Tokenizer(num_words=5000, oov_token='<OOV>')`
   - Apprendre le vocabulaire **uniquement sur le train** (éviter la fuite de données)
5. **Séquences** : `texts_to_sequences()` — chaque mot devient un entier
6. **Padding** : `pad_sequences(maxlen=50, padding='post', truncating='post')`
   - Uniformise la longueur de toutes les séquences

### Paramètres retenus

| Paramètre | Valeur | Justification |
|---|---|---|
| `num_words` | 5 000 | Couvre ~63 % du vocabulaire réel (7 919 mots) |
| `maxlen` | 50 | Longueur moyenne des SMS : 80 caractères ≈ 15-20 tokens |
| `embedding_dim` | 32 | Bon compromis expressivité / vitesse |

---

## 3. Architectures de deep learning

### Modèle 1 — Embedding + GlobalAveragePooling1D (BASELINE) 🏆
```
Embedding(5000, 32) → GlobalAveragePooling1D → Dense(16, relu) → Dropout(0.3) → Dense(1, sigmoid)
```
- **GlobalAveragePooling1D** : fait la moyenne de tous les vecteurs → représentation globale du message
- Rapide, simple, très efficace sur texte court
- **Résultat : F1 = 0,9448 ← MEILLEUR MODÈLE**

### Modèle 2 — Embedding + LSTM
```
Embedding(5000, 32) → LSTM(32) → Dropout(0.3) → Dense(16, relu) → Dense(1, sigmoid)
```
- **LSTM** : modélise les dépendances séquentielles entre les mots
- Plus lent, plus complexe, mais pas meilleur ici
- **Résultat : F1 = 0,8643**
- Limite : les SMS sont courts → peu de dépendances longues à capturer

### Modèle 3 — Transfer Learning (NNLM via TensorFlow Hub)
```
Input(string) → NNLM-en-dim50 (pré-entraîné Google News) → Dropout(0.3) → Dense(16) → Dense(1, sigmoid)
```
- Embeddings pré-entraînés sur des milliards de mots
- **Résultat : F1 = 0,8425 ← moins bon**
- Explication : domaine trop différent (Google News vs SMS informels)

---

## 4. Résultats complets

| Modèle | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| **Embedding + GAP** 🏆 | **0,9857** | **0,9716** | **0,9195** | **0,9448** |
| Embedding + LSTM | 0,9659 | 0,9237 | 0,8121 | 0,8643 |
| Transfer Learning (NNLM) | 0,9587 | 0,8601 | 0,8255 | 0,8425 |

### Matrice de confusion — Modèle retenu

| | Prédit ham | Prédit spam |
|---|---:|---:|
| **Réel ham** | 962 ✓ | 4 (FP) |
| **Réel spam** | 12 (FN) | 137 ✓ |

- **962 ham correctement classés**
- **137 spams détectés** (sur 149)
- **4 faux positifs** seulement → expérience utilisateur préservée
- **12 faux négatifs** → 8,05 % de spams ratés

---

## 5. Métriques — définitions et formules

| Métrique | Formule | Ce qu'elle mesure |
|---|---|---|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Taux global de bonnes prédictions |
| **Precision** | TP/(TP+FP) | Parmi les messages classés spam, combien sont vraiment spam ? |
| **Recall** | TP/(TP+FN) | Parmi tous les vrais spams, combien ont été détectés ? |
| **F1-score** | 2×(P×R)/(P+R) | Compromis entre Precision et Recall |

**Dans ce contexte métier :**

- Le **Recall** est critique : manquer un spam (faux négatif) est coûteux
- La **Precision** protège l'expérience utilisateur : un ham classé spam frustre l'utilisateur
- Le **F1-score** combine les deux et est la métrique de référence

---

## 6. Concepts clés à retenir

### EarlyStopping
```python
EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
```
- Arrête l'entraînement si `val_loss` ne s'améliore plus pendant 3 époques
- `restore_best_weights=True` : restaure les poids du meilleur epoch
- Évite le surapprentissage automatiquement

### Dropout
- Désactive aléatoirement des neurones pendant l'entraînement
- Réduit le surapprentissage (overfitting)
- Taux de 0.3 = 30 % des neurones désactivés à chaque batch

### Stratified split
```python
train_test_split(X, y, test_size=0.2, stratify=y)
```
- Garantit la même proportion spam/ham dans train et test
- Indispensable sur un dataset déséquilibré

### Transfer Learning
- Réutiliser un modèle pré-entraîné sur de grandes données
- `trainable=True` → fine-tuning (adapter les poids pré-entraînés)
- `trainable=False` → feature extraction (geler les poids)
- **Attention** : le domaine source doit être proche du domaine cible

### Embedding
- Transforme chaque entier (token) en vecteur dense de taille fixe
- Permet au modèle d'apprendre des relations sémantiques entre les mots
- `Embedding(vocab_size, embedding_dim)` → matrice apprise pendant l'entraînement

### GlobalAveragePooling1D
- Prend la moyenne des vecteurs sur toute la dimension temporelle
- Transforme une séquence (longueur × dim) en un vecteur (dim)
- Beaucoup plus rapide que les RNN, souvent aussi efficace sur texte court

### LSTM (Long Short-Term Memory)
- Type de RNN avec des "portes" qui contrôlent ce que le réseau retient ou oublie
- Conçu pour capturer des dépendances longues dans des séquences
- Pertinent pour des textes longs, moins utile pour des SMS courts

---

## 7. Enseignements du projet

1. **La simplicité gagne** : un Embedding + GAP surpasse le LSTM et le transfer learning
2. **Le transfer learning n'est pas toujours meilleur** : la distance de domaine est cruciale
3. **Le F1-score est la bonne métrique** sur dataset déséquilibré
4. **Stratifier le split** est indispensable pour évaluer correctement
5. **EarlyStopping** permet d'éviter l'overfitting sans chercher manuellement le bon nombre d'époques
6. **Compatibilité bibliothèques** : TF Hub (Keras 2) / Keras 3 → utiliser `tf_keras` pour les modèles TF Hub

---

## 8. Questions potentielles et réponses

**Pourquoi avoir choisi le modèle simple plutôt que le LSTM ou le transfer learning ?**
Le modèle GAP obtient le meilleur F1 (0,9448) sur les 3 métriques (precision, recall, F1). Sur des SMS courts (~15-20 tokens), le LSTM n'apporte pas de valeur car les dépendances séquentielles longues sont rares. Le transfer learning NNLM échoue parce que son corpus source (Google News) est trop éloigné du langage SMS informel.

**Pourquoi le F1-score et pas l'accuracy ?**
Le dataset est déséquilibré (87 % ham / 13 % spam). Un modèle qui prédit toujours "ham" aurait déjà 86,6 % d'accuracy sans rien apprendre. Le F1-score combine precision et recall pour évaluer correctement la détection de la classe minoritaire (spam).

**Pourquoi `stratify=y` dans le split ?**
Pour garantir la même proportion ham/spam dans le train et le test. Sans stratification, le hasard pourrait créer un test set non représentatif et rendre l'évaluation peu fiable.

**Pourquoi entraîner le Tokenizer uniquement sur le train ?**
Pour éviter la fuite de données. Si le Tokenizer voyait le vocabulaire du test, le modèle bénéficierait d'informations qu'il n'aurait pas en production.

**Qu'est-ce que `tf_keras` et pourquoi l'utiliser ?**
Les modèles TensorFlow Hub sont compatibles Keras 2. TensorFlow 2.16+ utilise Keras 3 par défaut. `tf_keras` est le wrapper Keras 2 fourni avec TF récent pour permettre la compatibilité avec hub.KerasLayer.

**Pourquoi `padding='post'` et pas `'pre'` ?**
- `'post'` : les zéros sont ajoutés à la fin → préserve le début du message
- `'pre'` : les zéros sont ajoutés au début → préserve la fin du message
- Pour les SMS, le début du message contient souvent l'information clé (sujet, intention), donc `'post'` est plus naturel.

**Pourquoi `embedding_dim = 32` ?**
- Trop petit (8-16) : pas assez expressif
- Trop grand (128-256) : surapprentissage sur un petit dataset
- 32 est un bon compromis pour un vocabulaire de 5 000 mots et 4 457 exemples d'entraînement.

**Pourquoi `EarlyStopping` ?**
Sans ce callback, le modèle pourrait continuer à s'améliorer sur le train tout en se dégradant sur la validation (surapprentissage). Avec `restore_best_weights=True`, on garantit que les poids finaux correspondent au meilleur point de la courbe validation.

**Qu'est-ce que le transfer learning et pourquoi a-t-il échoué ici ?**
Le transfer learning consiste à réutiliser un modèle pré-entraîné sur une grande quantité de données. Ici, NNLM a été entraîné sur Google News (articles formels, langage soutenu). Les SMS sont informels (abréviations, fautes, codes). La distance entre le domaine source et le domaine cible est trop grande pour que le pré-entraînement apporte un avantage.

**Comment ce modèle pourrait-il être amélioré ?**
- Nettoyage textuel plus poussé (suppression de ponctuation, normalisation des nombres)
- Tuning du seuil de décision (0,5 → ajuster selon le coût métier)
- Class weight balanced pour compenser le déséquilibre
- Tester un modèle Transformer (BERT, DistilBERT) avec fine-tuning
- Validation croisée 5-fold pour valider la robustesse

---

## 9. Commandes d'installation

```bash
# Environnement virtuel
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS/Linux

# Dépendances
pip install -r requirements.txt
```

### Fichier requirements.txt
```
pandas
numpy
matplotlib
scikit-learn
tensorflow==2.21.0
tensorflow-hub==0.16.1
tf-keras
setuptools==69.5.1
jupyter
nbformat
```

---

## 10. Synthèse en une phrase

> Un pipeline NLP simple (Tokenizer + Embedding + GAP), correctement préparé et évalué avec la bonne métrique, surpasse à la fois le LSTM et le transfer learning sur un problème de classification de SMS courts — confirmant que **la complexité architecturale n'est pas une fin en soi**.
