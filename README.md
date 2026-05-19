# Détection de SMS indésirables avec le Deep Learning — AT&T

## En bref

| Élément | Valeur |
|---|---|
| **Problème** | Classification binaire ham/spam à partir du texte SMS |
| **Données** | 5 572 SMS annotés — 4 825 ham / 747 spam |
| **Approche** | Pipeline NLP + comparaison de 3 architectures Deep Learning |
| **Modèle retenu** | Embedding + GlobalAveragePooling1D |
| **F1-score** | **0,9448** |
| **Recall spam** | **0,9195** — plus de 9 spams sur 10 détectés |

---

## Contexte métier

**AT&T** souhaite automatiser la détection des SMS indésirables (spam) afin de protéger ses utilisateurs.

L'objectif est de construire un modèle capable de classifier chaque SMS comme **spam** ou **ham** (message normal), en se basant **uniquement sur le contenu textuel** du message.

Un tel modèle peut contribuer à :

- filtrer automatiquement les messages suspects
- réduire les faux messages promotionnels ou frauduleux
- améliorer l'expérience utilisateur
- assister un système plus large de modération ou de pré-classification

Dans ce contexte, l'objectif n'est pas seulement d'obtenir de bonnes performances, mais aussi de construire une solution **simple, cohérente et défendable**.

---

## Question centrale

> Peut-on détecter efficacement un SMS spam à partir du seul texte du message ?

**Réponse : oui.** Le modèle retenu atteint un F1-score de 0,9448 et détecte 9 spams sur 10 avec seulement 4 faux positifs sur 1 115 messages testés.

---

## Données

Le projet repose sur le fichier `data/spam.csv`, qui contient **5 572 SMS** annotés selon deux classes :

| Classe | Code | Nombre | Part |
|---|---|---|---|
| **ham** | 0 | 4 825 | 86,6 % |
| **spam** | 1 | 747 | 13,4 % |

Le dataset est **déséquilibré**, ce qui justifie l'utilisation du **F1-score** plutôt que de l'accuracy.

Après nettoyage, seules deux colonnes sont conservées :

- `label` : la classe du message (encodée 0/1)
- `text` : le contenu du SMS

---

## Méthodologie

### 1. Préparation des données

| Étape | Détail |
|---|---|
| Nettoyage | Suppression des colonnes `Unnamed: 2, 3, 4` |
| Renommage | `v1 → label`, `v2 → text` |
| Encodage labels | `ham → 0`, `spam → 1` |
| Split | `train_test_split(test_size=0.2, stratify=y)` |
| Tokenization | `Tokenizer(num_words=5000, oov_token='<OOV>')` |
| Séquences | `texts_to_sequences()` — chaque mot → entier |
| Padding | `pad_sequences(maxlen=50, padding='post')` |

### 2. Paramètres clés

| Paramètre | Valeur | Justification |
|---|---|---|
| `num_words` | 5 000 | Couvre ~63 % du vocabulaire réel (7 919 mots) |
| `maxlen` | 50 | Longueur moyenne des SMS ~80 caractères ≈ 15-20 tokens |
| `embedding_dim` | 32 | Bon compromis expressivité / vitesse |
| `batch_size` | 32 | Standard pour ce volume |
| `epochs` max | 20 | Avec EarlyStopping (patience=3) |

### 3. Architectures testées

#### Modèle 1 — Embedding + GlobalAveragePooling1D (baseline)
```
Embedding(5000, 32) → GlobalAveragePooling1D → Dense(16, relu) → Dropout(0.3) → Dense(1, sigmoid)
```
Architecture simple, rapide à entraîner. L'embedding est appris from scratch, le GAP fait la moyenne des vecteurs sur toute la séquence.

#### Modèle 2 — Embedding + LSTM
```
Embedding(5000, 32) → LSTM(32) → Dropout(0.3) → Dense(16, relu) → Dense(1, sigmoid)
```
Architecture récurrente qui modélise l'ordre des mots. Plus complexe, mais pas toujours supérieur sur les textes courts.

#### Modèle 3 — Transfer Learning (NNLM via TensorFlow Hub)
```
Input(string) → NNLM-en-dim50 (pré-entraîné Google News) → Dropout(0.3) → Dense(16) → Dense(1, sigmoid)
```
Encodeur pré-entraîné sur des milliards de mots de Google News. Fine-tuning sur nos données SMS.

---

## Résultats

### Comparaison des modèles

| Modèle | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| **Embedding + GlobalAveragePooling1D** 🏆 | **0,9857** | **0,9716** | **0,9195** | **0,9448** |
| Embedding + LSTM | 0,9659 | 0,9237 | 0,8121 | 0,8643 |
| Transfer Learning (NNLM) | 0,9587 | 0,8601 | 0,8255 | 0,8425 |

### Matrice de confusion du modèle retenu

| | Prédit ham | Prédit spam |
|---|---:|---:|
| **Réel ham** | 962 ✓ | 4 (faux positifs) |
| **Réel spam** | 12 (faux négatifs) | 137 ✓ |

- **962 ham correctement classés** ✓
- **137 spams correctement détectés** ✓
- **4 faux positifs seulement** — expérience utilisateur préservée
- **12 spams manqués** — taux de fuite de 8,05 %

### Pourquoi le modèle simple gagne ?

Trois enseignements forts émergent de ce projet :

1. **Le LSTM apporte peu** sur des textes très courts (SMS ≈ 15-20 tokens). Les dépendances séquentielles longues que le LSTM est conçu pour capturer n'existent pas vraiment dans des SMS.

2. **Le Transfer Learning NNLM échoue** parce que son corpus source (Google News, articles formels) est trop éloigné du domaine cible (SMS informels avec abréviations, fautes, codes).

3. **Un modèle simple bien calibré sur des données bien préparées** surpasse souvent une architecture sophistiquée mal adaptée au problème.

---

## Choix méthodologiques à justifier

**Pourquoi F1-score et pas accuracy ?**
Sur un dataset déséquilibré (87 % ham / 13 % spam), un modèle naïf qui prédirait "ham" pour tous les messages obtiendrait déjà 86,6 % d'accuracy sans rien apprendre. Le F1-score combine precision et recall pour évaluer correctement la détection de la classe rare.

**Pourquoi `stratify=y` dans le split ?**
Pour garantir la même proportion ham/spam dans le train et le test. Sans stratification, le hasard pourrait créer un test set très biaisé qui rendrait l'évaluation peu fiable.

**Pourquoi entraîner le Tokenizer uniquement sur le train ?**
Pour éviter la **fuite de données** (data leakage). Si le Tokenizer voyait le vocabulaire du test, le modèle bénéficierait indirectement d'informations qu'il n'aurait pas eu en production.

**Pourquoi `tf_keras` pour le modèle 3 ?**
Les modèles TensorFlow Hub sont compatibles **Keras 2**. Or TensorFlow 2.16+ utilise Keras 3 par défaut. Il faut donc importer explicitement `tf_keras` pour reconstruire le modèle 3.

**Pourquoi EarlyStopping(patience=3, restore_best_weights=True) ?**
Cette configuration arrête l'entraînement quand la `val_loss` ne s'améliore plus pendant 3 époques, et restaure automatiquement les poids du **meilleur** epoch. Cela évite à la fois le sous-apprentissage et le surapprentissage sans chercher manuellement le bon nombre d'époques.

---

## Interprétation métier

Le modèle retenu peut servir de première brique pour :

- **Filtrer automatiquement** une partie des messages indésirables
- **Réduire l'exposition** des utilisateurs à des contenus non souhaités
- **Assister un système plus large** de modération ou de protection

Avec un **F1-score de 0,9448** et **plus de 9 spams sur 10 détectés**, ce modèle constitue une base solide et directement utilisable.

---

## Structure du projet

```text
.
├── data/
│   └── spam.csv                          ← 5 572 SMS annotés
├── notebooks/
│   └── spam_detector_final.ipynb         ← Notebook complet exécuté
├── .gitignore
├── README.md                             ← Ce fichier
├── requirements.txt                      ← Dépendances Python
└── revision.md                           ← Fiche de révision pour la soutenance
```

---

## Technologies

| Catégorie | Outils |
|---|---|
| Langage | Python 3.12 |
| Manipulation de données | pandas, NumPy |
| Visualisation | Matplotlib |
| Deep Learning | TensorFlow 2.21, Keras, tf_keras |
| Transfer Learning | TensorFlow Hub 0.16, NNLM-en-dim50 |
| NLP | Tokenizer Keras, pad_sequences |
| Évaluation | scikit-learn (classification_report, ConfusionMatrixDisplay) |
| Environnement | Jupyter Notebook |
| Versioning | Git, GitHub |

---

## Installation

### 1. Cloner le dépôt

```bash
git clone <URL_DU_REPO>
cd atm-spam-detector_Birane
```

### 2. Créer un environnement virtuel

**Windows :**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux :**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer le notebook

```bash
jupyter notebook notebooks/spam_detector_final.ipynb
```

---

## Limites du projet

- Le modèle utilise **uniquement le texte brut du message**
- Pas de contexte externe (expéditeur, fréquence, historique utilisateur)
- Le dataset reste de **taille modérée** (5 572 messages)
- Les performances observées dépendent du jeu de données utilisé
- Une bonne performance sur ce dataset **ne garantit pas une généralisation parfaite** à des données réelles plus variées
- Le seuil de décision est fixé à 0,5 — il pourrait être ajusté selon le coût métier des erreurs

---

## Pistes d'amélioration

Plusieurs améliorations pourraient être envisagées dans une version plus avancée :

1. **Nettoyage textuel plus poussé** : suppression de caractères spéciaux, normalisation des nombres, traitement des emoji
2. **Embeddings pré-entraînés spécifiques au domaine** : GloVe, FastText, ou un modèle SMS-spam dédié
3. **Tuning du seuil de décision** : ajuster selon le coût métier (privilégier recall vs precision)
4. **Class weight balanced** : compenser explicitement le déséquilibre pendant l'entraînement
5. **Architectures modernes** : Transformers (BERT, DistilBERT) avec fine-tuning sur SMS
6. **Validation croisée** : 5-fold stratifié pour valider la robustesse du modèle
7. **Ajout d'une courbe Precision-Recall** : plus informative que ROC sur classes déséquilibrées
8. **Intégration dans une API** : démonstration applicative avec FastAPI ou Streamlit

---

## Auteur

Projet préparé par **Birane WANE** dans le cadre du Bloc 4 JEDHA Bootcamp — Deep Learning.
