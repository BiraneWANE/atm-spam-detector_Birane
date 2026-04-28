# Détection de SMS indésirables avec le deep learning

## En bref

- **Problème** : classifier automatiquement des SMS en `ham` ou `spam`
- **Données** : 5 572 messages annotés
- **Approche** : préparation de texte, tokenization, padding, puis comparaison de deux modèles de deep learning
- **Modèle retenu** : `Embedding + GlobalAveragePooling1D`

---

## Présentation du projet

Ce projet vise à construire un modèle de **classification binaire** capable de prédire si un SMS est un **spam** ou un **ham** (message normal) à partir du seul contenu textuel du message.

Le cas d’usage est simple et concret : automatiser la détection de SMS indésirables afin de réduire l’exposition des utilisateurs à des messages promotionnels, frauduleux ou non sollicités.

Le projet a été réalisé dans un cadre pédagogique, avec une approche volontairement sobre, reproductible et alignée avec les attendus d’un projet de deep learning appliqué au texte.

---

## Contexte métier

La détection automatique de spam est un problème fréquent dans les télécommunications et les services numériques.

Un tel modèle peut contribuer à :

- filtrer automatiquement les messages suspects ;
- réduire les faux messages promotionnels ou frauduleux ;
- améliorer l’expérience utilisateur ;
- alimenter un système de pré-classification ou de modération.

Dans ce contexte, l’enjeu n’est pas seulement d’obtenir de bonnes performances, mais aussi de construire une solution **simple, cohérente et défendable**.

---

## Objectifs du projet

L’objectif principal est de répondre à la question suivante :

> Peut-on détecter efficacement un SMS spam à partir du seul texte du message ?

Pour y répondre, le projet suit les étapes suivantes :

1. charger et explorer les données ;
2. préparer les messages textuels ;
3. transformer le texte en séquences numériques ;
4. entraîner plusieurs modèles de deep learning ;
5. comparer les performances ;
6. retenir le modèle le plus pertinent.

---

## Données utilisées

Le projet repose sur le fichier `spam.csv`, qui contient **5 572 SMS** annotés selon deux classes :

- **ham** : message normal
- **spam** : message indésirable

Après nettoyage, seules deux colonnes sont conservées :

- `label` : la classe du message
- `text` : le contenu du SMS

### Répartition des classes

- **ham** : 4 825 messages
- **spam** : 747 messages

Le dataset est donc **déséquilibré**, ce qui justifie l’utilisation de plusieurs métriques d’évaluation au-delà de la seule accuracy.

---

## Méthodologie

### 1. Préparation des données

Les principales étapes de préparation sont les suivantes :

- suppression des colonnes inutiles ;
- renommage des colonnes utiles ;
- encodage des labels :
  - `ham -> 0`
  - `spam -> 1`
- séparation entre variables explicatives (`X`) et variable cible (`y`) ;
- découpage en jeu d’entraînement et jeu de test avec `train_test_split` ;
- tokenization du texte ;
- transformation des SMS en séquences numériques ;
- padding des séquences pour uniformiser leur longueur.

### 2. Modèles testés

Deux architectures de deep learning ont été comparées :

#### Modèle 1 — Embedding + GlobalAveragePooling1D
Ce premier modèle constitue une base simple, légère et efficace pour la classification de texte court.

#### Modèle 2 — Embedding + LSTM
Ce second modèle permet de prendre davantage en compte l’ordre des mots dans la séquence.

---

## Résultats clés

Les performances ont été évaluées à l’aide des métriques suivantes :

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

### Comparaison des modèles

| Modèle | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Embedding + GlobalAveragePooling1D | 0.9785 | 0.9630 | 0.8725 | 0.9155 |
| Embedding + LSTM | 0.9776 | 0.9697 | 0.8591 | 0.9110 |

### Modèle retenu

Le modèle **Embedding + GlobalAveragePooling1D** a été retenu comme modèle principal.

Il obtient les meilleures performances globales sur ce dataset, avec le meilleur compromis entre :

- simplicité ;
- vitesse d’entraînement ;
- lisibilité ;
- performance globale.

Le modèle LSTM présente une précision légèrement supérieure, mais son recall et son F1-score restent légèrement inférieurs dans cette configuration.

---

## Principaux enseignements

Ce projet met en évidence plusieurs points importants :

- un pipeline simple et bien structuré peut déjà produire de très bons résultats sur un problème de classification de texte court ;
- un modèle plus complexe n’est pas automatiquement meilleur ;
- dans un dataset déséquilibré, l’accuracy seule ne suffit pas ;
- un modèle plus simple peut être préférable lorsqu’il reste plus facile à entraîner, à expliquer et à maintenir.

---

## Interprétation métier

Les résultats obtenus montrent qu’il est possible de détecter efficacement une grande partie des SMS spam à partir du seul texte du message.

D’un point de vue métier, cela signifie qu’un tel modèle peut servir de première brique pour :

- filtrer automatiquement une partie des messages indésirables ;
- réduire l’exposition des utilisateurs à des contenus non souhaités ;
- assister un système plus large de modération ou de protection.

Le modèle retenu ne constitue pas à lui seul une solution industrielle complète, mais il fournit une base solide et cohérente pour un cas d’usage réel.

---

## Structure du dépôt

```text
.
├── data/
│   └── spam.csv
├── notebooks/
│   ├── spam_detector_clean.ipynb
│   └── spam_detector_executed.ipynb
├── .gitignore
├── README.md
└── requirements.txt