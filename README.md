# Détection de SMS indésirables avec le deep learning

## Présentation du projet

Ce projet répond à un cas d'usage simple et concret : **classer automatiquement des SMS en _ham_ ou _spam_ à partir du seul contenu textuel**.

Le contexte métier est celui d'un opérateur télécom souhaitant réduire l'exposition de ses utilisateurs aux messages indésirables. Le sujet est traité comme un problème de **classification binaire de texte**.

Le travail reste volontairement aligné avec le brief pédagogique :
- préparation des données ;
- entraînement d'un ou plusieurs modèles de deep learning ;
- comparaison des performances ;
- conclusion claire et défendable.

## Objectif

Construire un pipeline minimal mais rigoureux permettant de :

1. charger et nettoyer le dataset ;
2. transformer les SMS en séquences numériques ;
3. entraîner deux architectures de deep learning ;
4. comparer leurs performances avec des métriques adaptées à un dataset déséquilibré.

## Données utilisées

Le jeu de données fourni contient **5 572 SMS** annotés en deux classes :

- **ham** : message normal ;
- **spam** : message indésirable.

Répartition observée :

- **4 825 ham**
- **747 spam**

Le dataset est donc **déséquilibré**, ce qui justifie l'usage de métriques complémentaires à l'accuracy.

## Méthodologie

### 1. Préparation des données
- conservation des colonnes utiles (`label`, `text`) ;
- encodage de la cible (`ham -> 0`, `spam -> 1`) ;
- séparation train/test avec `stratify` ;
- tokenization du texte ;
- padding à longueur fixe.

### 2. Modèles évalués
Deux modèles de deep learning ont été comparés :

- **Modèle 1** : `Embedding + GlobalAveragePooling1D + Dense`
- **Modèle 2** : `Embedding + LSTM + Dense`

Le premier modèle suit explicitement la recommandation du brief : **commencer simple**.

### 3. Métriques suivies
Compte tenu du déséquilibre des classes, les résultats sont évalués avec :

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

## Résultats clés

Sur l'exécution retenue dans ce dépôt, les résultats obtenus sont les suivants :

| Modèle | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Embedding + GlobalAveragePooling1D | 0.9785 | 0.9562 | 0.8792 | 0.9161 |
| Embedding + LSTM | 0.9632 | 0.9286 | 0.7852 | 0.8509 |

## Lecture des résultats

Le modèle **Embedding + GlobalAveragePooling1D** est retenu comme meilleur compromis global :

- il est plus simple ;
- il s'entraîne plus vite ;
- il obtient de meilleures performances globales sur ce dataset ;
- il reste plus facile à expliquer et à défendre.

Le LSTM est intéressant comme benchmark séquentiel, mais il ne montre pas ici de gain suffisant pour justifier sa complexité supplémentaire.

## Structure du dépôt

```text
.
├── README.md
├── .gitignore
├── requirements.txt
├── data/
│   └── spam.csv
├── notebooks/
│   ├── spam_detector_clean.ipynb
│   └── spam_detector_executed.ipynb
└── docs/
    ├── AUDIT_PROJET.md
    ├── CHECKLIST_AVANT_PUSH.md
    └── GUIDE_GIT.md
```

## Notebooks fournis

### `notebooks/spam_detector_executed.ipynb`
Version exécutée avec :
- outputs visibles ;
- tableaux ;
- courbes d'apprentissage ;
- matrices de confusion ;
- comparaison finale.

### `notebooks/spam_detector_clean.ipynb`
Version propre, sans outputs, destinée à être relancée facilement.

## Technologies utilisées

- Python
- pandas
- numpy
- matplotlib
- scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

## Exécution

### 1. Créer un environnement virtuel
```bash
python -m venv .venv
```

### 2. Activer l'environnement

Sous Windows :
```bash
.\.venv\Scripts\activate
```

Sous macOS / Linux :
```bash
source .venv/bin/activate
```

### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 4. Ouvrir le notebook
Ouvrir ensuite l'un des notebooks dans VS Code ou Jupyter :

- `notebooks/spam_detector_clean.ipynb`
- `notebooks/spam_detector_executed.ipynb`

## Limites

- le dataset reste de taille modeste pour du deep learning ;
- seules les données textuelles sont utilisées ;
- aucun signal contextuel ou métier additionnel n'est disponible ;
- l'évaluation repose sur un split statique et non sur un flux réel de production.

## Pistes d'amélioration

- tester une approche de **transfer learning** si le cadre le permet ;
- comparer avec des embeddings pré-entraînés ;
- ajuster le seuil de décision selon le coût métier des erreurs ;
- enrichir les données avec des variables contextuelles.

## Conclusion

Le projet remplit l'objectif pédagogique attendu : mettre en place un pipeline de deep learning sobre, cohérent et défendable pour détecter les SMS spam.

L'enseignement principal est qu'un **modèle simple, bien préparé et bien évalué** peut offrir de très bonnes performances sur ce type de problème, sans surcharger inutilement la solution.