---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Comparer des classificateurs

Dans la [feuille précédente](3_extraction_d_attributs.md), nous avons
observé que les performances pouvaient varier selon le niveau de
prétraitement des données (données brutes, données nettoyées,
extractions d'attributs, etc.). Cependant, l'analyse de performance
n'a été conduite qu'avec un seul classificateur (plus proches
voisins).

Il est posssible que chaque classificateur donne des résultats
différents pour un niveau de pré-traitement donné.

Dans cette feuille, on étudie les performances selon le type de
classificateur. Les étapes seront:
1. Déclarer et entraîner les différents classificateurs sur nos
   données;
2. Visualiser les performances de chaque classificateur;
3. Comprendre et identifier le sous-apprentisssage et le
   surapprentissage;
4. **Pour aller plus loin ♣**: Étudier un comité de classificateurs et
   l'incertitude dans les données;
5. **Pour aller plus loin ♣**: Apprentissage profond et transfert
   d'apprentissage.

+++

## Entraînement des différents classificateurs

+++

### Import des bibliothèques

```{code-cell} ipython3
import os, re
from glob import glob as ls
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd
import seaborn as sns; sns.set()
from PIL import Image
%load_ext autoreload
%autoreload 2
import warnings
warnings.filterwarnings("ignore")
from sys import path

from utilities import *
from intro_science_donnees import data
from intro_science_donnees import *
```

+++ {"user_expressions": []}

### Import des données

+++ {"user_expressions": []}

Nous chargeons les images prétraitées dans la feuille précédente :

```{code-cell} ipython3
dataset_dir = 'clean_data'

images = load_images(dataset_dir, "*.png")
```

+++ {"user_expressions": []}

Vérifions l'import en affichant les 20 premières images :

```{code-cell} ipython3
image_grid(images[:20])
```

+++ {"user_expressions": []}

Pour mettre en œuvre des classificateurs, nous pouvons repartir des
attributs extraits dans la fiche précédente :

```{code-cell} ipython3
df_features = pd.read_csv('features_data.csv', index_col=0)
df_features
```

+++ {"user_expressions": []}

On vérifie que nos données sont normalisées :

```{code-cell} ipython3
df_features.describe()
```

+++ {"user_expressions": []}

### Déclaration des classificateurs

+++ {"user_expressions": []}

Nous allons à présent importer les classificateurs depuis la librairie
`scikit-learn`. Pour cela, on stockera:
- les noms des classificateurs dans la variable `model_name`;
- les classificateurs eux-mêmes dans la variable `model_list`.

```{code-cell} ipython3
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

model_name = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
model_list = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025, probability=True),
    SVC(gamma=2, C=1, probability=True),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=10),
    RandomForestClassifier(max_depth=10, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
```

+++ {"user_expressions": []}

La bibliothèque `scikit-learn` nous simplifie la tâche : bien que les
classificateurs aient des fonctionnements très différents, leurs
interfaces sont identiques (rappelez vous la notion d'encapsulation
vue dans le cours «Programmation Modulaire»). Les méthodes sont:
- `.fit` pour entraîner le classificateur sur des données
  d'entraînement;
- `.predict` pour prédire des étiquettes sur les données de test;
- `.predict_proba` pour obtenir des prédictions sur les données de
  test sous forme de probabilités sur les classes;
- `.score` pour calculer la performance du classificateur.

+++ {"user_expressions": []}

## Visualisation des performances des classificateurs

+++ {"user_expressions": []}

Nous allons à présent faire des tests systématiques sur cet ensemble
de données (attributs de l'analyse de variance univiarié). La fonction
`systematic_model_experiment(data_df, model_name, model_list,
sklearn_metric)` permet de réaliser ces tests systématiques:

```{code-cell} ipython3
df_features
```

```{code-cell} ipython3
from sklearn.metrics import balanced_accuracy_score as sklearn_metric
compar_results = systematic_model_experiment(df_features, model_name, model_list, sklearn_metric)
compar_results.style.set_precision(2).background_gradient(cmap='Blues')
```

+++ {"user_expressions": []}

**Exercice:** Quelle méthode obtient les meilleures performances de
test?

```{code-cell} ipython3
model_list[compar_results.perf_te.argmax()]
```

+++ {"user_expressions": []}

On peut également représenter les performances dans un graphique en
barres:

```{code-cell} ipython3
compar_results[['perf_tr', 'perf_te']].plot.bar()
plt.ylim(0.5, 1)
plt.ylabel(sklearn_metric.__name__);
```

+++ {"user_expressions": []}

##  Sous-apprentissage et surapprentissage

+++ {"user_expressions": []}

Lorsque l'on entraîne un classificateur sur des données, deux
comportements sont à éviter:

- Lorsque à la fois les performances d'entraînement `perf_tr` et les
  performances de test `perf_te` sont mauvaises, on dit alors que le
  classificateur a ***sous-appris*** (*under-fitting*).
- Lorsque les performances d'entraînement `perf_tr` sont bonnes mais
  les performances de test `perf_te` sont mauvaises, on dit alors que
  le classificateur a ***sur-appris*** (*over-fitting*).

Ces notions sont liés à la granularité de la frontière de décision,
qui peut être illustré par les graphiques suivants: 

<img class="fit-picture"
src="https://qph.fs.quoracdn.net/main-qimg-0d12f79a596c9b2ee1e23476fa3d44aa"
alt="Under-fitting, optimal and Over-fitting models">

Analysons quels classificateurs ont sur-appris resp. sous-appris. Pour
cela nous allons:
1. Identifier les classificateurs dont les performances de test sont
   inférieures à la performance de test médiane.
2. Parmi ceux-ci, nous dirons que les classificateurs dont la
   performance d'entraînement est **inférieure à la médiane** ont
   sous-appris, tandis que ceux dont la performance d'entraînement est
   **supérieure à la médiane** ont sur-appris.

```{code-cell} ipython3
analyze_model_experiments(compar_results)
```

+++ {"user_expressions": []}

**Exercice:** Quelles sont les classificateurs qui ont sous-appris
resp. sur-appris ?

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "c366698d4803f51a5a5cf85ab64cc394", "grade": true, "grade_id": "cell-c4a46b1532ae7fe4", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}, "user_expressions": []}

Les classificateurs qui ont sur-appris sont RBF SVM, Decision Tree, Gaussian Process, AdaBoost. QDA a sous-appris.

+++ {"user_expressions": []}

## Pour aller plus loin ♣: Comité de classificateurs et incertitude

+++ {"user_expressions": []}

Un comité de classificateurs est un classificateur dans lequel les
réponses de plusieurs classificateurs sont combinées en une seule
réponse. En d'autres termes, les classificateurs **votent** pour les
prédictions émises.

On pourrait considérer notre liste de classificateur `model_list`
comme un comité de classificateurs, entraînés sur les mêmes données
d'entraînement, et faisant des prédictions sur les mêmes données de
test. Regardez le code suivant; il est en réalité simple: on définit
les méthodes `fit`, `predict`, `predict_proba` et `score` comme
faisant la synthèse des résultats des classificateurs contenus dans
`self.model_list`.

```{code-cell} ipython3
class ClassifierCommittee():
    def __init__(self, model_list):
        self.model_list = model_list
        
    def fit(self,X,y):
        for model in self.model_list:
            model.fit(X,y)
    def predict(self,X):
        predictions = []
        for model in self.model_list:
            predictions.append(model.predict(X))
        predictions = np.mean(np.array(predictions),axis = 0)
        results = []
        for v in predictions:
            if v < 0:
                results.append(-1)
            else:
                results.append(1)
        return np.array(results)
    
    def predict_proba(self,X):
        predictions = []
        for model in self.model_list:
            predictions.append(model.predict_proba(X))
        return np.swapaxes(np.array(predictions), 0, 1)
    def score(self,X):
        scores = []
        for model in self.model_list:
            scores.append(model.score(X,y))
        return np.swapaxes(np.array(scores), 0, 1)
```

+++ {"user_expressions": []}

Quelles seraient les performances d'un tel classificateur?

```{code-cell} ipython3
commitee = ClassifierCommittee(model_list)
```

```{code-cell} ipython3
perf_tr, std_tr, perf_te, std_te = df_cross_validate(df_features, commitee, sklearn_metric)
print(perf_te, std_te)
```

+++ {"user_expressions": []}

La performance de l'ensemble de classificateurs n'est pas forcément
meilleure que les classificateurs pris individuellement. Cependant,
l'accord ou le désaccord des classificateurs sur les prédictions peut
nous donner des informations sur l'incertitude des données.

+++ {"user_expressions": []}

En effet, chaque classificateur peut émettre des probabilités sur les
classes avec la méthode `predict_proba`.  Pour quantifier
l'incertitude d'une image, on peut distinguer deux cas de figure :
- Les classificateurs du comité sont d'accord sur des probabilités
  incertaines (à gauche sur la figure) : chaque classificateur émet
  une probabilité proche de `[0.5, 0.5]`. On parle alors d'incertitude
  aléatorique, et elle est lié à l'**ambiguïté** intrinsèque de la
  tâche de classification.
- Les classificateurs du comité sont certains de leurs prédictions,
  mais ils sont en désaccord entre eux (à droite sur la figure) : les
  classificateurs émettent chacun une probabilité confiante mais
  différentes: `[1.,0.]`, `[0., 1.]`, `[0.9, 0.1]`, `[0.05, 0.95]`,
  etc. Dans ce cas on parle d'incertitude `épistémique`, et elle est
  lié à l'idée de **nouveauté** dans les données. Cette incertitude
  peut être réduite en ajoutant de nouvelles données.

<img class="fit-picture"
     src=media/aleatoric_epistemic.png
     alt="Under-fitting, optimal and Over-fitting models">
     

Intéressons nous aux images avec une faible incertitude épistémique
(les classificateurs sont d'accord) et une grande incertitude
aléatorique (les classificateurs sont incertains de la
prédiction). Cela correspond à des images se situant aux abords de la
frontière de décision entre nos deux catégories.

+++ {"user_expressions": []}

### Incertitude aléatorique

+++ {"user_expressions": []}

Nous allons utiliser l'[entropie de
Shannon](https://fr.wikipedia.org/wiki/Entropie_de_Shannon), qui est
une mesure en théorie de l'information permettant d'estimer la
quantité d'information contenu dans une source d'information (ici
notre probabilité sur les classes): 

$$H(X) = - \sum_{i=1}^{n}P(x_i)log_2(x_i)$$

où $x_i$ est la probabilité de classification sur la classe $i$.

On récupère alors les prédictions de nos images pour les 10
classificateurs :

```{code-cell} ipython3
X = df_features.iloc[:, :-1].to_numpy()
Y = df_features.iloc[:, -1].to_numpy()
commitee.fit(X, Y)
prediction_probabilities = commitee.predict_proba(X)
prediction_probabilities.shape
```

+++ {"user_expressions": []}

La dimension de notre matrice de prédiction est donc: `(nombre
d'images, nombre de classificateur, nombre de classess)`. Appliquons
l'entropie pour chaque prédiction de chaque classificateur :

```{code-cell} ipython3
from scipy.stats import entropy
entropies_per_classifier = entropy(np.swapaxes(prediction_probabilities, 0, 2))
```

+++ {"user_expressions": []}

On moyenne les entropies d'une image entre les classificateurs :

```{code-cell} ipython3
entropies = np.mean(entropies_per_classifier, axis = 0)
```

+++ {"user_expressions": []}

Puis on ajoute ces valeurs dans une table que l'on trie par ordre
décroissant d'entropie. Les images avec l'entropie la plus grande sont
les plus incertaines et donc les plus informatives pour le modèle :

```{code-cell} ipython3
df_uncertainty = pd.DataFrame({"images" : images,
                           "entropy" : entropies})
df_uncertainty = df_uncertainty.sort_values(by=['entropy'],ascending=False)
```

```{code-cell} ipython3
df_uncertainty.style.background_gradient(cmap='RdYlGn_r')
```

+++ {"user_expressions": []}

Affichons les 10 images **les plus incertaines** pour le comité de
classificateurs, selon cette mesure aléatorique de l'incertitude. Ces
images nous donne une idée de l'ambiguité intrinsèque de notre base de
données.

```{code-cell} ipython3
uncertain_aleatoric_images = df_uncertainty['images'].tolist()
image_grid(uncertain_aleatoric_images[:10])
```

+++ {"user_expressions": []}

Affichons maintenant les 10 images les **moins incertaines** pour
notre comité de classificateurs :

```{code-cell} ipython3
image_grid(uncertain_aleatoric_images[-10:])
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "cca222eea861c11f3138a513913c4d36", "grade": true, "grade_id": "cell-a4f0ee22f69b361f", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}, "user_expressions": []}

**Exercice:** Ces résultats vous semblent-ils surprenants? Expliquer.

Non les resultats sont plutots evidents puisque les 10 images les plus incertaines sont celles qui sont mal vus, contrairement les autres sont plus visibles.

+++ {"user_expressions": []}

### Incertitude épistémique

+++ {"user_expressions": []}

Pour l'incertitude épistémique, on va simplement moyenner entre les
classificateurs les écarts types entre les classes :

```{code-cell} ipython3
# std entre les classses
epistemic_uncertainty = np.std(prediction_probabilities, axis = 2)
print(epistemic_uncertainty.shape)
# mean entre les classificateurs
epistemic_uncertainty = np.mean(epistemic_uncertainty, axis = 1)
print(epistemic_uncertainty.shape)
```

+++ {"user_expressions": []}

On ajoute cette mesure au tableau puis on classe les images par ordre
décroissant :

```{code-cell} ipython3
df_uncertainty["std_epistemic"] = epistemic_uncertainty
df_uncertainty = df_uncertainty.sort_values(by=['std_epistemic'],ascending=False)
df_uncertainty.style.background_gradient(cmap='RdYlGn_r')
```

```{code-cell} ipython3
df_uncertainty.corr()
```

+++ {"user_expressions": []}

Les valeurs d'incertitude aléatorique (entropie) et épistémiques (std)
ne semblent pas très corrélées. Affichons les images les plus
incertaines selon cette mesure d'incertitude épistémique :

```{code-cell} ipython3
uncertain_epistemic_images = df_uncertainty['images'].tolist()
# les images où les classificateurs sont le moins d'accord
image_grid(uncertain_epistemic_images[:10])
```

```{code-cell} ipython3
# les images où les classificateurs sont le plus d'accord
image_grid(uncertain_epistemic_images[-10:])
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "1d7c9bbfcf4b2868464a20f6b3efdc7c", "grade": true, "grade_id": "cell-d32fa098a68274b0", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}, "user_expressions": []}

**Exercice:** Ces résultats vous semblent-ils surprenants? Expliquer.

VOTRE RÉPONSE ICI

+++ {"user_expressions": []}

## Pour aller plus loin ♣: apprentissage profond

+++ {"user_expressions": []}

En classe, vous apprendrez ce qu'est un ***réseau de neurones
artificiel*** et l'***apprentissage profond***. En résumé, il s'agit
de modèles (classificateurs par exemple) dont l'architecture est
articulée en «couches» composées de différentes transformations non
linéaires: des couches de neurones, des convolutions ou d'autres
transformations.

Des ***poids*** (paramètres appris par le réseau) sont associés aux
neurones et sont optimisés lors de l'entraînement sur les données par
une technique appelée ***descente de gradient***.  Quand le réseau
fait une prédiction, les valeurs des données brutes (pixels) sont
propagées dans le réseau jusqu'à donner le résultat en sortie. La
spécificité de ces architectures est qu'elles manipulent des données
brutes, sans utiliser d'attributs: les attributs sont en quelque sorte
appris par le réseau.

Pour mieux comprendre l'architecture connexioniste d'un réseau de
neurone profond, consultez [ce
site](https://www.cs.ryerson.ca/~aharley/vis/conv/) créé par Adam
Harley de l'Université Carnegie Mellon. Il présente un réseau
convolutionnel pour la reconnaissance de chiffres manuscrits. On peut
y dessiner des chiffres (en haut à gauche) et visualiser les
prédictions faites lors de la propagation des valeurs des pixels dans
le réseau. Passez votre souris sur un neurone et vous verrez ses
connexions avec les neurones des autres couches s'afficher.

Ces réseaux profonds comptent beaucoup de paramètres (neurones) à
optimiser. Ils nécessitent des infrastructures spéciales pour les
entraîner: de puissantes cartes graphiques (GPU), comme celles
utilisées pour les jeux vidéos, qui permettent de paralléliser les
calculs. Nous ne disposons pas de telles infrastructures sur le
service JupyterHub de l'université.

Nous allons donc voir comment transférer les connaissances apprises
par un réseau pour l'adapter à votre problème. On appelle cela
l'***apprentissage par transfert*** (*Transfer Learning*). Il nous
permet de partir d'un réseau de neurones déjà entraîné pour des tâches
similaires, et de n'avoir plus qu'à l'entraîner un peu plus pour notre
tâche spécifique.

+++ {"user_expressions": []}

### MobileNet et transfert

+++ {"user_expressions": []}

<div class="alert alert-info" role="alert">

La suite de cette feuille utilise les bibliothèques `tensorflow` et
`keras`. Celles-ci sont installées sur JupyterHub, mais pas forcément
ailleurs (elles sont assez volumineuses).

</div>

+++ {"user_expressions": []}

Nous allons utiliser un réseau pré-entraîné appelé `MobileNet`, dont
l'architecture a été optimisé pour fonctionner sur les téléphones
portables. Il a été pré-entraîné pour reconnaître mille catégories
différentes (dont des animaux, des objets de la vie courante etc.).

Depuis la librairie `keras`, importons le modèle pré-entraîné sur la
base de donnée `imagenet` :

```{code-cell} ipython3
import tensorflow as tf
tf.config.run_functions_eagerly(True)

mobilenet = tf.keras.applications.MobileNet(include_top=False, weights='imagenet', input_shape = (32,32,3))
```

Observons l'architecture en couches de ce réseau de neurones; chaque
ligne ci-dessous décrit brièvement l'une des couches successives :

```{code-cell} ipython3
mobilenet.summary()
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "3165c0614988c0cb4d5397f10689c1a7", "grade": true, "grade_id": "cell-8c9901655c0067be", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}}

**Exercice:** Combien ce réseau a-t-il de paramètres? Parmi ces
paramètres, combien sont entraînables?

VOTRE RÉPONSE ICI

+++ {"user_expressions": []}

En déclarant le réseau avec le paramètre `include_top=False`, on a
retiré la dernière couche de neurone. **Le transfert d'apprentissage
va pouvoir être fait en ajoutant des nouvelles couches de neurones
entraînables à la fin de ce réseau pré-entraîné (dont les paramètres
vont être gelés).** Ainsi, on pourra entraîner seulement les dernières
couches de notre réseau sans avoir à réentraîner tout le réseau, ce
qui nécessiterait autrement une puissante carte graphique. En d'autres
termes, le réseau `mobilenet` va nous donner des attributs abstraits
sur lesquels nous allons réentraîner un nouveau réseau de neurones,
plus petit.

+++

On commence par geler les poids du réseau `mobilenet` :

```{code-cell} ipython3
for layer in mobilenet.layers:
    layer.trainable=False
```

Vérifiez que tous les paramètres de `mobilenet` sont à présent
non-entraînables :

```{code-cell} ipython3
mobilenet.summary()
```

On va à présent ajouter de nouvelles couches entraînables de neurones :

```{code-cell} ipython3
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense,GlobalAveragePooling2D

myneuralnet = keras.Sequential(
    [
        mobilenet,
        GlobalAveragePooling2D(),
        Dense(64, activation="relu", name="layer1"),
        Dense(64, activation="relu", name="layer2"),
        Dense(2, activation="softmax",name="layer3"),
    ]
)
```

Configurons le réseau `mobilenet` pour prendre en entrée des images de
taille `(32,32,3)`:

```{code-cell} ipython3
myneuralnet.build((None,32,32,3))
myneuralnet.summary()
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "2ae57494f5a64c9084b8a03ad86cf770", "grade": true, "grade_id": "cell-134c13e252ae4e67", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}}

**Exercice**: À présent combien ce réseau a-t-il de paramètres? Parmi
ces paramètres, combien sont entraînables?

VOTRE RÉPONSE ICI

+++

Chargeons nos données recentrées et recadrées de taille 32x32 :

```{code-cell} ipython3
df_clean = pd.read_csv('clean_data.csv', index_col=0)  # chargement du fichier dans le notebook
df_clean
```

+++ {"user_expressions": []}

On souhaite avoir une variable `X` de type `np.ndarray` de taille
`(491, 32, 32, 3)` c'est à dire `(nombre d'images, largeur, hauteur,
nombre de couches de couleurs)` :

```{code-cell} ipython3
X = np.array(df_clean.drop(['class'], axis=1)).reshape((491, 32, 32, 3))
```

+++ {"user_expressions": []}

Il est nécessaire d'encoder les étiquettes sous la forme d'un *one-hot
vector* c'est-à-dire d'un tableau `y_onehot` de taille `(nombre
d'images, nombre de classes)` tel que `y_onehot[i,j]==1` si la i-ème
étiquette est la j-ème classe et `0` sinon. Dans notre cas, la i-ème
ligne vaut `(1,0)` si le i-ème fruit est une pomme, et `(0,1)` si
c'est une banane :

```{code-cell} ipython3
y = np.array(df_clean["class"])
classes = np.unique(y)
class_to_class_number = { cls: number for number, cls in enumerate(classes) }
y_onehot = np.zeros((len(y), len(np.unique(y))))
for i, label in enumerate(y):
    y_onehot[i, class_to_class_number[label]] = 1
```

+++ {"user_expressions": []}

On doit maintenant définir l'***optimisateur*** et la ***fonction de
coût*** qui vont permettre d'optimiser les paramètres de nos
neurones. Les poids des neurones vont être optimisés pour ***minimiser
la fonction de coût*** (*loss*) :

```{code-cell} ipython3
opt = keras.optimizers.Adam(learning_rate=0.01)
myneuralnet.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer=opt)
```

Avant de pouvoir entraîner notre réseau, on va d'abord diviser nos
données en deux: les données d'entraînement et les données de test :

```{code-cell} ipython3
test = np.array(y == 1, dtype = int)
X_train = np.concatenate((X[:165], X[333:333+ 79]), axis = 0)
y_train = np.concatenate((y_onehot[:165], y_onehot[333:333+ 79]), axis = 0)

X_test = np.concatenate((X[165:333], X[333+ 79:]), axis = 0)
y_test = np.concatenate((y_onehot[165:333], y_onehot[333+ 79:]), axis = 0)
```

On peut finalement entraîner notre réseau de neurone !  

**Attention :** Le temps de calcul étant potentiellement élevé, la
cellule ci-dessous est désactivée par défaut. Mettez en commentaire la
première ligne pour lancer le calcul.

```{code-cell} ipython3
%%script echo cellule désactivée

from sklearn.model_selection import StratifiedKFold
import pdb
stop = pdb.set_trace
def create_model():
    myneuralnet = keras.Sequential(
    [
        mobilenet,
        GlobalAveragePooling2D(),
        Dense(64, activation="relu", name="layer1"),
        Dense(64, activation="relu", name="layer2"),
        Dense(2, activation="softmax",name="layer3"),
    ])
    myneuralnet.build((None,32,32,3))
    opt = keras.optimizers.Adam(learning_rate=0.01)
    myneuralnet.compile(loss='categorical_crossentropy', metrics=["accuracy"], optimizer=opt)
    return myneuralnet

def train_evaluate(model, x_train, y_train, x_test, y_test):
    history = model.fit(x_train, y_train, epochs = 10, shuffle = True)
    return model.evaluate(x_test, y_test)

kFold = StratifiedKFold(n_splits=10)
scores = np.zeros(10)
idx = 0
for train, test in kFold.split(X, y):
    model = create_model()
    scores[idx] = train_evaluate(model, X[train], y_onehot[train], X[test], y_onehot[test])[1]
    idx += 1
print(scores)
print(scores.mean())
```

Afficher les valeurs d'accuracy au fur et à mesure des itérations
d'optimisations :

```{code-cell} ipython3
%%script echo cellule désactivée

plt.plot(scores)
plt.xlabel("Nombre d'itération d'optimisations")
plt.ylabel("Accuracy")
```

## Conclusion

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "175abe395c45f2d230bcc2e7396363ac", "grade": true, "grade_id": "cell-48ecfb6975c2fe0b", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}}

Cette feuille a fait un tour d'horizon d'outils de classification à
votre disposition. Prenez ici quelques notes sur ce que vous avez
appris, observé, interprété.

VOTRE RÉPONSE ICI

+++

Itérez sur le prétraitement, l'extraction d'attributs et la
classification de vos images dans les feuilles respectives pour
améliorer votre classification, puis préparez votre diaporama!
