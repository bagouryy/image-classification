---
jupytext:
  notebook_metadata_filter: rise
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
rise:
  auto_select: first
  autolaunch: false
  backimage: fond.png
  centered: false
  controls: false
  enable_chalkboard: true
  height: 100%
  margin: 0
  maxScale: 1
  minScale: 1
  scroll: true
  slideNumber: true
  start_slideshow_at: selected
  transition: none
  width: 90%
---

+++ {"slideshow": {"slide_type": "slide"}}

# Plane and cars

- Binôme: Mohammed MASMOUDI, Nour ALBAGOURY
- Adresses mails: mohammed.masmoudi@universite-paris-saclay.fr , nour.albagoury@universite-paris-saclay.fr 
- [Dépôt GitLab](https://gitlab.dsi.universite-paris-saclay.fr/xxx.yyy/Semaine8/)*

+++ {"slideshow": {"slide_type": "slide"}}

## Jeu de données


Pour ce projet, nous avons collecté un jeu de données sur les voitures et les avions. Le jeu de données contient des informations sur différentes caractéristiques des véhicules, telles que la taille, les couleurs, etc. On a donc commencé par mettre toutes nos images en format .jpg et les mettre dans un dossier "data". Dans le code ci-dessous, nous avons donc affiché nos images.

```{code-cell} ipython3
# Automatically reload code when changes are made
%load_ext autoreload
%autoreload 2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
%matplotlib inline
import scipy
from scipy import signal
import pandas as pd
from glob import glob as ls
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score as sklearn_metric
import os, re
import numpy as np
import seaborn as sns; sns.set()
import warnings
warnings.filterwarnings("ignore")
from sys import path



from utilities import *
from intro_science_donnees import data
from intro_science_donnees import *
```

```{code-cell} ipython3
dataset_dir = 'data'
images = load_images(dataset_dir, "*.jpg")

sample = list(images[:10]) + list(images[-10:])
image_grid(sample)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Prétraitement

Pour le prétraitement, on a commencé par un code qui applique un filtre de détection du foreground aux images de 'sample', inverse les couleurs des images avec des fonds clairs (si nécessaire), puis affiche les images résultantes dans une grille pour permettre leur visualisation.
On a ensuite utilisé deux fonctions qui sont "my_foreground_filter" et "my_preprocessing". On a enfin utiliser un code qui applique une fonction de prétraitement my_preprocessing à chaque image dans une liste images, stocke les images prétraitées dans une nouvelle liste clean_images, puis affiche les images prétraitées dans une grille pour permettre leur visualisation.

```{code-cell} ipython3
image_grid([invert_if_light_background(foreground_filter(img, theta=85))
            for img in sample])
```

```{code-cell} ipython3
def my_foreground_filter(img):
    foreground = foreground_filter(img, theta=85)
    foreground = invert_if_light_background(foreground)
    foreground = scipy.ndimage.gaussian_filter(foreground, sigma=.1)
    return foreground
```

```{code-cell} ipython3
def my_preprocessing(img):
    """
    Prétraitement d'une image
    
    - Calcul de l'avant plan
    - Mise en transparence du fond
    - Calcul du centre
    - Recadrage autour du centre
    """
    foreground = my_foreground_filter(img)
    img = transparent_background(img, foreground)
    coordinates = np.argwhere(foreground)
    if len(coordinates) == 0: # Cas particulier: il n'y a aucun pixel dans l'avant plan
        width, height = img.size
        center = (width/2, height/2)
    else:
        center = (np.mean(coordinates[:, 1]), np.mean(coordinates[:, 0]))
    img = crop_around_center(img, center)
    return img
```

```{code-cell} ipython3
clean_images = images.apply(my_preprocessing)
image_grid(clean_images)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Visualisation des données


Nous avons utilisé plusieurs techniques de visualisation de données pour explorer notre jeu de données. Nous avons utilisé des tableaux pour examiner la distribution des données et identifier les valeurs aberrantes.

```{code-cell} ipython3
sklearn_model = KNeighborsClassifier(n_neighbors=3)
performances = pd.DataFrame(columns = ['Traitement', 'perf_tr', 'std_tr', 'perf_te', 'std_te'])
df_raw = images.apply(image_to_series)
df_raw['class'] = df_raw.index.map(lambda name: 1 if name[0] == 'a' else -1)
# Validation croisée
p_tr, s_tr, p_te, s_te = df_cross_validate(df_raw, sklearn_model, sklearn_metric)
metric_name = sklearn_metric.__name__.upper()
print("AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_tr, s_tr))
print("AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_te, s_te))
performances.loc[0] = ["Images brutes", p_tr, s_tr, p_te, s_te]
performances.style.format(precision=2).background_gradient(cmap='Blues')
```

```{code-cell} ipython3
df_features = pd.DataFrame({'redness': clean_images.apply(redness),
                            'greenness': clean_images.apply(greenness),
                            'blueness': clean_images.apply(blueness),
                            'elongation': clean_images.apply(elongation),
                            'perimeter': clean_images.apply(perimeter),
                            'surface': clean_images.apply(surface)})
epsilon = sys.float_info.epsilon
df_features = (df_features - df_features.mean())/(df_features.std() + epsilon) # normalisation 
df_features.describe() # nouvelles statistiques de notre jeu de donnée
df_features["class"] = df_features.index.map(lambda name: 1 if name[0] == 'a' else -1)
df_features[df_features.isna()] = 0
df_features.style.background_gradient(cmap='coolwarm')
```

Aprés avoir vu la performance des tests avec les 6 attributs ad-hoc, on cherche les attributs qui corrèlent le plus avec la class

```{code-cell} ipython3
header = ['R','G','B','M=maxRGB', 'm=minRGB', 'C=M-m', 'R-(G+B)/2', 'G-B', 'G-(R+B)/2', 'B-R', 'B-(G+R)/2', 'R-G', '(G-B)/C', '(B-R)/C', '(R-G)/C', '(R+G+B)/3', 'C/V']

df_features_large = df_features.drop("class", axis = 1)

df_features_large = pd.concat([df_features_large, clean_images.apply(get_colors)], axis=1)

epsilon = sys.float_info.epsilon # epsilon
df_features_large = (df_features_large - df_features_large.mean())/(df_features_large.std() + epsilon) # normalisation 
df_features_large[df_features_large.isna()] = 0
df_features_large.describe() # nouvelles statistiques de notre jeu de donnée
    
    
df_features_large["class"] = df_features["class"]
df_features_large
```

On trouve donc 23 attributs ad-hoc, on fait les tests pour trouver lesquelles corrèlent le plus avec class et ensuite on trouve que le nombre d'attributs le plus efficace et 5 attributs qui sont : elongation' 'redness' 'R-G' 'R' 'R-(G+B)/2'

+++

On fait donc le test avec 5 attributs par analyse de variance univarié et oon trouve un training set de 89% et test set de 755

+++ {"slideshow": {"slide_type": "slide"}}

## Classificateurs favoris

+++

Pour trouver les classificateurs on part des attributs trouvé.

```{code-cell} ipython3
df_features = pd.read_csv('features_data.csv', index_col=0)
df_features
```

Nous avons importer les classificateurs depuis la librairie scikit-learn. Nous stockons les noms des classificateurs dans la variable model_name,
les classificateurs eux-mêmes dans la variable model_list.

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

```{code-cell} ipython3
import warnings
warnings.filterwarnings("ignore")
```

```{code-cell} ipython3
from sklearn.metrics import balanced_accuracy_score as sklearn_metric
compar_results = systematic_model_experiment(df_features, model_name, model_list, sklearn_metric)
compar_results.style.background_gradient(cmap='Blues').format(precision=2)
```

```{code-cell} ipython3
model_list[compar_results.perf_te.argmax()]
```

Nous avons utilisé systematic_model_experiment pour visualiser les performances des tests fait avec chaque classificateurs. Et nous avons trouver qu'on obtient les meilleurs performances de test set avec le classificateur Linear SVM avec un resultat de 76% et un training de 82%. On peut aussi choisir Nearest Neighbors avec un test de 74% et un training de 89%

+++ {"slideshow": {"slide_type": "slide"}}

## Résultats

### Observations

Le projet visait à classifier des images d'avions et de voitures à partir de caractéristiques extraites de ces images. Les résultats montrent que le classificateur Linear SVM a obtenu une performance de 0,76 avec une erreur standard de 0,08, ce qui est supérieur à la plupart des autres classificateurs testés. Les observations suggèrent que les caractéristiques extraites à partir des images ont une forte influence sur les performances de classification, et que le Linear SVM est capable de capturer les interactions non linéaires entre les caractéristiques. En conséquence, le Random Forest est choisi comme le meilleur classificateur pour ce projet

+++

### Interprétations

+++

Nous avons utilisé un ensemble de données comprenant des images d'avions et de voitures afin de prédire leur classe respective à l'aide de différents algorithmes de classification. Nous avons constaté que le classificateur Linear SVM a donné les résultats les plus précis parmi les autres algorithmes testés. Cela peut être dû à sa capacité à combiner les prédictions de plusieurs arbres de décision pour améliorer la précision globale de la classification. En outre, nous avons observé que les performances des différents algorithmes varient en fonction de la complexité du modèle, de la taille de l'ensemble de données et des paramètres spécifiques de chaque algorithme. Enfin, il est important de noter que les performances de classification pourraient être encore améliorées avec des techniques de prétraitement d'image plus avancées

+++ {"slideshow": {"slide_type": "slide"}}

## Discussion 

Premièrement, parlons des biais potentiels de nos données. Notre ensemble de données a été rassemblé à partir d'une variété de sources en ligne, ce qui signifie qu'il pourrait y avoir des biais dans la façon dont les images ont été collectées et étiquetées. En outre, notre ensemble de données est limité en termes de variété de marques et de modèles d'avions et de voitures, ce qui pourrait affecter les performances de notre classificateur.

En ce qui concerne l'utilisation d'un tel projet dans la vraie vie, il pourrait être utile pour diverses applications telles que la surveillance de la circulation, la reconnaissance de véhicules de transport en commun, etc. Cependant, avant de l'utiliser pour des applications réelles, il est important de vérifier les performances du classificateur sur un ensemble de données plus diversifié et plus grand.

Nous avons également rencontré des difficultés lors de la préparation de l'ensemble de données et de l'entraînement du classificateur. Tout d'abord, certaines images étaient floues ou mal cadrées, ce qui a affecté les performances du classificateur. 

Enfin, nous avons choisi le classificateur Linear SVM pour sa précision. Cependant, il est important de noter que d'autres classificateurs pourraient être plus appropriés pour d'autres types de données et que le choix du classificateur dépendra des besoins spécifiques de chaque projet.

Dans l'ensemble, notre projet a démontré la faisabilité de la classification d'images d'avions et de voitures avec un modèle Linear SVM. Cependant, il y a encore des défis à relever pour améliorer les performances du classificateur et l'adapter à des applications réelles

+++

## Conclusion

+++

En conclusion, ce projet avait pour objectif d'explorer la faisabilité de l'utilisation de l'apprentissage automatique pour la classification d'images de voitures et d'avions à partir de leurs caractéristiques visuelles. Les résultats obtenus ont montré que le Linear SVM était le meilleur classificateur avec une précision de 76%, ce qui indique que l'utilisation de l'apprentissage automatique pour la classification de ces images est prometteuse. Cependant, des limitations telles que la qualité des images, la diversité des angles de vue et des biais de classification potentiels doivent être pris en compte lors de l'utilisation de ce modèle dans la vie réelle

```{code-cell} ipython3

```
