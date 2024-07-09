---
jupytext:
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

# Extraction d'attributs

+++

Les traitements réalisés dans cette feuille sont majoritairement des
rappels des semaines précédentes. **Le but de cette feuille est que
vous puissiez appliquer facilement et rapidement les traitements vus
les semaines précédentes afin de vous consacrer à l'interprétation et
l'application de ces traitements sur vos données.**

La feuille comprends trois parties:
1. Le prétraitement de vos données par extraction d'avant plan,
   recentrage et recadrage (rappels de la semaine 6)
2. L'extraction d'attributs ad-hoc (rappels de la semaine 4)
3. La sélection d'attributs pertinents

+++

Dans un premier temps, vous exécuterez cette feuille sur le jeu de
données de pommes et de bananes. Puis vous la reprendrez avec votre
propre jeu de données, en visualisant les résultats à chaque étape, et
en ajustant comme nécessaire.

+++

## Prétraitement (rappels de la semaine 6)

+++

Lors de la Semaine 6, nous avons prétraité les images de manière naïve
par une recadrage en 32x32 centré sur l'image.

Nous allons améliorer cela en détectant le centre du fruit et en
recadrant l'image sur ce centre.

Nous enregistrerons les images prétraitées dans un dossier
`clean_data`, et une table des pixels dans `clean_data.csv`.

+++

### Import des bibliothèques

+++

Nous commençons par importer les bibliothèques dont nous aurons
besoin. Comme d'habitude, nous vous fournissons quelques utilitaires
dans le fichier `utilities.py`. Vous pouvez ajouter vos propres
fonctions au fur et à mesure du projet:

```{code-cell} ipython3
# Automatically reload code when changes are made
%load_ext autoreload
%autoreload 2
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
%matplotlib inline
import scipy
from scipy import signal
import pandas as pd
import seaborn as sns
from glob import glob as ls
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score as sklearn_metric

from utilities import *
from intro_science_donnees import data
from intro_science_donnees import *
```

```{code-cell} ipython3

```

Pour avoir une première idée de l'impact des prétraitements sur la
classification, nous calculerons au fur et à mesure les performances
d'un classificateur -- ici du plus proche voisin kNN -- et nous les
stockerons dans une table de données `performances`:

```{code-cell} ipython3
sklearn_model = KNeighborsClassifier(n_neighbors=3)
performances = pd.DataFrame(columns = ['Traitement', 'perf_tr', 'std_tr', 'perf_te', 'std_te'])
performances
```

+++ {"user_expressions": []}

### Import des données

+++ {"user_expressions": []}

En mettant en commentaire la ligne 2 puis la ligne 1, vous choisirez
ici sur quel jeu de données le prétraitement s'applique: d'abord les
pommes et les bananes, puis le vôtre.

```{code-cell} ipython3
#dataset_dir = os.path.join(data.dir, 'ApplesAndBananas')
dataset_dir = 'data'

images = load_images(dataset_dir, "*.jpg")
```

+++ {"user_expressions": []}

Comme le jeu de données peut être gros, nous extrayons un échantillon
de dix pommes et dix bananes pour expérimenter et visualiser :

```{code-cell} ipython3
sample = list(images[:10]) + list(images[-10:])
```

```{code-cell} ipython3
image_grid(sample)
```

```{code-cell} ipython3
df_test = images.apply(contours)
df_test
plt.imshow(df_test[14])
```

+++ {"user_expressions": []}

Calculons les performances de notre classificateur en l'appliquant
directement à la représentation en pixels des images :

```{code-cell} ipython3
df_raw = images.apply(image_to_series)
df_raw['class'] = df_raw.index.map(lambda name: 1 if name[0] == 'a' else -1)
```

```{code-cell} ipython3
# Validation croisée
p_tr, s_tr, p_te, s_te = df_cross_validate(df_raw, sklearn_model, sklearn_metric)
metric_name = sklearn_metric.__name__.upper()
print("AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_tr, s_tr))
print("AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_te, s_te))
```

+++ {"user_expressions": []}

Ajoutons ces résultats à notre table `performances` :

```{code-cell} ipython3
performances.loc[0] = ["Images brutes", p_tr, s_tr, p_te, s_te]
performances.style.format(precision=2).background_gradient(cmap='Blues')
```

+++ {"user_expressions": []}

### Extraction de l'avant-plan

+++ {"user_expressions": []}

Pour trouver le centre du fruit, il faut déjà arriver à séparer les
pixels qui appartiennent au fruit de ceux qui appartiennent au
fond. Souvenez-vous, vous avez déjà fait cela en Semaine 6 avec la
fonction `foreground_filter`.

Dans l'exemple suivant, on utilise une variante
`foreground_redness_filter` qui sépare le fruit du fond en prenant en
compte la rougeur de l'objet et pas uniquement les valeurs de
luminosité.

**NB:** Comme pour `foreground_filter`, cette fonction a besoin d'un
seuil (compris entre 0 et 1) sur les valeurs des pixels à partir
duquel on décide s'il s'agit du fruit ou du fond. Avec les pommes et
les bananes, on se contentera de la valeur 2/3. Avec vos données,
faites varier cette valeur de seuil pour trouver la valeur qui semble
la plus adéquate.

```{code-cell} ipython3
image_grid([foreground_filter(img, theta=85)
            for img in sample])
```

+++ {"user_expressions": []}

On peut voir que selon s'il s'agit d'objets sombres sur fond clair ou
d'objets clairs sur fond sombre, on n'obtient pas les mêmes valeurs
booléenne en sortie de `foreground_filter`. La fonction
`invert_if_light_background` inverse simplement les valeurs booléennes
si une majorité de `True` est détectée. Voilà le résultat.

```{code-cell} ipython3
image_grid([invert_if_light_background(foreground_filter(img, theta=85))
            for img in sample])
```

+++ {"user_expressions": []}

# C'est légèrement mieux et nous nous en contenterons.

**Pour aller plus loin ♣**: Les images restent très bruitées; on peut
appliquer un filtre afin de réduire les pixels isolés:

```{code-cell} ipython3
image_grid([scipy.ndimage.gaussian_filter(
              invert_if_light_background(
                  foreground_filter(img, theta=85)),
              sigma=.1)
            for img in sample])
```

+++ {"user_expressions": []}

Les traitements précédents résultent d'un certain nombre de choix
faits au fil de l'eau (seuil, filtre par rougeur, débruitage).

La fonction suivante résume tous les choix faits pour extraire l'avant
plan: <a name="choix_extraction"></a>

```{code-cell} ipython3
def my_foreground_filter(img):
    foreground = foreground_filter(img, theta=85)
    foreground = invert_if_light_background(foreground)
    foreground = scipy.ndimage.gaussian_filter(foreground, sigma=.1)
    return foreground
```

+++ {"user_expressions": []}

Cette fonction fait partie intégrale de la **narration des
traitements** que nous menons dans cette feuille. C'est pour cela que
nous la définissons directement dans cette feuille, et non dans
`utilities.py` comme on l'aurait fait pour du code réutilisable.

+++ {"user_expressions": []}

### Détection du centre du fruit

+++ {"user_expressions": []}

Nous allons à présent calculer une estimation de la position du centre
du fruit, en prenant la moyenne des coordonnées des pixels de l'avant
plan.

Faisons cela sur la première image :

```{code-cell} ipython3
img = images[0]
img
```

+++ {"user_expressions": []}

On calcule l'avant plan de l'image et l'on extrait les coordonnées de
ses pixels :

```{code-cell} ipython3
foreground = my_foreground_filter(img)
coordinates = np.argwhere(foreground)
```

```{code-cell} ipython3
img
```

+++ {"user_expressions": []}

que l'on affiche comme un nuage de points :

```{code-cell} ipython3
plt.scatter(coordinates[:,1], -coordinates[:,0], marker="x");
```

+++ {"user_expressions": []}

On calcule maintenant le barycentre des pixels de l'avant plan --
c'est-à-dire la moyenne des coordonnées sur les X et les Y -- afin
d'estimer les coordonnées du centre du fruit :

```{code-cell} ipython3
center = (np.mean(coordinates[:,1]), np.mean(coordinates[:,0]))
center
```

```{code-cell} ipython3
plt.scatter(coordinates[:,1], -coordinates[:,0], marker="x");
plt.scatter(center[0], -center[1], 300, c='r', marker='+',linewidth=5);
```

+++ {"user_expressions": []}

Ce n'est pas parfait: du fait des groupes de pixels à droite qui ont
été détectées comme de l'avant plan, le centre calculé est plus à
droite que souhaité. Mais cela reste un bon début.

+++ {"user_expressions": []}

### Recadrage

+++ {"user_expressions": []}

Maintenant que nous avons (approximativement) détecté le centre du
fruit, il nous suffit de recadrer autour de ce centre. Une fonction
`crop_around_center` est fournie pour cela. Comparons le résultat de
cette fonction par rapport à l'ancienne fonction `crop_image` de la
semaine précédente :

```{code-cell} ipython3
crop_image(img) 
```

```{code-cell} ipython3
crop_around_center(img, center)
```

+++ {"user_expressions": []}

On constate que le recadrage sur le fruit est amélioré, même si pas
encore parfait.

+++ {"user_expressions": []}

### Récapitulatif du prétraitement

+++ {"user_expressions": []}

À nouveau, nous centralisons tous les choix faits au fil de l'eau en
une unique fonction effectuant le prétraitement<a
name="choix_pretraitement"></a>. Cela facilite l'application de ce
traitement à toute image et permet de documenter les choix faits :

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
plt.imshow(my_preprocessing(images[0]));
```

+++ {"user_expressions": []}

Appliquons le prétraitement à toutes les images :

```{code-cell} ipython3
clean_images = images.apply(my_preprocessing)
clean_sample = list(clean_images[:10]) + list(clean_images[-10:])
```

```{code-cell} ipython3
image_grid(clean_sample)
```

+++ {"user_expressions": []}

### Performance de la classification après prétraitement

+++ {"user_expressions": []}

Convertissons maintenant les images prétraitées dans leurs
représentations en pixels, regroupées dans une table:

```{code-cell} ipython3
# conversion
df_clean = clean_images.apply(image_to_series)
# ajout des étiquettes
df_clean['class'] = df_clean.index.map(lambda name: 1 if name[0] == 'a' else -1)
```

+++ {"user_expressions": []}

On vérifie les performance de notre classificateur sur les images
prétraitées et on les ajoute à notre table `performances` :

```{code-cell} ipython3
# Validation croisée (LENT)
p_tr, s_tr, p_te, s_te = df_cross_validate(df_clean, sklearn_model, sklearn_metric)
metric_name = sklearn_metric.__name__.upper()
print("AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_tr, s_tr))
print("AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_te, s_te))
```

```{code-cell} ipython3
performances.loc[1] = ["Images prétraitées", p_tr, s_tr, p_te, s_te]
performances.style.format(precision=2).background_gradient(cmap='Blues')
```

+++ {"user_expressions": []}

### Sauvegarde intermédiaire

+++ {"user_expressions": []}

Nous sauvegardons maintenant les images prétraitées dans le répertoire
`clean_data` au format `PNG` :

```{code-cell} ipython3
os.makedirs('clean_data', exist_ok=True)
for name, img in clean_images.items():
    img.save(os.path.join('clean_data', os.path.splitext(name)[0]+".png"))
```

+++ {"user_expressions": []}

**Explication :** `splitext` sépare un nom de fichier de son extension :

```{code-cell} ipython3
os.path.splitext("machin.jpeg")
```

+++ {"user_expressions": []}

Nous sauvegardons la table de données dans un fichier `clean_data.csv` :

```{code-cell} ipython3
df_clean.to_csv('clean_data.csv')
```

+++ {"user_expressions": []}

Ainsi il sera possible de travailler sur la suite de cette feuille et
les feuilles ultérieures sans avoir besoin de refaire le
prétraitement :

```{code-cell} ipython3
df_clean = pd.read_csv('clean_data.csv', index_col=0)
```

+++ {"user_expressions": []}

## Extraction des attributs (rappels de la semaine 4)

+++ {"user_expressions": []}

Durant les semaines précédentes, vous avez déjà implémenté des
attributs tels que :
- La rougeur (*redness*) et l'élongation durant la semaine 4;
- D'autres attributs (adhoc, pattern matching, PCA, etc.) durant le
  premier projet.

L'idée de cette section est de réappliquer ces attributs sur vos
nouvelles données.

+++ {"user_expressions": []}

### Filtres

+++ {"user_expressions": []}

Récapitulons les types de filtres que nous avons en magasin:

+++ {"user_expressions": []}

#### La rougeur (redness)

Il s'agit d'un filtre de couleur qui extrait la différence entre la
couche rouge et la couche verte (R-G). Lors de la Semaine 2, nous
avons fait la moyenne de ce filtre sur les fruits pour obtenir un
attribut (valeur unique par image). Ici, affichons simplement la
différence `R-G` :

```{code-cell} ipython3
image_grid([redness_filter(img)
            for img in clean_sample])
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "de257da85939f2134e56d5f7e043e9df", "grade": true, "grade_id": "cell-2ba20f803e981265", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}, "user_expressions": []}

**Exercice:** À quelles couleurs correspondent les zones claires
resp. sombres? Pourquoi le fond n'apparaît-il pas toujours avec la
même clarté?

VOTRE RÉPONSE ICI

+++ {"user_expressions": []}

#### Variante de la rougeur

Il s'agit d'un filtre de couleur qui extrait la rougeur de chaque
pixel calculée avec $R-(G+B)/2$ :

```{code-cell} ipython3
image_grid([difference_filter(img)
            for img in clean_sample])
```

+++ {"user_expressions": []}

Pour d'autres idées de mesures sur les couleurs, consulter [cette page
wikipédia](https://en.wikipedia.org/wiki/HSL_and_HSV).

+++ {"user_expressions": []}

#### Seuillage

+++ {"user_expressions": []}

Souvenez vous que vous pouvez également seuiller les valeurs des
pixels (par couleur ou bien par luminosité). C'est ce que l'on fait
dans les fonctions `foreground_filter` ou `foreground_color_filter`.

NB: N'oubliez pas de convertir les images en tableau `numpy` pour
appliquer les opérateurs binaires `<`, `>`, `==`, etc.

```{code-cell} ipython3
image_grid([np.mean(np.array(img), axis = 2) < 70
            for img in clean_sample])
```

+++ {"user_expressions": []}

#### Contours

+++ {"user_expressions": []}

Pour extraire les contours d'une image (préalablement seuillée), on
doit soustraire l'image seuillée avec elle même en la décalant d'un
pixel vers le haut (resp. à droite). On fourni cette extraction de
contours avec la fonction `contours` :

```{code-cell} ipython3
image_grid([contours(np.mean(np.array(img), axis = 2) < 100 ) 
            for img in clean_sample])
```

+++ {"user_expressions": []}

### Création d'attributs à partir des filtres

+++ {"user_expressions": []}

Maintenant que nous avons récapitulé les filtres en notre possession,
nous allons calculer un large ensemble d'attributs sur nos images. Une
fois cette ensemble recueilli, nous allons ensuite sélectionner
uniquement les attributs les plus pertinents.

On se propose d'utiliser trois attributs sur les couleurs:

1. `redness` : moyenne de la différence des couches rouges et vertes
   (R-G), en enlevant le fond avec `foreground_color_filter`;
2. `greenness` : La même chose avec les couches (G-B);
3. `blueness` : La même chose avec les couches (B-R).

Ainsi que trois autres attributs sur la forme:

4. `elongation` : différence de variance selon les axes principaux des
   pixels du fruits (cf Semaine2);
5. `perimeter` : nombre de pixels extraits du contour;
6. `surface` : nombre de pixels `True` après avoir extrait la forme.

```{code-cell} ipython3
df_features = pd.DataFrame({'redness': clean_images.apply(redness),
                            'greenness': clean_images.apply(greenness),
                            'blueness': clean_images.apply(blueness),
                            'elongation': clean_images.apply(elongation),
                            'perimeter': clean_images.apply(perimeter),
                            'surface': clean_images.apply(surface)})
df_features
```

+++ {"user_expressions": []}

**Exercice:** Ajouter les attributs que vous avez implémenté dans le
projet précédent.

+++ {"user_expressions": []}

Les amplitudes des valeurs sont très différentes. Il faut donc
normaliser ce tableau de données (rappel de la semaine 7) afin que les
moyennes des colonnes soient égales à 0 et les déviations standard des
colonnes soient égales à 1.

Rappel: notez l'utilisation d'une toute petite valeur epsilon pour
éviter une division par 0 au cas où une colonne soit constante :

```{code-cell} ipython3
epsilon = sys.float_info.epsilon
df_features = (df_features - df_features.mean())/(df_features.std() + epsilon) # normalisation 
df_features.describe() # nouvelles statistiques de notre jeu de donnée
```

+++ {"user_expressions": []}

On ajoute nos étiquettes (1 pour les pommes, -1 pour les bananes) dans
la dernière colonne :

```{code-cell} ipython3
df_features["class"] = df_clean["class"]
```

+++ {"user_expressions": []}

Et des valeurs par défaut:

```{code-cell} ipython3
df_features[df_features.isna()] = 0
```

```{code-cell} ipython3
df_features.style.background_gradient(cmap='coolwarm')
```

+++ {"user_expressions": []}

On vérifie les performance de notre classificateur sur les attributs
et on les ajoute à notre table `performances` :

```{code-cell} ipython3
# Validation croisée (LENT)
p_tr, s_tr, p_te, s_te = df_cross_validate(df_features, sklearn_model, sklearn_metric,)
metric_name = sklearn_metric.__name__.upper()
print("AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_tr, s_tr))
print("AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_te, s_te))
```

```{code-cell} ipython3
performances.loc[2] = ["6 attributs ad-hoc", p_tr, s_tr, p_te, s_te]
performances.style.format(precision=2).background_gradient(cmap='Blues')
```

## Sélection des attributs (**nouveau!**)

+++

Maintenant que nous avons extrait un ensemble d'attributs, nous
souhaitons analyser lesquels améliorent le plus les performances de
notre classificateur. Pour cela, nous tenterons deux approches :

- **Analyse de variance univariée** : On considère que les attributs
  qui, pris individuellement, corrèlent le plus avec nos étiquettes
  amélioreront le plus la performance une fois groupés.
- **Analyse de variance multi-variée** : On considère qu'il existe un
  sous-ensemble d'attributs permettant d'améliorer davantage les
  performances que les attributs étudiés séparément.

+++

### Analyse de variance univariée

+++

Dans cette approche, on commence par calculer les corrélations de
chacun de nos attributs avec les étiquettes :

```{code-cell} ipython3
# Compute correlation matrix
corr = df_features.corr()
corr.style.format(precision=2).background_gradient(cmap='coolwarm')
```

+++ {"user_expressions": []}

Pour les pommes et les bananes, seule la "blueness" resp. la "redness"
a une légère correlation resp. anti-corrélation avec les
étiquettes. Nous allons rajouter de nouveaux attributs sur les
couleurs afin d'identifier s'il y aurait un attribut qui corrélerait
davantage avec les étiquettes.

**NB**: On en profite pour renormaliser en même temps que l'on ajoute
des attributs.

```{code-cell} ipython3
clean_images.apply(get_colors)
```

```{code-cell} ipython3
s = df_features.iloc[3].replace({'redness': 4})
```

```{code-cell} ipython3
df_features
```

```{code-cell} ipython3
dict([[1,2], [4,5]])
```

```{code-cell} ipython3
header = ['R','G','B','M=maxRGB', 'm=minRGB', 'C=M-m', 'R-(G+B)/2', 'G-B', 'G-(R+B)/2', 'B-R', 'B-(G+R)/2', 'R-G', '(G-B)/C', '(B-R)/C', '(R-G)/C', '(R+G+B)/3', 'C/V']

df_features_large = df_features.drop("class", axis = 1)

df_features_large = pd.concat([df_features_large, clean_images.apply(get_colors)], axis=1)

epsilon = sys.float_info.epsilon # epsilon
df_features_large = (df_features_large - df_features_large.mean())/(df_features_large.std() + epsilon) # normalisation 
df_features_large[df_features_large.isna()] = 0
df_features_large.describe() # nouvelles statistiques de notre jeu de donnée
    
    
df_features_large["class"] = df_clean["class"]
df_features_large
```

+++ {"user_expressions": []}

On vérifie les performance de notre classificateur sur ce large
ensemble d'attributs et on les ajoute à notre table `performances` :

```{code-cell} ipython3
# Validation croisée (LENT)
p_tr, s_tr, p_te, s_te = df_cross_validate(df_features_large, sklearn_model, sklearn_metric)
metric_name = sklearn_metric.__name__.upper()
print("AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_tr, s_tr))
print("AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_te, s_te))
```

```{code-cell} ipython3
performances.loc[3] = ["23 attributs ad-hoc", p_tr, s_tr, p_te, s_te]
performances.style.format(precision=2).background_gradient(cmap='Blues')
```

```{code-cell} ipython3
# Compute correlation matrix
corr_large = df_features_large.corr()
corr_large.style.format(precision=2).background_gradient(cmap='coolwarm')
```

+++ {"user_expressions": []}

Dans l'approche univariée, les attributs qui nous intéresse le plus
sont ceux qui ont une grande corrélation en valeur absolue avec les
étiquettes. Autrement dit, les valeurs très positives (corrélation) ou
très négatives (anti-corrélation) de la dernière colonne sont
intéressants pour nous.

+++ {"user_expressions": []}

On va donc ordonner les attributs qui corrèlent le plus avec nos
étiquettes (en valeur absolue) :

```{code-cell} ipython3
# Sort by the absolute value of the correlation coefficient
sval = corr_large['class'][:-1].abs().sort_values(ascending=False)
ranked_columns = sval.index.values
print(ranked_columns) 
```

+++ {"user_expressions": []}

Sélectionnons seulement les cinq premiers attributs et visualisons
leur valeurs dans un pair-plot :

```{code-cell} ipython3
col_selected = ranked_columns[0:5]
df_features_final = pd.DataFrame.copy(df_features_large)
df_features_final = df_features_final[col_selected]

df_features_final['class'] = df_features_large["class"]
g = sns.pairplot(df_features_final, hue="class", markers=["o", "s"], diag_kind="hist")
```

+++ {"user_expressions": []}

### Trouver le nombre optimal d'attributs

+++ {"user_expressions": []}

On s'intéresse à présent au nombre optimal d'attributs. Pour cela, on
calcule les performances en rajoutant les attributs dans l'ordre du
classement fait dans la sous-section précédente (classement en
fonction de la corrélation avec les étiquettes).

```{code-cell} ipython3
# On importe notre modèle
from sklearn.metrics import balanced_accuracy_score as sklearn_metric
sklearn_model = KNeighborsClassifier(n_neighbors=3)
feat_lc_df, ranked_columns = feature_learning_curve(df_features_large, sklearn_model, sklearn_metric)
```

```{code-cell} ipython3
#feat_lc_df[['perf_tr', 'perf_te']].plot()
plt.errorbar(feat_lc_df.index+1, feat_lc_df['perf_tr'], yerr=feat_lc_df['std_tr'], label='Training set')
plt.errorbar(feat_lc_df.index+1, feat_lc_df['perf_te'], yerr=feat_lc_df['std_te'], label='Test set')
plt.xticks(np.arange(1, 22, 1)) 
plt.xlabel('Number of features')
plt.ylabel(sklearn_metric.__name__)
plt.legend(loc='lower right');
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "790e646c27c05dcd65ab7e28a0e3e183", "grade": true, "grade_id": "cell-4996b8b770bbb95c", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}, "user_expressions": []}

**Exercice** Combien d'attributs pensez-vous utile de conserver?
Justifiez.

Il serait mieux de laisser 5 attributs puisque le test set est decroissant d'apres 5.

+++ {"user_expressions": []}

On pourra exporter un nouveau fichier CSV appelé `features_data.csv`
contenant les attributs utiles. Pour l'exemple, nous exporterons les
cinq premiers attributs comme dans l'exemple plus haut :

```{code-cell} ipython3
df_features_final.to_csv('features_data.csv') # export des données dans un fichier
#df_features_final = pd.read_csv('features_data.csv')  # chargement du fichier dans le notebook
```

+++ {"user_expressions": []}

Enfin, on ajoute les performance de notre classificateur sur ce
sous-ensemble d'attributs sélectionnées par analyse de variance
univariée et on les ajoute à notre tableau de données `performances` :

```{code-cell} ipython3
# Validation croisée
p_tr, s_tr, p_te, s_te = df_cross_validate(df_features_final, sklearn_model, sklearn_metric)
metric_name = sklearn_metric.__name__.upper()
print("AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_tr, s_tr))
print("AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_te, s_te))
```

```{code-cell} ipython3
performances.loc[4] = ["5 attributs par analyse de variance univarié", p_tr, s_tr, p_te, s_te]
performances.style.format(precision=2).background_gradient(cmap='Blues')
```

+++ {"user_expressions": []}

### ♣ Analyse de variance multi-variée

+++ {"user_expressions": []}

La seconde approche est de considérer les attributs de manière groupés
et non pas par ordre d'importance en fonction de leur corrélation
individuelle avec les étiquettes. Peut-être que deux attributs, ayant
chacun une faible corrélation avec les étiquettes, permettront une
bonne performance de classification pris ensemble.

Pour analyser cela, on considère toutes les paires d'attributs et on
calcule nos performances avec ces paires :

```{code-cell} ipython3
best_perf = -1
std_perf = -1
best_i = 0
best_j = 0
for i in np.arange(5): 
    for j in np.arange(i+1,5): 
        df = df_features_large[[ranked_columns[i], ranked_columns[j], 'class']]
        p_tr, s_tr, p_te, s_te = df_cross_validate(df_features_large, sklearn_model, sklearn_metric)
        if p_te > best_perf: 
            best_perf = p_te
            std_perf = s_te
            best_i = i
            best_j = j
            
metric_name = sklearn_metric.__name__.upper()
print('BEST PAIR: {}, {}'.format(ranked_columns [best_i], ranked_columns[best_j]))
print("AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f}".format(metric_name, p_te, s_te))
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "25e7d599b01c542e5d5474c59fce0a69", "grade": true, "grade_id": "cell-3bc0c874f5b7944c", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}, "user_expressions": []}

**Exercice:** Quelle est la paire d'attributs qui donne les meilleurs
performances? Est-ce que l'approche multi-variée est nécessaire avec
les pommes et les bananes? Avec votre jeu de données?

VOTRE RÉPONSE ICI

+++ {"user_expressions": []}

## Conclusion

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "8c915152460e526409a66869cfb03130", "grade": true, "grade_id": "cell-c5a8e3d90bxe5375", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}, "user_expressions": []}

Cette feuille a fait un tour d'horizon d'outils à votre disposition
pour le prétraitement de vos images et l'extraction
d'attributs. Prenez ici quelques notes sur ce que vous avez appris,
observé, interprété.

VOTRE RÉPONSE ICI

Passez ensuite à la feuille sur la [comparaison de
classificateurs](4_classificateurs.md)!

```{code-cell} ipython3

```
