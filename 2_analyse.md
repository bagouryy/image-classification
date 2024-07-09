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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "f4fb5e4d189ee37bad5f15c104bac2c3", "grade": false, "grade_id": "cell-883bbb5e1919ca1e", "locked": true, "schema_version": 3, "solution": false, "task": false}}

# Analyse de données

Dans cette feuille, nous allons mener une première analyse sur des
données, afin d'obtenir une référence. Nous utiliserons d'abord les
données brutes (représentation en pixel), puis nous extrairons
automatiquement des attributs par Analyse en Composantes Principales.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "db5c526707cd89f13a44889cd668cae5", "grade": false, "grade_id": "cell-e5b1c2171c7f8c60", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Dans un premier temps, l'analyse sera faite sur le jeu de données des
pommes et des bananes, puis vous la relancerez sur votre propre jeu de
données.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "5505231ef795dc8e2187a9d01fe6baa9", "grade": false, "grade_id": "cell-e15fdd0116a9b9e2", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Import des librairies

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "37fab34064691bf8b1ca6806420171df", "grade": false, "grade_id": "cell-406ad2a159064cfa", "locked": true, "schema_version": 3, "solution": false, "task": false}}

On commence par importer les librairies dont nous aurons besoin. Comme
d'habitude, nous utiliserons un fichier `utilities.py` où nous vous
fournissons quelques fonctions et que vous complèterez au fur et à
mesure du projet:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 8b6606817a987853e15d8e7508ef2992
  grade: false
  grade_id: cell-76e64ee7bcb96bb5
  locked: true
  schema_version: 3
  solution: false
  task: false
---
# Automatically reload code when changes are made
%load_ext autoreload
%autoreload 2
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
%matplotlib inline
from scipy import signal
import seaborn as sns

from intro_science_donnees import *
from utilities import *
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "4573e3f162383e711fdca57544abff87", "grade": false, "grade_id": "cell-6ec03d05a4e1c693", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

## Chargement des images

En commentant la ligne 2 ou la ligne 3, vous choisirez ici sur quel
jeu de données vous travaillerez: les pommes et les bananes ou le
vôtre.

```{code-cell} ipython3
from intro_science_donnees import data
#dataset_dir = os.path.join(data.dir, 'ApplesAndBananas')
dataset_dir = 'data'

images = load_images(dataset_dir, "?[012]?.jpg")
```

```{code-cell} ipython3
show_source(load_images)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "e340ceddb7a3c40363bfae50e6f524ce", "grade": false, "grade_id": "cell-efcac6252de311fd", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

Nous allons aborder des jeux de données **beaucoup plus grands** que
les précédents, en nombre et en résolution des images; il faut donc
être **un peu prudent** et parfois **patient**. Par exemple, le jeu de
données des pommes et bananes contient plusieurs centaines
d'images. Cela prendrait du temps pour les afficher et les traiter
toutes dans cette feuille Jupyter. Aussi, ci-dessus, utilisons nous le
glob `?[012]?.png` pour ne charger que les images dont le nom fait
moins de trois caractères et dont le deuxième caractère est 0, 1, ou 2.

De même, dans les exemples suivants, nous n'afficherons que les
premières et dernières images :

```{code-cell} ipython3
head = images.head()
image_grid(head, titles=head.index)
```

```{code-cell} ipython3
tail = images.tail()
image_grid(tail, titles=tail.index)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a3468a13e1c73a96095bd154d014d8f9", "grade": false, "grade_id": "cell-e723e3fb5001bd97", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Étape 2: Redimensionner et recadrer

Comme vu au TP6, il est généralement nécessaire de redimensionner et/ou de recadrer les
images. DLors du premier projet, nous avons utilisé des images
recadrées pour se simplifier la tâche. Si l'objet d'intérêt est petit
au milieu de l'image, il est préférable d'éliminer une partie de
l'arrière-plan pour faciliter la suite de l'analyse. De même, si
l'image a une résolution élevée, on peut la réduire pour accélérer les
calculs sans pour autant détériorer les performances. Dans l'ensemble,
les images ne doivent pas dépasser 100x100 pixels.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "711a7a7a1f402497b41a5a07748a0dc6", "grade": false, "grade_id": "cell-1f5b70462bf6e53f", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Dans les cellules suivantes, on recadre et redimensionne les images en
32x32 pixels. Aucun effort particulier n'est fait pour centrer l'image
sur le fruit. Vous pourrez faire mieux ensuite, dans la fiche sur le
[prétraitement](3_extraction_d_attributs.md).

```{code-cell} ipython3
images_cropped = images.apply(crop_image)
```

```{code-cell} ipython3
image_grid(images_cropped.head())
```

```{code-cell} ipython3
plt.imshow(images_cropped[7])
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "658d05d7ba30f63d0eea685c3446d37f", "grade": false, "grade_id": "cell-1848d8ac9dce2aa3", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

## Étape 3: Représentation en pixels

Dans les cours précédents, nous avons extrait des attributs avec des
fonctions ad-hoc, comme la rougeur (*redness*) ou l'élongation. Les
attributs ad-hoc sont pratiques pour faciliter la visualisation des
données. On peut cependant obtenir d'assez bons résultats en partant
directement des données brutes -- ici les valeurs des pixels de nos
images recadrées -- et en extrayant automatiquement des attributs par
[décomposition en valeurs
singulières](https://fr.wikipedia.org/wiki/D%C3%A9composition_en_valeurs_singuli%C3%A8res),
à la base de la technique de
[PCA](https://fr.wikipedia.org/wiki/Analyse_en_composantes_principales). Cette
analyse, présentée ci-dessous est simple à mettre en œuvre et peut
donner une première base de résultats.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "031e3263ae21088dc0d9523cb86b5025", "grade": false, "grade_id": "cell-6c8ca7bdc9253abf", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

Vous trouverez dans `utilities.py` une fonction appelée
`image_to_series` qui transforme une image en une série
unidimensionnelle contenant les valeurs de tous les pixels de
l'image. En appliquant cette fonction à toutes les images, on obtient
un tableau de données où chaque ligne correspond à une image:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: a612f0aa6892caed701c54e5dd4daeda
  grade: false
  grade_id: cell-f376c04452c0a2c4
  locked: true
  schema_version: 3
  solution: false
  task: false
---
show_source(image_to_series)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: f3ad69fd3e0f42fcb610ebaff9cd418d
  grade: false
  grade_id: cell-e726f7c95a38b80b
  locked: true
  schema_version: 3
  solution: false
  task: false
---
df = images_cropped.apply(image_to_series)
df
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "de756b9c5df67fab6e3962df4c1e6272", "grade": true, "grade_id": "cell-edf8fc36ed15faf7", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}}

**Exercice:**  Pouvez-vous expliquer le nombre de colonnes que nous obtenons?

Oui, il ya 32x32 pixels donc 1024 et pour chaque pixel il ya 3 couleurs RGB. Il ya donc 3072 colonnes.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "1bb2f91f68025b739029f3e7e72a1597", "grade": false, "grade_id": "cell-bac63aa1dc5a364a", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Nous rajoutons à notre tableau de données une colonne contenant
l'étiquette (*ground truth*) de chaque image; dans le
jeu de données fournies, ce sera 1 pour une pomme et -1 pour une
banane:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 45dcbbd75dc72916ab9d2c5a71cb4e37
  grade: false
  grade_id: cell-ef9776de8892681b
  locked: true
  schema_version: 3
  solution: false
  task: false
---
df['étiquette'] = df.index.map(lambda name: 1 if name[0] == 'a' else -1)
df
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a0f0a7d04559dde1ab89a5ba75c9e671", "grade": false, "grade_id": "cell-b521e08d53e6f7e5", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

**Exercice:** Afficher les statistiques descriptives de Pandas sur votre base de données. Peut-on interpréter ces statistiques?

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: d9aecf0f1e72279e18e23c52f68f08c4
  grade: false
  grade_id: cell-265099e5a3fefb08
  locked: false
  schema_version: 3
  solution: true
  task: false
---
df.describe()
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "0b04fc60580d53165e039ab88751a746", "grade": false, "grade_id": "cell-a69bb602ef2221a4", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

## Sauvegarde intermédiaire

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "fcfa65f265bc5005f0f6dfbd091e52c8", "grade": false, "grade_id": "cell-8cb1846130b9cfe2", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

Nous allons sauvegarder ces données brutes dans un fichier:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 75453876738eaeed631d6014965c9b42
  grade: false
  grade_id: cell-2359fca3d8d7ac9b
  locked: true
  schema_version: 3
  solution: false
  task: false
---
df.to_csv('crop_data.csv')
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a0de1b361571fc89143f64d76d6c6d6f", "grade": false, "grade_id": "cell-c8e307c24e9822df", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

Cela vous permettra par la suite de reprendre la feuille à partir
d'ici, sans avoir à reexécuter tous les traitements depuis le début. Vous pourrez le faire pour votre projet aux moments que vous jugez opportuns.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 4eaef49f3c7ef47b7321ba7ea4753ffe
  grade: false
  grade_id: cell-59f5ff38837b92ad
  locked: true
  schema_version: 3
  solution: false
  task: false
---
df = pd.read_csv('crop_data.csv', index_col=0)
df
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "f250aaad53cded5510907c2fdb7dea2f", "grade": false, "grade_id": "cell-4576e2c2723b724d", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

## [VI]sualisation des données brutes par carte thermique

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "1b59c8053df96e34b00ee320edd5e87b", "grade": false, "grade_id": "cell-2a67648ea9c07906", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

Les données de la représentation en pixels sont volumineuses; par
principe il faut tout de même essayer de les visualiser. On peut pour
cela utiliser une carte thermique avec `Seaborn`.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: f3c870bca4130ab9236929eb09f24dab
  grade: false
  grade_id: cell-057378ec5d717db0
  locked: true
  schema_version: 3
  solution: false
  task: false
---
plt.figure(figsize=(20,20))
sns.heatmap(df, cmap="YlGnBu");
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "0678d196c74da7082fc94a1c44cba369", "grade": false, "grade_id": "cell-261308b1d940c89a", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

Comme vous le constatez, les données ne sont pas aléatoires: il y a
des corrélations entre les lignes. Cela reste cependant difficilement
exploitable visuellement.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ef9d44752be33ec160c85d16e15e9adc", "grade": false, "grade_id": "cell-c103279002f68467", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

## Étape 4: Performance de [RÉ]férence

À présent, nous allons appliquer la méthode des plus proches voisins sur la représentation en pixels pour avoir une performance de référence.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "3ec4ac8e3153841592b101733260e91b", "grade": false, "grade_id": "cell-c67679043304e3f1", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

On déclare le classifieur par plus proche voisin (KNN), depuis la
librairie `scikit-learn`.

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 1c40a2534a64d7032b31b7f3934a34d3
  grade: false
  grade_id: cell-4794ca5ee133a97b
  locked: true
  schema_version: 3
  solution: false
  task: false
---
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score as sklearn_metric
sklearn_model = KNeighborsClassifier(n_neighbors=3)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "6755637b8e61e36fa13483e02a060b06", "grade": false, "grade_id": "cell-f5d1905f8f1e86bf", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

Comme les séances précédentes, on calcule la performance et les barres
d'erreurs du classifieur par validation croisée (*cross validate*), en
divisant de multiples fois nos données en ensemble d'entraînement et
en ensemble de test.

La fonction `df_cross_validate` fournie automatise tout le
processus. Consultez son code pour retrouver les étapes:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 18a09476bc1299fb57649633a2d04405
  grade: false
  grade_id: cell-1b3c24803336e33f
  locked: true
  schema_version: 3
  solution: false
  task: false
---
show_source(df_cross_validate)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 156a4a7721cdf2f999902e582da677dc
  grade: false
  grade_id: cell-10d1cd02c980dc5b
  locked: true
  schema_version: 3
  solution: false
  task: false
---
p_tr, s_tr, p_te, s_te = df_cross_validate(df, sklearn_model, sklearn_metric, verbose=True)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "62ce352416dedeff46a2c6c49a6fb028", "grade": false, "grade_id": "cell-7a16870bc8b1d91d", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

On a obtenu cette performance en comparant nos images pixel à pixel
(distance euclidienne), sans aucun attribut.

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a211a7ef8290a24c18f439adf985494f", "grade": true, "grade_id": "cell-2b68d19d35bceb2f", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}, "user_expressions": []}

**Exercice:** Est-ce que ce score vous semble étonnant?

Le score qu'on a obtenu est de 68% de reussite donc pas tres bon.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "2ccd5d21c4e16e3002e18b313f690135", "grade": false, "grade_id": "cell-a93e6286afe7c468", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

## Réduction de la dimension par analyse en composants principales

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "4dd1aa9fb22a480cfd812e37a4d3263e", "grade": false, "grade_id": "cell-720d9a35a3d6ce4d", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

Si vous reprenez cette feuille à partir d'ici, vous pouvez charger les
données pour éviter d'avoir à tout reexécuter:

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 53781ef78f556cb1954a615aaffcc6fa
  grade: false
  grade_id: cell-c140b4ab67bf1e1d
  locked: true
  schema_version: 3
  solution: false
  task: false
---
df = pd.read_csv('crop_data.csv', index_col=0)
df
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "51cbfa67aaa1ba850da9823f50831c0e", "grade": false, "grade_id": "cell-79e2147e6deb20b3", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

Comme noté lorsque nous avons visualisé les données brutes, la
représentation en pixels contient beaucoup trop d'information pour
être directement exploitable: chaque ligne est un vecteur de grandes
dimensions ($32 \times 32 \times 3$). Il faut donc **réduire la dimension** en
synthétisant l'information contenue dans chaque ligne en un petit
nombre d'attributs. C'est ce que nous avions fait avec des attributs
ad-hoc. Nous allons cette fois utiliser la technique d'[Analyse en
Composantes
Principales](https://fr.wikipedia.org/wiki/Analyse_en_composantes_principales)
(*PCA*: Principal Component Analysis). Elle utilise un algorithme de
décomposition de matrices en valeur singulière (SVD) pour déterminer
les «directions principales» des données, c'est-à-dire là où elles ont
la plus grande variance, comme vu lors du [CM3](../Semaine3/CM3.md). Nous l'avions déjà utilisé lors du TP 3 et 4 pour
calculer l'élongation d'une forme.

Dans ce qui suit, vous pourrez vous aider des tutoriels que vous
pouvez trouver sur internet (par exemple
[ici](https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning)
ou
[là](https://cmdlinetips.com/2019/05/singular-value-decomposition-svd-in-python)),
et de la documentation de `pandas` et/ou de `numpy`.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "8aa016a50bf50b2c269368847d470365", "grade": false, "grade_id": "cell-b6ce9c809eb0bf2e", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

**Exercice:** Construisez un tableau `df_scaled` contenant les
colonnes de `df` après normalisation mais ne contenant pas les étiquettes. Vérifiez que la normalisation a
fonctionné en appelant `df.describe()` avant et après la
normalisation.

**Indications:**
- La dernière colonne de `df` contient les étiquettes; il faut donc
  l'ignorer. Pour celà, vous pouvez utiliser la méthode `drop`.
- Certaines colonnes pourraient être constantes, donc d'écart type
  nul. Dans ce cas, pour éviter une division par zéro, vous pouvez
  systématiquement ajouter la valeur `sys.float_info.epsilon` qui,
  comme son nom le suggère, est une toute petite valeur non nulle.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: fd9bc5b4a4cf00a8f605fbe64e4904e3
  grade: false
  grade_id: cell-162a0c66c26f9c65
  locked: false
  schema_version: 3
  solution: true
  task: false
---
import sys
df_scaled = (df - df.mean())/df.std()
df_scaled =df_scaled.drop(['étiquette'],axis=1)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 78f49e7fa408ac7a917a5d1cdd7c6b11
  grade: false
  grade_id: cell-de918321d3654071
  locked: true
  schema_version: 3
  solution: false
  task: false
---
df_scaled.describe()
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 8eb3ed2753741ba8abf9bc6dcfc84556
  grade: true
  grade_id: cell-cdafbcb6707e7900
  locked: true
  points: 2
  schema_version: 3
  solution: false
  task: false
---
assert df_scaled.shape[1] == 32 * 32 * 3
assert (abs(df_scaled.mean()) < 0.01).all()
assert (abs(df_scaled.std()-1) < 0.01).all()
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "b4a938e0be76c14a96e48134ce9bd951", "grade": false, "grade_id": "cell-6dc407ec0a37f340", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

**Exercice**: Effectuez une décomposition en valeurs singulières de
`df_scaled`. Stockez les matrices résultantes dans les variables `u`,
`s` et `v`.

**Indication** : Vous pourrez vous inspirer de la fonction `elongation` vu lors du TP3/4 et du projet 1.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 3ffb25347e2480b75dd733d4fda00719
  grade: false
  grade_id: cell-2dff66e19143b1d5
  locked: false
  schema_version: 3
  solution: true
  task: false
---
U, S, V = np.linalg.svd(df_scaled, full_matrices=True)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 4df100241993a5b195afded7e5d79bb2
  grade: false
  grade_id: cell-186b32b88482f0ea
  locked: true
  schema_version: 3
  solution: false
  task: false
---
U.shape
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: a8e9e8df52ec06fc531cf518b8871424
  grade: false
  grade_id: cell-f8c27d5c1249dc17
  locked: true
  schema_version: 3
  solution: false
  task: false
---
S.shape
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 8940f7ea210030caf7c6ea764bca1fc1
  grade: false
  grade_id: cell-9d219cdb7c3dd40c
  locked: true
  schema_version: 3
  solution: false
  task: false
---
V.shape
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "565fecc46a5d4390779c5ec88c018290", "grade": false, "grade_id": "cell-3f02c719b0a1dc65", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

Pour information (et qui vous servira pour le Projet 2):

- `U` est une matrice carrée dont les colonnes sont les attributs
  extraits automatiquement; chacun est une combinaison linéaire de
  colonnes de `df_scaled`. Il y en a toujours un grand nombre, mais
  ils sont triés par ordre d'intérêt: les premiers contiennent le plus
  d'information (axes de plus grande variance).

- `S` est un vecteur contenant *les valeurs singulières* correspondant
  aux attributs. Les carrés de celles-ci sont les *valeurs propres*;
  elles donnent l'importance de chaque attribut pour expliquer la
  variance des données.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "d0af7a3e4ab21e98432ed0ca234171d7", "grade": false, "grade_id": "cell-4cebd4f2ce937828", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

Un [scree
plot](https://en.wikipedia.org/wiki/Scree_plot#:~:text=In%20multivariate%20statistics%2C%20a%20scree,principal%20component%20analysis%20(PCA))
affiche toujours les valeurs propres dans une courbe descendante, en
classant les valeurs propres de la plus grande à la plus petite. Cette
courbe aide à sélectionner le nombre de vecteurs propres qui
expliquent la majorité de la variance des données.

**Exercice:** Afficher le scree plot des valeurs propres

**Indications:**
- Utilisez `plt.plot`
- Les valeurs singulières sont déjà triées de la plus grande à la plus
  petite
- N'utiliser que les 50 premières valeurs
- Renormaliser les valeurs propres par leur somme pour obtenir un
  pourcentage

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 929babf1b55d716f785832f2a77f02b3
  grade: false
  grade_id: cell-2b0bffda2b44a8c0
  locked: false
  schema_version: 3
  solution: true
  task: false
---
percent = 100 * S**2/np.sum(S**2)
plt.plot(list(range(20)), percent[:20])

plt.show
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "1bf6af8c6c9f30a56b1b12d35ee2e51d", "grade": false, "grade_id": "cell-4a804da27ed72447", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

D'après le scree plot obtenu, on voit que les premiers attributs
contiennent la majorité de l'information. Nous prendrons les
cinq premières.

**Exercice:** Créer un nouveau tableau de données `svd_df` avec les
cinq premières colonnes de `U` et l'étiquette du fruit dans la dernière
colonne.

**Indication:** Extraire les cinq premières colonnes de `U`, les
passer comme argument à `pd.DataFrame()` en spécifiant `index = df.index`.
Puis insérer la colonne `étiquette` de `df`.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: ebe12448393b884ae5138fa3eaf76f89
  grade: false
  grade_id: cell-3fd6e3f44337f707
  locked: false
  schema_version: 3
  solution: true
  task: false
---
U_reduced = U[:,:5]
svd_df = pd.DataFrame(U_reduced, index=df.index)
svd_df['étiquette'] = df['étiquette']
svd_df
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "c6155827763e3f5a72250e1a003b24ee", "grade": false, "grade_id": "cell-4d9a7b75517fa2e9", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice:** Ajoutez un assert et une condition if/else qui passerait pour votre propre jeu de données.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: 672ddacafd8c2890efdf19c84a4aa9b9
  grade: true
  grade_id: cell-40d78fb4393bd291
  locked: false
  points: 0
  schema_version: 3
  solution: true
  task: false
---
if dataset_dir == os.path.join(data.dir, 'ApplesAndBananas'):
    assert svd_df.shape == (58,6) 
elif dataset_dir == 'data':
    assert svd_df.shape == (20,6) 
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "6cec2d7e9bc3a70790b0948b0b404a22", "grade": false, "grade_id": "cell-c3d368d566dc7676", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

**Exercice:** Afficher des diagrammes de dispersion par paires des
attributs obtenus. Observez-vous que certaines caractéristiques ou
paires de caractéristiques séparent bien les données?

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 40449b8368e08eea0f85b5b3127ad102
  grade: false
  grade_id: cell-faeb04dc1e0af87f
  locked: true
  schema_version: 3
  solution: false
  task: false
---
g = sns.pairplot(svd_df, hue="étiquette", markers=["o", "s"], diag_kind="hist")
```

Non les caracteristiques ne sont pas bien separes

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "74f867f35647e5fad8213218191a41d0", "grade": false, "grade_id": "cell-31bc05cee7d026d8", "locked": true, "schema_version": 3, "solution": false, "task": false}}

**Exercice:** Calculer les performances obtenues avec la méthode des
trois plus proches voisins en utilisant les cinq premiers
attributs. Comparez les performances à celles obtenues avec la
représentation des pixels directement.

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: ef1158ffe5a966a47ad5a7a5ace37901
  grade: false
  grade_id: cell-99d52b0de49e5f4f
  locked: false
  schema_version: 3
  solution: true
  task: false
---
p_tr, s_tr, p_te, s_te = df_cross_validate(svd_df, sklearn_model, sklearn_metric, verbose=True)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "a6b8a616ee8b8596eebed342558d63d1", "grade": false, "grade_id": "cell-36f11e07b9930a24", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

**Pour aller plus loin $\clubsuit$:** Tracer le score de performance
(accuracy) sur l'ensemble de test en fonction du nombre d'attributs de
la PCA prise en compte par le classifieur (de 1 à 10 par exemple).

```{code-cell} ipython3
---
deletable: false
nbgrader:
  cell_type: code
  checksum: cd2d984619f1e103e3ad1cc1dc8b4bfc
  grade: false
  grade_id: cell-d41b42634ff70965
  locked: false
  schema_version: 3
  solution: true
  task: false
---
# VOTRE CODE ICI
raise NotImplementedError()
```

+++ {"deletable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ad7aaa7d31ca9544864f9b28bbb9cb7c", "grade": true, "grade_id": "cell-c5a8e3d90bb35375", "locked": false, "points": 0, "schema_version": 3, "solution": true, "task": false}}

## Conclusion

Cette première feuille vous fait passer par le formatage de base des
données et une analyse de données donnant une référence. Prenez ici
quelques notes sur ce que vous avez appris, observé, interprété.

VOTRE RÉPONSE ICI

Les feuilles suivantes aborderont d'autres aspects de l'analyse
(biais, prétraitement, classifieur). Ouvrez la feuille
sur le [prétraitement et l'extraction d'attributs](3_extraction_d_attributs.md).

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
