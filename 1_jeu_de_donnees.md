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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "218cab99bd1047beedc4741f6dd32fdd", "grade": false, "grade_id": "cell-7d5ac2d722ca418e", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

# Créer votre jeu de données

## Consignes

Vous allez collecter votre propre jeu de données d'images puis, de la même manière que dans le projet 1 apprendre à les classifier selon deux classes. Chaque
classe devra conteni entre 10 et 30 images chacunes, typiquement prises avec votre
téléphone portable.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "de8547b6efbdb5cdcf76e2b33d132243", "grade": false, "grade_id": "cell-1e200cdecbc22414", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

### Pour aller plus loin $\clubsuit$

Si vous êtes à l'aise et souhaitez aller plus loin, vous pouvez:

- Avoir plus de deux classes d'images.
- Traiter d'autres données que des images. 

Notez que, dans ce cas, vous pourrez être amenés à devoir adapter
fortement les feuilles que nous vous fournissons.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "0a075649f082eb8f92f9ccca0a9ea904", "grade": false, "grade_id": "cell-1e200cdecbc22415", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

## Préparation du jeu de données

*Conseil important: Choisissez des images qui sont toutes dans la même
orientation (carrée, portrait ou paysage) car avoir un mélange rend la
préparation des images plus complexe.*

Pour information, les images jpg ne prennent pas en compte la transparence. Les images jpg n'auront donc que 3 couches R, G et B alors que les png en auront 4.

Voilà comment préparer votre jeu de données :
- [ ] Déposez votre propre jeu de données dans un sous-dossier `data`
    dans `IntroSciencesDonnees/Semaine8`. Sur JupyterHub, vous pouvez
    utiliser le bouton `Téléverser` en haut à droite.

- [ ] Suivez les mêmes conventions de nommage que lors du premier projet:
    - `a01.jpg`, `a02.jpg`, ... pour les images de la première classe;
    - `b01.jpg`, `b02.jpg`, ... pour les images de la deuxième classe.

- [ ] Vérifiez la taille individuelle des images:

        cd ~/IntroScienceDonnees/Semaine8/data
		ls -lh

    Si elle dépasse 50~ko par image, réduisez la résolution pour
    limiter les temps de calcul et la taille de votre dépôt avant de
    démarrer le projet 2. Pour cela, utilisez la commande ci-dessous,
    disponible sur JupyterHub et en salles de TP :

        cd ~/IntroScienceDonnees/Semaine8/data
        mogrify -geometry 256x256 *.jpg 
       
    Vous pourrez ensuite les redimensionner et les recadrer en Python,
    comme vu la semaine dernière.

- [ ] Assurez-vous que le répertoire `data` ne contienne rien
    d'autre. Lorsque vous avez terminé, utilisez les commandes
    suivantes pour rajouter les fichiers dans votre dépôt git:
  
       cd ~/IntroScienceDonnees/Semaine8
       git add data

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "0454f7600a6154cb89ddb3da1c61e9bf", "grade": false, "grade_id": "cell-d5d94b152bdd3df8", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Chargement de votre jeu de données

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: bd5f0ad4b5f86790bbdccf1e29043af7
  grade: false
  grade_id: cell-98dd9e0867be908c
  locked: true
  schema_version: 3
  solution: false
  task: false
---
# Automatically reload code when changes are made
%load_ext autoreload
%autoreload 2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from intro_science_donnees import *
from utilities import *
```

```{code-cell} ipython3
dataset_dir = 'data'
images = load_images(dataset_dir, "*.jpg")
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ce0ec4237071bfe1e6b88f7f6226638c", "grade": false, "grade_id": "cell-6eec6709bb303693", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Combien d'images contient votre première série de données ? Stockez ce nombre dans une variable `n1`

```{code-cell} ipython3
n1 = 10 # Changez cette valeur
```

```{code-cell} ipython3
first = images[:n1]
image_grid(first, titles=first.index)
```

```{code-cell} ipython3
last = images[n1:]
image_grid(last, titles=last.index)
```

```{code-cell} ipython3
---
deletable: false
editable: false
nbgrader:
  cell_type: code
  checksum: 1df33b3dacdfcf9c14d2ce74e6a2f918
  grade: true
  grade_id: cell-062ab2c6cccf78d4
  locked: true
  points: 0
  schema_version: 3
  solution: false
  task: false
---
assert len(images) >= 20
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ef4ae4fc53cb9b133fdea98db18d232f", "grade": false, "grade_id": "cell-c3380759576c39ca", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Métadonnées

Il est important de noter qu'il est très difficile de collecter de
bonnes données qui ne soient pas entachées de biais. En vue de
rechercher et corriger de tels biais, vous devez essayer de conserver
le plus d'informations possible sur vos exemples : par exemple, le
lieu, la date et l'heure à laquelle les images ont été prises
etc. Heureusement, la plupart des téléphones ou appareils photos
récents capturent ce genre d'informations, encodées dans le fichier
d'image lui-même! Nous appelons cela les
[métadonnées](https://fr.wikipedia.org/wiki/M%C3%A9tadonn%C3%A9e).

En vous inspirant de la feuille sur les [biais](Semaine7/1_biais.md)
de la semaine 7, vous pourrez effectuer une analyse des biais dans vos
images. Pour cela vous devrez en extraire les métadonnées. Un mini
utilitaire vous est fournis dans `utilities.py` (regardez comment il
fonctionne!). Voici par exemple les métadonnées pour l'une des images
en JPEG de notre jeu de données sur les mélanomes :

```{code-cell} ipython3
from intro_science_donnees import data
img = Image.open(os.path.join(data.dir, 'Melanoma', 'a01.jpg'))
extract_metadata(img)
```

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "ef7f0462ff7a046f6e38947da04e5141", "grade": false, "grade_id": "cell-cb50119ba6b0ead8", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Fin

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "16dbde732834f2a8a3b8e9f61dee84dc", "grade": false, "grade_id": "cell-0f781d88612a087a", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Maintenant que vous avez préparé votre jeu de données, vous pouvez
reprendre l'analyse de la feuille [2_analyse](2_analyse.md) avec vos propres images.

```{code-cell} ipython3

```

```{code-cell} ipython3

```
