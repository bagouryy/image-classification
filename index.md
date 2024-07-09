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

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "c6b54348e775856d9c9f78d2f3f98e09", "grade": false, "grade_id": "cell-ebb8f94141812ddb", "locked": true, "schema_version": 3, "solution": false, "task": false}}

# Semaine 8

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "f3f2d46fae590c69bea03665443c0b13", "grade": false, "grade_id": "cell-ebb8f94141812ddc", "locked": true, "schema_version": 3, "solution": false, "task": false}}

## Objectifs

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "48f92d8a1abf93e2dee632c9a9ecbe27", "grade": false, "grade_id": "cell-d94d0b397fc433ff", "locked": true, "schema_version": 3, "solution": false, "task": false}}

Vous êtes désormais familiers avec le schéma VI-ME-RÉ-BAR --
[VI]sualisation, [ME]sure, [RÉ]férence (baseline), [BAR]res d'erreur,
et vous avez appris à classifier des jeux de données simples:

- Collecter et traiter des images (recentrage automatique, seuillage
  etc.).
- Extraire des attributs sur les couleurs (ad-hoc), la forme (ad-hoc
  ou matched filters) ou les pixels (PCA).
- Entraîner un classificateur (oneR, k-NN, arbre de décision, perceptron
  ou autres) sur vos données d'entraînement.
- Calculer un score de performance du classificateur sur vos données
  de test.

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "9cb9021531b977b2619745096271847b", "grade": false, "grade_id": "cell-3c07cc11be90a0d2", "locked": true, "schema_version": 3, "solution": false, "task": false}}

À l'occasion du second mini-projet, vous allez appliquer ces
connaissances et votre savoir faire à vos propres données, en
développant de nouveaux aspects de l'analyse (biais sur les
méta-données) ou en allant plus loin des les concepts déjà introduits
(PCA, nouveaux classificateurs).

+++ {"deletable": false, "editable": false, "nbgrader": {"cell_type": "markdown", "checksum": "59f31064e6fee8777e80c48c45850595", "grade": false, "grade_id": "cell-c8b352e502568e10", "locked": true, "schema_version": 3, "solution": false, "task": false}, "user_expressions": []}

Semaine 8 et 9 : dépôt le 11 avril à 21h, travail en binome
- Commencez par vérifier votre binome ici: https://codimd.math.cnrs.fr/m4cSohttSLyrAPl5Qoe5NQ?view Si il y a une erreur inscrivez la dans le tableau "Je ne suis pas correctement inscrit.e"
- Validation du jeu de données auprès de vos chargées et chargés de TP
  et import de celui-ci dans votre dépôt de la semaine 8. Préparation
  de votre jeu de données, prétraitement et analyse, retour sur les classificateurs.

Semaine 10 : Soutenances de projet
- La soutenance durera 7min suivies de 3min de questions pour chaque binome.
- Le contenu attendu de votre présentation est disponible dans la feuille 5_diaporama.md. Il consiste en: 
    * la présentation de votre jeu de données, 
    * la présentation de vos préparations de données et attributs choisis/ACP (montrez le code si cela est pertinent), 
    * la présentation des résultats (quel classificateur, taux d'erreur etc), 
    * la présentation des difficultés rencontrées, des biais possibles de vos images, 
    * une conclusion et des perspectives (à quoi pourrait servir votre projet dans notre société?)
- Votre horaire de passage et votre binome (le même qu'au projet 1) sont indiqués ici : https://codimd.math.cnrs.fr/m4cSohttSLyrAPl5Qoe5NQ?view 
- Posez vos questions relatives à la soutenance sur le forum: je ne répondrai pas par mail privé.

Le travail est à faire en binôme, le même que pour le premier
mini-projet sauf demande motivée auprès de vos chargées et chargés de
TP.

### Autres bonnes pratiques pour le projet

- Gardez une trace des expérimentations intermédiaires («nous avons essayé telle combinaison de features; voici les résultats et la performance obtenue»).  
- Vous mettrez dans le fichier `utilities.py` les utilitaires des TP
  précédents (`load_images`, ...)  que vous souhaiterez réutiliser,
  ainsi que vos nouvelles fonctions.
- Lorsque pertinent pour les explications, vous pourrez afficher le
  code de vos fonctions avec `show_source`.
- Complétez régulièrement le paragraphe de revue de code ci-dessous,
  pour qu'il affiche le code de toutes les fonctions que vous avez
  implantées. Vérifiez à chaque fois le résultat des outils de
  vérifications (`flake8`, ...).
- Lorsque vous aurez besoin de brouillon -- par exemple pour mettre au
  point du code -- créez des petites feuilles Jupyter séparées pour ne
  pas polluer votre document.


## Au travail!

1.  Commencez par vérifier votre binome ici: https://codimd.math.cnrs.fr/m4cSohttSLyrAPl5Qoe5NQ?view Si il y a une erreur inscrivez la dans le tableau "Je ne suis pas correctement inscrit.e"
2.  Ouvrez la feuille [jeu_de_donnees](1_jeu_de_donnees.md) et suivez
    les instructions pour préparer votre propre jeu de données.
3.  Ouvrez la feuille [analyse](2_analyse.md) et suivez les
    instructions pour lancer une première analyse sur le jeu de
    données fourni.
4.  Ouvrez la feuille [extraction
    d'attributs](3_extraction_d_attributs.md) et suivez les
    instructions pour prétraiter vos données et en
    extraire des attributs.
5.  Ouvrez la feuille [classificateur](4_classificateurs.md) et suivez
    les instructions pour faire une première classification.
6.  Itérez entre les feuilles précédentes pour améliorer les
    performances de votre classification.
7.  Préparer votre présentation. Elle devra présenter de façon
    synthétique votre jeu de données, vos expériences, vos
    observations, vos interprétations.

    Un squelette de [diaporama](5_diaporama.md) Jupyter vous est
    fourni; vous pouvez aussi produire un diaporama en pdf.

+++ {"user_expressions": []}

<div class="alert alert-info" role="alert">

Note: les feuilles Jupyter fournies sont très guidées. Si vous
souhaitez effectuer d'autres traitements et analyses, ou simplement
pour faire des essais, n'hésitez pas à utiliser de nouvelles feuilles
Jupyter libres. De même, vous pouvez ajouter d'autres fichiers, comme
des images pour illustrer votre diaporama.  Assurez-vous de signaler
ces fichiers supplémentaires à `git` pour qu'ils soient déposés avec
votre projet sur Gitlab. Par exemple, pour ajouter un fichier nommé
`truc.pdf` dans votre dépot, ouvrez un terminal, accéder au dossier de la Semaine où vous aurez mis truc.pdf puis tapez :

    cd ~/IntroScienceDonnees/Semaine8
    git add truc.pdf

</div>

+++

### Travail en binome et dépots

- [ ] Si vous ne l'avez pas encore fait : donnez-vous réciproquement
      les droits de lecture sur vos dépôts avec votre binôme. Si A et
      B sont les deux membres du binôme, alors A fait :

          cd ~/IntroScienceDonnees
          ./course.py share_with Semaine8 prenomB.nomB

      et B fait :

          cd ~/IntroScienceDonnees
          ./course.py share_with Semaine8 prenomA.nomA

- [ ] Indiquez à GitLab que vous travaillez en binôme. Pour cela, l'un
      des deux membres, ici A, fait :

          cd ~/IntroScienceDonnees
          ./course.py submit Semaine8 --leader_name prenomB.nomB

      Nous ne corrigerons que le devoir de B: à charge pour vous de
      vous assurer que celui-ci contient bien l'intégralité du travail
      du binôme! La note sera attribuée à A et à B.

+++

Bon Projet!
