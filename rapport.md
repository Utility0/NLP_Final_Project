# Rapport Final, Analyse de Code - Desembiguation de Sens de Mot.

Julien ASSUIED - Cy-Tech ING3_Intelligence_Artificielle - Janvier 2023

## Introduction

Le but de ce projet est de completer et d'analyser un code de desambiguation de mot basés sur des études de traitement naturels du language.
Le code est présenté sous forme de notebook jupyter, et est disponible sur le github du projet [ici](https://github.com/Utility0/NLP_Final_Project).

Dans un premier temps, nous allons présenter les données utilisées dans le code.

Ensuite, nous allons présenter les différentes étapes de manipulation des données afin de les rendre utilisables dans le code.

Nous allons ensuite présenter la méthode de calcul du sens le plus fréquent.

Enfin, nous allons présenter 4 modèles de classification utilisés dans le code.


## Présentation des données

Les données utilisées proviennent de la base de données Asfalda, cette base à été créée par l'INRIA et l'Université de Grenoble. Elle contient 1.5 millions de phrases annotées par des humains. 

Ces données sont composées de 7 colonnes : 

 - sentid - Identifiant de la phrase

 - triggerid - Index du mot à désambiguer dans la phrase

 - trigger_frame - Sens du mot à désambiguer

 - trigger_lemma - Mot à désambiguer

 - trigger_pos - POS du mot à désambiguer

 - roles - Indication de la structure du mot à désambiguer dans la phrase

 - sentence - Phrase 

 Ces données nous permettent d'obtenir plusieurs informations sur le mot à désambiguer :

 - Le sens du mot est obtenu grace à la colonne `trigger_frame` qui contient le sens du mot sous forme d'un identifiant.

 - Le type de mot est obtenu grace à la colonne `trigger_pos` qui contient le POS (Part of Speech) du mot. Celui-ci prends différentes valeurs en fonction du type de mot, : 
     - `V` pour les verbes
     - `N` pour les noms
     - `A` pour les adjectifs
     - `P` pour les pronoms
     - `R` pour les adverbes
     - `C` pour les conjonctions
     - `D` pour les déterminants
     - `I` pour les interjections
     - `P` pour les prépositions
     - `T` pour les articles
     - `X` pour les autres mots 

## Manipulation des données

La première étape afin d'utiliser ces données est de les chargées dans un format utilisable dans le code.

Cette étape est réalisée par la fonction `load_asfalda_data`. Cette fonction prend en paramètre le chemin vers le fichier de données, et retourne quatres dictionnaire contenant les données qui nous interessent.

`sentences` : Dictionnaire contenant la liste des phrases segmentées en mots. Les clés du dict sont les différents types de corpus (`train`, `dev`, `test`, `val`).

`tg_wrks` : Dictionnaire contenant la liste d'index des mots à désambiguer dans les phrases. Les clés du dict sont les différents types de corpus (`train`, `dev`, `test`, `val`).


`tg_lemmas` : Dictionnaire contenant la liste des lemmes des mots à désambiguer dans les phrases. Les clés du dict sont les différents types de corpus (`train`, `dev`, `test`, `val`).

`labels` : Dictionnaire contenant la liste des sens des mots à désambiguer dans les phrases (obtenus dans la colonne `trigger_frame`). Les clés du dict sont les différents types de corpus (`train`, `dev`, `test`, `val`).

Une fois ces données chargées, nous pouvons les manipuler afin de créer d'autres données qui seront utilisées dans le code.

`lemma2i` et `i2lemma` : Dictionnaire et Array faisant le lien entre les lemmes et un index pour chaque lemme. 

`label2i` et `i2label` : Dictionnaire et Array faisant le lien entre les sens et un index pour chaque sens. 

## Calcul Most Frequent Sense

Le premier modèle que nous allons utiliser est le modèle de sens le plus fréquent. Ce modèle est basé sur le fait que le sens le plus fréquent d'un mot est le sens le plus probable.

Grâce à la fonction `frequence`, nous pouvons calculer le sens le plus fréquent de chaque lemme. Cette fonction retourne deux dictionnaires. Le premier `dict_lemmes` contient le nombre d'apparition de chaque sens pour chaque lemmes. le second, `dict_most_frequent` contient, pour chaque lemme, le sens le plus commun.

Afin de tester notre modèle, nous utilisons la fonction `baseline`. Cette fonction calcule le pourcentage de bonne prédiction ce qui nous permets d'avoir une information sur l'effience de notre modèle.

Nous obtenons des valeurs autour de 82%

## Modèles et Tokenization de type BERT

Nous allons utiliser un modèle de fine tuning de type BERT appelé FlauBERT. Nous pouvons obtenir ce model déjà pré-entrainé grâce à HuggingFace.

Afin d'utiliser ce même modèles, nos données doivent être similaires aux données d'entraînement d'un point de vue architectural.

### Tokenization

Pour se faire, nous utilisons la classe custom `WSDEncoder` qui utilise la librairie nltk, afin de suivre le même modèle de tokenisation.

Cette classe possède deux méthodes. La première, `encode` transforme une liste de phrase et l'index des mots à désambiguiser en liste de phrases tokenisés, compatible avec le modèle FlauBERT, ainsi qu'un liste d'index des premiers tokens correspondants aux mots pointés par les indexs d'entrés.

La deuxième méthode, `new_rank`, n'est utilisée que par `encode` et permet de calculer l'index des premiers tokens correspondants aux mots pointés par les indexs d'entrés.

### Création des données d'entrainement

Afin de créer les données d'entrainements, nous utilisons la classe custom `WSDData`, cette classe utilise le type de corpus (`train`, `dev`, `test`, `val`), la liste des phrases, la liste des index des mots à désambiguiser, la liste des lemmes des mots à désambiguiser et la liste des sens des mots à désambiguiser, le tout pour le corpus correspondant.

Cette classe possède la méthode `shuffle` qui permet de mélanger les données d'entrainement.

La seconde méthode, `make_batches`, permet de créer des batches de données d'entrainement. Cette méthode prend en paramètre la taille des batches, et retourne un générateur sur les trois tenseurs permettant d'alimenter le modèle.



### Création du modèle

Nous utilisons le modèle FlauBERT, qui est un modèle pré-entrainé de type BERT. Nous utilisons la classe custom `WSDClassifier` qui utilise le modèle FlauBERT, ainsi que la taille des tenseurs d'entrée et de sortie.

Nous initialisons cette classe avec les paramètres suivants :
- `num_labels` : entier indiquant le nombre de labels (ou classes) que la classe doit prédire.

- `device` : chaîne de caractères indiquant le type d'appareil utilisé pour l'entraînement.

- `bert_model_name` : chaîne de caractères indiquant le nom du modèle BERT utilisé.

- `freeze_bert` : booléen indiquant si les paramètres de BERT doivent être mis à jour lors de l'entraînement.

- `use_mlp` : booléen indiquant s'il faut utiliser un réseau de neurones multi-couches (MLP) pour la classification ou un simple perceptron.

- `hidden_size` : entier indiquant la taille cachée du MLP, utilisé uniquement si use_mlp est vrai.

- `nbr_lemmas` : entier indiquant le nombre de lemmes différents utilisés pour le modèle

- `lemma_embedding_size` : entier indiquant la taille de l'embedding pour les lemmes

- `add_lemmas` : booléen indiquant s'il faut ajouter les informations de lemma ou non dans le modèle

Cette classe possède la méthode `forward` qui permet de faire passer les tenseurs d'entrée dans le modèle et de retourner les tenseurs de sortie.

et la méthode `run_on_dataset` qui permet de lancer l'entraînement du modèle sur un dataset.

Une dernère methode `eval` permet d'évaluer le modèle sur différentes métriques. Tel que l'epoch loss, la val loss et la val accuracy.

### Entraînement du modèle

Une fois le modèle importé, nous pouvons continuer de l'entraîner avec nos données. Pour cela, nous utilisons la méthode `run_on_dataset` de la classe `WSDClassifier`. 

Pour analyser l'efficacité de notre modèle, nous utilisons les indicateurs `epoch_losses` et `val_accuracy` qui nous permettent de voir l'évolution de la perte et de la précision sur le jeu de validation.

Nous entraînons distinctement 4 modèles différents :

 - Un modèle basique.

 - Un modèle avec MLP lemmas et poids.

 - Un modèle avec poids.

 - Un modèle avec MLP poids.

Malhereusement, les limitations de ressources de Google Colab nous ont empêché d'entraîner les différents modèles par nous mêmes. Nous utiliserons donc les résultats présents dans le code pour comparer les différents modèles et conclure sur les résultats.

### Evaluation du modèle

Lors du fine-tuning, la méthode `run_on_dataset` retourne les indicateurs `epoch_losses` et `val_accuracy` qui nous permettent de voir l'évolution de la perte et de la précision sur le jeu de validation.

Les résultats obtenus sont les suivants :

| Modèle | Dev Accuracy | Test Accuracy |
| --- | --- | --- | 
| Basique | 0.84 | 0.85 |
| Poids | 0.85 | 0.86 |
| MLP et poids | 0.86 | 0.86 |
| MLP lemmas et poids | 0.86 | 0.87 | 


## Conclusion

Nous avons pu voir que la méthode de Most Frequent Sense est une méthode simple et efficace pour la désambiguïsation de mots.
Cependant, nous avons pu voir que cette méthode ne comporte qu'un préci de 82%, ce qui est loin d'être optimal.

L'utilisation de fine-tuning sur un modèle pré-entrainé comme FlauBERT nous a permis d'obtenir des résultats plus intéressants. Nous avons pu voir que l'utilisation de MLP lemmas et poids nous a permis d'obtenir un score de 87% sur le jeu de test, ce qui est un score très intéressant.

Afin d'améliorer les résultats, il serait intéressant d'entraîner le modèle sur un plus grand nombre de données, et d'ajouter des informations supplémentaires comme les POS tags ou les dépendances syntaxiques.

De plus, il serait intéréssant d'entaîner le modèle avec différents hyperparamètres afin de trouver les meilleurs paramètres pour le modèle.

