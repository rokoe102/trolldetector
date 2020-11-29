# TrollDetector
This command line tool is used to apply five different classification techniques on a dataset consisting of troll and nontroll tweets.
## Installation

1\. Clone this repository
```
git clone https://github.com/rokoe102/trolldetector
```
2\. change the directory
```
cd path/to/directory/trolldetector
```
3\. run the installation command
```
pip install .
```

## Usage
### custom hyperparameters
If you wish to execute any classification technique with custom hyperparameters, execute the program in the following manner:

```
trolldetector [common options] {KNN, NB, SVM, tree, MLP} [specific options]
```
There are several options all techniques have in common:
- ``` --tfidf ``` changes the  feature weighting from TF to TF-IDF
- ``` --stopwords ``` activates english stopword filtering
- ``` --ngram <n> ``` changes the range of extracted ngrams to (1,n)
- ``` --dim <components>``` determines the level of dimensionality reduction
- ``` --test <(0,1)>``` changes the amount of the dataset used for testing
- ``` -v``` producing more detailed output
- ``` -h``` displays help message

KNN specific options:
- ```-k <value>``` determines the number of k nearest neighbors considered in computation (default: 5)
- ```-m {euclidean, manhattan}``` changes the metric used to compute distance (default: euclidean)

NB specific options:
- ``` -d {gaussian, multinomial, complement}``` changes the assumed distribution (default: gaussian)

SVM specific options:
- ```--cost <[0,1]>``` changes the degree of misclassification penalty (default: 1)

tree specific options:
- ```-m {gini, entropy}``` changes the metric used to determine the best split (default: gini)

MLP specific options:
- ```-a {relu, tanh, logistic}``` changes the activation function (default: relu)
- ```--iter <n>```tells the neuronal network to stop after n iterations of no change higher than tol (default: 5)
- ```--tol <f>```determines tolerance for stopping condition (default: 0.0025)

Example of execution with default hyperparameters:
```
trolldetector KNN
```
Examples of execution with custom hyperparameters:
```
trolldetector --tfidf -ngram 2 --dim 15 --test 0.2 NB -d multinomial
```
```
trolldetector --dim 2 --test 0.01 MLP -a tanh --iter 7 --tol 0.005
```

### hyperparameter optimization
To execute a hyperparameter optimization for any classification technique, put a ``` --optimize``` before the technique:
```
trolldetector --optimize KNN
```
Other options except ```-v``` and ```-h``` will be ignored.

### comparison of classifiers
If you wish to compare the performances of all the techniques, select the ```all``` command:
```
trolldetector all
```
Other options except ```-v``` and ```-h``` will be ignored.