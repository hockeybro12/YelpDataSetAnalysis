This program implements a Naive Bayes classification algorithm that is evaluated on the Yelp dataset. We are trying to predict whether a given restaurant is good for groups based on the attributes provided in the dataset as features. 


This program can be run like this:
`python3 naive_bayes.py train-set1.csv test-set1.csv`

The program will output the zero_one_loss and squared_loss. With the provded train-set and test-set files, the output is:
```
ZERO_ONE_LOSS:
0.1279
SQUARED LOSS:
0.1137
```

Make sure you have numpy, random, sys, and pandas installed with python3. 