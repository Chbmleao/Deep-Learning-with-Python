import pandas as pd

predictors = pd.read_csv('input_breast.csv')
rank = pd.read_csv('output_breast.csv')

from sklearn.model_selection import train_test_split
trainingPredictors, testPredictors, trainingRank, testRank = train_test_split(predictors, rank, test_size=0.25)
