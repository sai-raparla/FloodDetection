import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


train = pd.read_csv('Data/train.csv')
test = pd.read_csv('Data/test.csv')

print(train.columns)
print(test.columns)


print(train.head())
print(test.head())
print(train.shape, test.shape)

sns.histplot(train["FloodProbability"], bins=50, kde=True)
plt.title("FloodProbability distribution")
plt.savefig("charts/class_distribution.png")
plt.show()