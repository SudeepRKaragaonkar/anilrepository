from sklearn.datasets import load_iris
import pandas as pd

df = pd.DataFrame(load_iris().data, columns=load_iris().feature_names)
df.to_csv("iris.csv", index=False)

