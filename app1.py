import streamlit as st
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()
model = LogisticRegression(max_iter=200).fit(iris.data, iris.target)
st.title("Iris Prediction")

inputs = [st.slider(l, *v) for l, v in zip(
    ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
    [(4.0, 8.0, 5.1), (2.0, 4.5, 3.5), (1.0, 7.0, 1.4), (0.1, 2.5, 0.2)])]

if st.button('Predict'):
    st.success(f"Prediction: {iris.target_names[model.predict([inputs])[0]]}")