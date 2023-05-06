import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pipe = make_pipeline(StandardScaler(), SVC())

pipe.fit(x_train, y_train)

st.set_page_config(page_title="Iris Flower Classification")
st.title("Iris Flower Classification App")

st.write("""
         Enter the measurements of iris flower to predict its species
         """)
sepal_length = st.slider('Sepal length', float(iris.data[:,0].min()), float(iris.data[:,0].max()), float(iris.data[:,0].mean()))
sepal_width = st.slider('Sepal width', float(iris.data[:,1].min()), float(iris.data[:,1].max()), float(iris.data[:,1].mean()))
petal_length = st.slider('Petal length', float(iris.data[:,2].min()), float(iris.data[:,2].max()), float(iris.data[:,2].mean()))
petal_width = st.slider('Petal width', float(iris.data[:,3].min()), float(iris.data[:,3].max()), float(iris.data[:,3].mean()))

def prediction():
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = pipe.predict(input_data)
    species = iris.target_names[prediction][0]
    st.text(f"Prediction: {species}")

st.button("Predict", on_click=prediction)



