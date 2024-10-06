import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

def preprocess_input_data(sex, age, pclass, sibsp, parch, fare, embarked):
    sex = 0 if sex == 'male' else 1
    
    embarked_dict = {'C': 1, 'Q': 2, 'S': 0}
    embarked = embarked_dict.get(embarked, 0) 

    input_data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],  
        'Parch': [parch],  
        'Fare': [fare],   
        'Embarked': [embarked]
    })
    
    return input_data

titanic_data = pd.read_csv('C:/Users/MOHAMMED RIFAIZ/OneDrive/Desktop/Codsoft/Titanic-Dataset.csv')

titanic_data['Age'].fillna(titanic_data['Age'].median(), inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data['Fare'].fillna(titanic_data['Fare'].median(), inplace=True)

titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})

titanic_data = pd.get_dummies(titanic_data, columns=['Embarked'], drop_first=True)

X = titanic_data.drop(['Survived', 'Name', 'Ticket', 'Cabin'], axis=1)  
y = titanic_data['Survived']
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

st.title("Titanic Survival Prediction")
st.sidebar.header("User Input Parameters")

passenger_id = st.sidebar.selectbox("Passenger ID", list(range(1, 892)))  
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["male", "female"])
age = st.sidebar.selectbox("Age", list(range(0, 101)))  
sibsp = st.sidebar.selectbox("Siblings/Spouses Aboard", list(range(0, 11)))  
parch = st.sidebar.selectbox("Parents/Children Aboard", list(range(0, 11)))  
fare = st.sidebar.selectbox("Fare", [round(f, 2) for f in np.arange(0.0, 513.0, 0.5)])  

embarked = st.sidebar.selectbox("Embarked", ["C", "Q", "S"])

input_data = preprocess_input_data(sex, age, pclass, sibsp, parch, fare, embarked)

input_data = input_data.reindex(columns=X.columns, fill_value=0)

st.write("Input Data for Prediction:")
st.write(input_data)

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write("Survived: Yes" if prediction[0] == 1 else "Survived: No")
