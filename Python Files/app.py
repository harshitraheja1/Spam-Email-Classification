# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import nltk
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#%%
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word not in string.punctuation]
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens]
    return " ".join(tokens)

#%%
loaded_model=pickle.load(open("E:\\upes\\Summer internship Celebal Technologies\\trained_lr.sav",'rb'))
loaded_model_vect=pickle.load(open("E:\\upes\\Summer internship Celebal Technologies\\trained_vector.sav",'rb'))
loaded_model_dt=pickle.load(open("E:\\upes\\Summer internship Celebal Technologies\\trained_dt.sav",'rb'))
loaded_model_nb=pickle.load(open("E:\\upes\\Summer internship Celebal Technologies\\trained_nb.sav",'rb'))
loaded_model_svm=pickle.load(open("E:\\upes\\Summer internship Celebal Technologies\\trained_svm.sav",'rb'))
#%%
with st.sidebar:
    st.title('SPAM EMAIL CLASSIFICATION')
    st.markdown('''
    ### ABOUT
    - Filters the Subject of the Mail.
    - Real-time updates and learning
    - Multiple layered analysis to provide accurate detection''')
#%%  
def main():
    st.title("Spam Email Classifier")
    st.subheader("Enter the email Subject to check if it's spam or ham:")


    user_input = st.text_area("Enter email text here:", "")

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        processed_input = preprocess_text(user_input)
        input_vector = loaded_model_vect.transform([processed_input])
        
    col1, col2, col3, col4= st.columns([1,1,1,1])
    with col1:
        if st.button("Logistic Regression Prediction"):
            prediction_lr = loaded_model.predict(input_vector)[0]
            if prediction_lr==0:
                st.write("The email belongs to ham class.")
            else:
                st.write("The email belongs to spam class")

    with col2:
        if st.button("Naive-Bayes classifier Prediction"):
            prediction_nb =loaded_model_nb.predict(input_vector)[0]
            if prediction_nb==0:
                st.write("The email belongs to ham class.")
            else:
                st.write("The email belongs to spam class")
    with col3:
        if st.button("Decision Tree Prediction"):
            prediction_dt =loaded_model_dt.predict(input_vector)[0]
            if prediction_dt==0:
                st.write("The email belongs to ham class.")
            else:
                st.write("The email belongs to spam class")
    with col4:
        if st.button("Support Vector Machine Prediction"):
            inp=input_vector[0].toarray()
            prediction_svm =loaded_model_svm.predict(inp)
            if prediction_svm==0:
                st.write("The email belongs to ham class.")
            else:
                st.write("The email belongs to spam class")
#%%
if __name__ == "__main__":
    main()
