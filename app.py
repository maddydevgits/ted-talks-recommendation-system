import streamlit as st
import nltk
nltk.download('stopwords')
from model import *

st.title('Ted Talks Recommendation System')

inp=st.text_input('Enter Some ted talk from the dataset')

if st.button('Recommend some Ted Talks'):
    result=recommend_pearson(inp)
    st.write(result)
