import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
# adding lib
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
import sklearn
ps = PorterStemmer()


def text_transform(text):
    # lower case
    text = text.lower()

    # tokenziation
    text = nltk.word_tokenize(text)

    # removing special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # removing stop words and punctuation
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # stemming
    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf = pickle.load(open('Task_2/Email-Spam-Detection-Webapp/vectorizer.pkl','rb'))
model = pickle.load(open('Task_2/Email-Spam-Detection-Webapp/model.pkl','rb'))

st.title('Email Spam Detection')
from PIL import Image

image = Image.open('Task_2/Email-Spam-Detection-Webapp/email_spam.jpg')

st.image(image)


input_mail = st.text_area('Enter text below to check if it is a spam email or not.')


if st.button('Predict'):


    # preprocess the text
    mail_transform = text_transform(input_mail)

    vector = tfidf.transform([mail_transform])

    # predict
    result = model.predict(vector)[0]

    # result
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not a Spam')

