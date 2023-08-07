
import streamlit as st
from streamlit_option_menu import option_menu

import sklearn
print(sklearn.__version__)

import pickle
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import  PorterStemmer 
import re

import pandas as pd
import nltk
from nltk.corpus import stopwords
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import pickle
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")
model = pickle.load(open('best_model.pkl','rb')) 
vectorizer = pickle.load(open('count_vectorizer.pkl','rb')) 

#FOR STREAMLIT
nltk.download('stopwords')

button_style = """
    <style>
    .stForm_submitButton {
        background-color: red; /* Change to your desired color */
        color: white;
        border-color: green;
    }
    </style>
"""

#TEXT PREPROCESSING
sw = set(stopwords.words('english'))

def ModelPrep():
    df = pd.read_csv("cleaned.csv",encoding="latin1") #due to special charas should be encoded as latin 1

    toCheck = pd.read_csv("updated.csv",encoding="latin1")
    #REMOVE MAX
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    #DROP EXTRA COLUMNS
    df.drop(['Unnamed: 0'], axis=1, inplace=True)
    toCheck.drop(['Unnamed: 0'], axis=1, inplace=True)

    #CHECKING WHICH ROW IS NULL FROM PRE-PROCESSING
    checkNULL = df.isnull()
    checkNULL = checkNULL.any(axis=1)

    df = df.dropna(how='any',axis=0) 
 
    df["verified_purchase"].value_counts(normalize=True)

    # MODELING
    #ASSIGN THE VARIABLES
    X = df['review_text'] #input var
    y = df['verified_purchase'] #target var

    #SPLIT DATA
    X_train, X_test, y_train, y_test = train_test_split(
        df['review_text'], df['verified_purchase'],test_size=0.4, random_state=42) #40% gives best results, 42 is no of life...

    entiredf = format(df.shape[0])
    traindf = format(X_train.shape[0])
    testdf = format(X_test.shape[0])

    count_vectorizer  = CountVectorizer(stop_words='english')
    count_vectorizer.fit(X_train)

    st.write("COUNT VECTORIZER AND MODELING")
    train_c = count_vectorizer.fit_transform(X_train)
    test_c = count_vectorizer.transform(X_test)

    # return X_train, X_test, train_c, test_c, y_train, y_test

    
    # Multinomial Naive Bayes model
    #IMPLEMENTING AND RUNNNING MNB MODEL - COUNT
    mnb1 = MultinomialNB()
    mnb1.fit(train_c, y_train)
    prediction = mnb1.predict(test_c)


    #EVALUATION
    mnb_a1 = accuracy_score(y_test, prediction)*100
    mnb_p1 = precision_score(y_test, prediction)* 100
    mnb_r1 = recall_score(y_test, prediction)*100
    mnb_f11 = f1_score(y_test, prediction)*100


    #CONFUSION MATRIX
    
    st.write("Naive Bays confusion matrix")
    cm =  confusion_matrix(y_test, prediction, labels=mnb1.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=mnb1.classes_) 
    display.plot()
    # plt.show()
    st.pyplot(plt)


    st.write("Support Vector Machine model")
    #IMPLEMENTING AND RUNNNING SVM MODEL - COUNT
    svm1 = SVC(kernel='linear')
    svm1.fit(train_c, y_train)
    prediction = svm1.predict(test_c)




    #EVALUATION
    svm_a1 = accuracy_score(y_test, prediction)*100
    svm_p1 = precision_score(y_test, prediction)* 100
    svm_r1 = recall_score(y_test, prediction)*100
    svm_f11 = f1_score(y_test, prediction)*100



    #CONFUSION MATRIX
    
    cm =  confusion_matrix(y_test, prediction, labels=svm1.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=svm1.classes_) 
    display.plot() 
    st.pyplot(plt)

    # ### Logistic Regression model
    st.write("Logistic Regression model")
    #IMPLEMENTING AND RUNNNING LR MODEL - COUNT
    lr1 = LogisticRegression()
    lr1.fit(train_c, y_train)
    prediction = lr1.predict(test_c)


    #EVALUATION
    lr_a1 = accuracy_score(y_test, prediction)*100
    lr_p1 = precision_score(y_test, prediction)* 100
    lr_r1 = recall_score(y_test, prediction)*100
    lr_f11 = f1_score(y_test, prediction)*100


    #CONFUSION MATRIX

    cm =  confusion_matrix(y_test, prediction, labels=lr1.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=lr1.classes_) 
    display.plot() 
    st.pyplot(plt)

    st.write("TFIDF VECTORIZER AND MODELING")

    TFIDF_vectorizer  = TfidfVectorizer(stop_words='english')

    TFIDF_vectorizer.fit(X_train)
    print('\nVocabulary: \n', TFIDF_vectorizer.vocabulary_)

    train_tf = TFIDF_vectorizer.fit_transform(X_train)
    test_tf = TFIDF_vectorizer.transform(X_test)


    # ### Multinomial Naive Bayes model

    st.write("Multinomial Naive Bayes model")
    #IMPLEMENTING AND RUNNING MNB MODEL - TFIDF
    mnb2 = MultinomialNB()
    mnb2.fit(train_tf, y_train)
    prediction = mnb2.predict(test_tf)


    #EVALUATION
    mnb_a2 = accuracy_score(y_test, prediction)*100
    mnb_p2 = precision_score(y_test, prediction)* 100
    mnb_r2 = recall_score(y_test, prediction)*100
    mnb_f12 = f1_score(y_test, prediction)*100


    #CONFUSION MATRIX
    cm =  confusion_matrix(y_test, prediction, labels=mnb2.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=mnb2.classes_) 
    display.plot() 
    st.pyplot(plt)


    # ### Support Vector Machine model

    st.write("Support Vector Machine model")
    #IMPLEMENTING AND RUNNING SVM MODEL - TFIDF 
    svm2 = SVC(kernel='linear')
    svm2.fit(train_tf, y_train)
    prediction = svm2.predict(test_tf)


    #EVALUATION
    svm_a2 = accuracy_score(y_test, prediction)*100
    svm_p2 = precision_score(y_test, prediction)* 100
    svm_r2 = recall_score(y_test, prediction)*100
    svm_f12 = f1_score(y_test, prediction)*100


    #CONFUSION MATRIX
    cm =  confusion_matrix(y_test, prediction, labels=svm2.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=svm2.classes_) 
    display.plot() 
    st.pyplot(plt)


    # ### Logistic Regression model

    st.write("Logistic Regression model")
    #IMPLEMENTATION AND RUNNING LR MODEL - TFIDF 
    lr2 = LogisticRegression()
    lr2.fit(train_tf, y_train)
    prediction = lr2.predict(test_tf)
    


    #EVALUATION
    lr_a2 = accuracy_score(y_test, prediction)*100
    lr_p2 = precision_score(y_test, prediction)* 100
    lr_r2 = recall_score(y_test, prediction)*100
    lr_f12 = f1_score(y_test, prediction)*100


    #CONFUSION MATRIX
    cm =  confusion_matrix(y_test, prediction, labels=lr2.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm,
                                display_labels=lr2.classes_) 
    display.plot() 
    st.pyplot(plt)


    st.write("COMPARING ACCURACY")


    model_accuracy={'MNB': [round(mnb_a1), round(mnb_a2)],
                    'SVM': [round(svm_a1), round(svm_a2)],
                    'LR': [round(lr_a1), round(lr_a2)]
                }
    ma = pd.DataFrame(model_accuracy, columns = ['MNB','SVM','LR'], index=['Count Vectorizer','Tfidf Vectorizer'])
    ma


    st.write("COMPARING PRECISION")


    model_precision={'MNB': [round(mnb_p1), round(mnb_p2)],
                    'SVM': [round(svm_p1), round(svm_p2)],
                    'LR': [round(lr_p1), round(lr_p2)]
                }
    mp = pd.DataFrame(model_precision, columns = ['MNB','SVM','LR'], index=['Count Vectorizer','Tfidf Vectorizer'])
    mp


    st.write("COMPARING RECALL")

    model_recall={'MNB': [round(mnb_r1), round(mnb_r2)],
                    'SVM': [round(svm_r1), round(svm_r2)],
                    'LR': [round(lr_r1), round(lr_r2)]
                }
    mr = pd.DataFrame(model_recall, columns = ['MNB','SVM','LR'], index=['Count Vectorizer','Tfidf Vectorizer'])
    mr


    st.write("COMPARING F1 SCORE")

    model_f1={'MNB': [round(mnb_f11), round(mnb_f12)],
                    'SVM': [round(svm_f11), round(svm_f12)],
                    'LR': [round(lr_f11), round(lr_f12)]
                }
    mf1 = pd.DataFrame(model_f1, columns = ['MNB','SVM','LR'], index=['Count Vectorizer','Tfidf Vectorizer'])
    mf1

    #SAVING THE BEST MODEL WITH ITS RESPECTIVE VECTORIZER
    pickle.dump(lr1, open('best_model.pkl', 'wb'))
    pickle.dump(count_vectorizer, open('count_vectorizer.pkl', 'wb'))

def text_preprocessing(text):
    txt = TextBlob(text)
    result = txt.correct()
    removed_special_characters = re.sub("[^a-zA-Z]", " ", str(result))
    tokens = removed_special_characters.lower().split()
    stemmer = PorterStemmer()
    
    cleaned = []
    stemmed = []
    
    for token in tokens:
        if token not in sw:
            cleaned.append(token)
            
    for token in cleaned:
        token = stemmer.stem(token)
        stemmed.append(token)

    return " ".join(stemmed)

def text_classification(text):
    if len(text) < 1:
        st.write("  ")
    else:
        with st.spinner("Classification in progress..."):
            cleaned_review = text_preprocessing(text)
            process = vectorizer.transform([cleaned_review]).toarray()
            prediction = model.predict(process)
            p = ''.join(str(i) for i in prediction)
        
            if p == 'True':
                st.success("The review entered is Legitimate.")
            if p == 'False':
                st.error("The review entered is Fraudulent.")



with st.sidebar:
    selected = option_menu('Menu', ['Add Review', 'Statistics'],
       styles={
        # "container": {"padding": "0!important", "background-color": "#fafafa"},
        # "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link-selected": { "background-color": "green"},    
    }
)


if(selected == 'Add Review'):
    st.title('Add Review')
    st.markdown(button_style, unsafe_allow_html=True)

    with st.form(key='my_form'):

        date = st.date_input(label='Enter the review Date')
        rating = st.number_input(label='Enter Review Rating(out of 5)', min_value=0, max_value=5)
        title = st.text_input(label='Enter Review Title')
        text = st.text_input(label='Enter Review text')
        verified_purchase= st.radio("Verified Purchase", ["True", "False"])
        if st.form_submit_button(label = "Check"):
            text_classification(text)

    
               
if(selected == 'Statistics'):
    st.title('Statistics')
    ModelPrep()




# Custom CSS to change the button color


# Display the custom CSS



   






    
    
   