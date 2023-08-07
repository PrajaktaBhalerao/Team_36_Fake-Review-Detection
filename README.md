# Team_36_Fake-Review-Detection
Walmart Hackathon Project

## Abstract
In the Walmart hackathon, our project focuses on developing a robust fake reviews detection system to address the growing issue of deceptive and misleading product reviews. 

With the increasing reliance on customer feedback for purchasing decisions, the presence of fake reviews can seriously impact consumer trust and the reputation of products and brands. 

Through this project, we aspire to create a solution that can help enhance the shopping experience for Walmart customers by providing them with more authentic and reliable reviews, promoting trust and transparency in the e-commerce ecosystem.

## About Project
1. Data Collection:
   - Used Amazon-Review-Dataset from kaggle.
     
2. Data Extraction & Preprocessing:
   The initial dataset downloaded from Kaggle had 32 columns, where it had more than enough information. Upon further analysis, 
 the information within the dataset was focused more on the (1) product and (2) the reviews and not so much on the reviewer themselves, 
 and hence it was subsequently concluded that this project will take the lingusitics approach to the fake review detection problem, and not behavioral.
 At the end of the notebook, the columns were dropped, and the only columns kept were review_text and verified_purchase, 
 where they were saved inside a csv file, so we can conduct the EDA and Data Pre-processing on the textual data present.

   - Cleaned the dataset by removing special characters, stopwords , null values and removed duplicate values .
   - Done EDA.

4. Feature Extraction:
   - After performing EDA came to conclusion that only few columns are relevant to the problem statement hence kept that columns
   - Dropped the unsused column from Dataset.


5. Model Training:
   - Divide the preprocessed data into training and testing sets.
   - Implement Multinomial Naive Bayes, Support Vector Machine, and Logistic Regression algorithms separately for training on the features extracted from the training set.

6. Model Evaluation:
   - Evaluate the performance of each model using appropriate metrics like accurracy , recall , precision.


## Libraries
Pandas: For reading dataset , Data manipulation and analysis

Matplotlib : For creating various types of graphs

Seaborn: High level interface for creating attractive and informative graphs.

TextBlob: TextBlob is a Python library that provides a simple API for common natural language processing tasks, such as part-of-speech tagging, noun phrase extraction, sentiment analysis, translation, and more. 

NLTK (Natural Language Toolkit): It includes various tools, datasets, and resources for tasks like tokenization, stemming, lemmatization, parsing, and more. 

Regex (Regular Expressions): They are commonly used for searching, extracting, and manipulating text based on specific patterns. 

scikit-learn (scikit-learn): It provides a wide range of tools for classification, regression, clustering, dimensionality reduction, model selection, and preprocessing. 

## Models 
Multinominal Naive Bayes 

Support Vector Machine

Logistic Regression

## Evaluation Metrics 
Accuracy

Precision

Recall

F1_score

confusion matrix







