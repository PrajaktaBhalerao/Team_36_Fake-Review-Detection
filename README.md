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

Following steps were performed for doing above operations:
 1.Shape of the dataset is studied, and the columns are seen in details. 
 2.Data description and Summary Statistics: analyzed statistic of each and evry attribute for EDA process.
 3.Normalization of data to support EDA analysis  -
 4.Checking for NULL and Duplicates: There are NULL values within this dataset which needs to be taken care of.
    review_title has 98 NULL values, which is the minority considering there are over 2k records.
    URL has 800+ missing values, which can be ignored since this is not significant to the nature of our project
    matched_keywords, time_of_publication, manufacturers_response, dimension4, dimension5, dimension6 don't have any values in them, and hence will most 
    definately be dropped.
    dimension 7 has 2 missing values, and from above it can be seen that it contains just an extra info on the product.
    ## There are no NULL values for review_rating, review_text, and verified_purchase, which are the main attributes needed for the analysis.
Non-texual attributes are giving the extra information about the product.
Texual attributes like review_date,review_title,review_text,review_rating,verified_purchase used for understanding in EDA process.

3.Explonatory Data Analysis(EDA):
  -plot for helpful_review_count:  
   which can aid us in understanding the reviews those helped users in purchases a product
   From that, we conclude that,out of 2000 reviews 0 review was helpful,out of 150, 1 review was helpful and out of 10 reviews 6 reviews were helpful.
   In this case, this will actually skew our understanding in identifying which reviews are fake and which ones are real, and therefore to eliminate bias,this 
   column will not be considered for our model building.
 -Pie-chart On Verified Purchase:
  Verified_purchases column is the target variable for this project. From Pie-chart we conclude that,there are near equal parts of true VP and 
  false VP (56% and 44% respectively).
Dropping of columns which are not necessary for model building
Saving final dataframe to csv.

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







