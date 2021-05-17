# Disaster Response Pipeline Project

## Summary

This project analyzes a dataset containing real messages that were sent during disaster events. A machine learning pipeline was created to categorize these events so that the messages can be sent to an appropriate disaster relief agency.

The project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the data.

## Package versions

python                    3.9.4
numpy                     1.20.1
pandas                    1.2.4
imbalanced-learn          0.8.0
joblib                    1.0.1
scikit-learn              0.24.2
sqlalchemy                1.4.7
nltk                      3.6.1
plotly                    4.14.3
flask                     1.1.2

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://localhost:3000

## Files

### 1. data/process_data.py:

* Extracts the messages and categories from the relevant csv files and combines them into a single pandas dataframe.
* Converts genre and categories into one hot encoded columns.
* Removes duplicates.
* Saves the dataframe as a sqlite database.

### 2. models/train_classifier.py:

* Loads the database and separates features and labels into separate numpy arrays.
* Splits the arrays into train and test sets, ensuring all labels are represented adequately in both.
* Builds and fits a model using GridSearchCV to optimise a multi-output pipeline including a word count vectorizer, tfdif transformer, smote sampler and gradient boosting classifier, using the train set.
* Evaluates the model using the test set, outputting precision, recall, f1 score and support for positive classification of each label.
* Saves the model as a pkl file.

### 3. models/custom_grid_search_cv.py

Slightly modifies scikit learn's GridSearchCV class:

* Extracts and store any labels and their class for which observations are all 0 or all 1 before fitting. This is required as the gradient boosting classifier requires at least 2 classes in its training data.
* Restores these labels and their classes in the appropriate position within the results, after predicting classes for other labels.

This class is stored in its own module to allow joblib to access it upon loading the saved model.

### 4. models/custom_tokenize.py

* Tokenizes messages (removes URLs, normalizes text, tokenizes to words, lemmatizes).

This function is stored in its own module to allow joblib to access it upon loading the saved model.

### 5. app/run.py

* Creates the web app.
* Loads the saved database and model.
* Displays some visualisations on the data.
* Allows new tweets to be categorized based on message and genre.

### 6. app/templates/master.html

* Controls the layout of the main web page and form.

### 7. go.html

* Controls the layout of the categorization results.

## Noteworthy features

### Imbalanced data

The dataset is highly imbalanced with some labels used infrequently. To cope with this property, several measures have been taken:

* A custom multilabel_test_train_split function ensures that all labels are represented adequately in train and test sets. Were scikit learn's standard test_train_split function to be used, some labels would not be represented in the training set at all, causing the gradient boosting classifier to return an error, and other algorithms to deduce the label is never used.

* imblearn's SMOTE sampler oversamples positive classifications for each label just before the gradient boosting classifier begins analysis, to provide an artificially balanced set of data.

### Genre as a feature

Genre is included as a feature for this model. To enable this, the natural language transformations are applied as a column transformer.

### F1 Micro Scoring

For the model to be useful, it needs to identify a high proportion of relevant messages for a given label (high recall), while simultaneously reducing the number of messages that need to checked through manually (high precision).

High recall with low precision would mean sending a large number of irrelevant messages to an agency, which doesn't reduce the manual workload by much. High precision with low recall would mean missing most of the relevant messages, so the rest of the messages would need to be sent over and checked manually anyway.

For this reason, F1 micro score is used for scoring by the grid search.