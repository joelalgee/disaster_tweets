import nltk
import numpy as np
import pandas as pd
import re
import sys
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from joblib import dump, load
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sqlalchemy import create_engine

nltk.download('punkt')
nltk.download('wordnet')

def load_data(database_filepath):
    """ Load database containing messages and labels, extract any labels containing a
        single class, split the data into features and labels, plus category names for
        the labels.

        Single class labels are extracted because they will break the gradient boosting
        classifier used for modeling.

    Args:
    database_filepath: str. The filepath of the saved database.

    Returns:
    X: ndarray. The array of features to be split into train and test sets.
    Y: ndarray. The 2d array of labels to be split into train and test sets.
    category_names: list. The names of the labels.
    """

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('categorised_tweets', engine)
    df, single_class_labels = extract_single_class_labels(df)
    X = df['message'].values
    Y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns.tolist()

    for key, value in single_class_labels.items():
        print('----')       
        print(f'All observations of `{key}` are classed as `{value}`.')
        print(f'`{key}` is therefore be excluded from the data for modeling.')

    return X, Y, category_names

def extract_single_class_labels(df):
    """Find any labels containing a single class (either 0 or 1)
       across all observations, record them and drop from df.
    
    Args:
    df: Pandas dataframe. The dataframe to search

    Returns:
    df: Pandas dataframe. The dataframe with any single class labels dropped.
    single_class_labels: dict. A dictionary containing the single class labels - 
                               key: label name (str), value: class (0 or 1).
    """
    
    single_class_labels = {}
    rows = len(df)

    # Record label name and class in dict if class is either 0 or 1 for all observations
    for colname, coldata in df.iloc[:, 4:].iteritems():
        colsum = coldata.sum()
        if colsum == 0 or colsum == rows:
            single_class_labels[colname] = coldata[0]

    # Drop any dataframe columns containing single class labels
    df = df.drop(single_class_labels.keys(), axis=1)

    return df, single_class_labels

def multilabel_test_train_split(X, Y, category_names=None, test_size=0.2, random_state=None, verbose=False):
    """Splits X and Y into train and test sets, ensuring that each label in Y is represented
       roughly equally in both sets.
    
    Args:
    X: ndarray. The array of features to be split into train and test sets.
    Y: ndarray. The 2d array of labels to be split into train and test sets.
    category_names: list. The names of the labels, for use in verbose reporting. If not 
                          provided, each label will be allocated a number instead.
    test_size: float. The proportion of observations to allocate to the test set.
    random_state: int. The random seed to be used for debugging purposes.
    verbose: boolean. Whether to output a detailed breakdown of splits for each label,
                      or a brief overall summary
    Returns:
    X_train: ndarray. The array of features for the train set.
    X_test: ndarray. The array of features for the test set.
    Y_train: ndarray. The array of labels for the train set.
    Y_test: ndarray. The array of labels for the test set.
    """
    
    # Set random seed
    if random_state is None:
        np.random.seed()
    else:
        np.random.seed(random_state)
    
    # Create indicator for each observation initialised to zeros:
    # 1 - in test set
    # 0 - not in test set, but available
    # nan - not in test set, not available (because at least one other label for this observation is 
    #       already fully represented in the test set)
    rows = len(X)
    test = np.zeros(rows)

    # Create an indexed list of the total number of occurrences of each label, sorted by rarity (rarest first)
    idx = list(range(Y.shape[1]))
    totals = Y.sum(axis=0)
    label_totals = list(zip(idx, totals))
    label_totals.sort(key=lambda x: x[1])

    if category_names is None:
        category_names = [str(i) for i in idx]

    print('----')
    print('Target test set size: ' + str(test_size))
    print('Splitting into train and test sets...')

    props = []

    # Starting with rarest label and working towards commonest, add random sample to test set
    for idx, total in label_totals:

        # Check how many observations in the test set already have this label
        n_sampled = len(np.where((Y[:,idx] == 1) * (test == 1))[0])

        # Check how many observations are available to be added to the test set
        available_obs = np.where((Y[:,idx] == 1) * (test == 0))[0]
        n_available = len(available_obs)

        # Calculate how many more observations with this label are needed
        n = max(round(total * test_size) - n_sampled, 0)

        if verbose == True:
            print('----')
            print('Label: ' + category_names[idx])
            print('Total occurrences: ' + str(total))
            print('Occurrences already in test set (due to other labels): ' + str(n_sampled))
            print('Occurrences not in test set, but available: ' + str(n_available))
            print('Occurrences to be added to test set: ' + str(n))
            print('Sampling...')

        # If there are enough available occurrences, add a random sample to the test set, mark the rest as unavailable
        if n_available > n:
            sample = np.random.choice(available_obs, size=n, replace=False)
            test[sample] = 1
            test[np.where((Y[:,idx] == 1) * (test == 0))[0]] = np.nan

        # Otherwise add whatever is available
        else:
            test[available_obs] = 1

        # Record label's final test set representation
        n_sampled = len(np.where((Y[:,idx] == 1) * (test == 1))[0])
        final_prop = n_sampled / total
        props.append(final_prop)

        if verbose == True:
            print(str(min(n_available, n)) + ' occurences added to test set.')
            print('Final proportion of occurrences in test set: ' + str(final_prop))
            print('Overall test set size: ' + str(round(np.nansum(test))))

    # Check how many observations remain available (and thus have no labels)
    remaining_obs = np.where(test == 0)[0]
    n_remaining = len(remaining_obs)

    # Calculate how many observations with no labels are needed in the test set
    n = round(n_remaining * test_size)

    if verbose == True:
        print('----')
        print('No labels:')
        print('Number of occurrences: ' + str(n_remaining))
        print('Number of occurrences to add to test set: ' + str(n))
        print('Sampling... ')

    # Add a random sample to the test set if needed
    if n > 0:
        sample = np.random.choice(remaining_obs, size=n, replace=False)
        test[sample] = 1

    final_test_size = round(np.nansum(test))

    if verbose == True:
        print(str(min(n_remaining, n)) + ' occurences added to test set.')
        print('Final overall test set size: ' + str(final_test_size))

    print('----')
    print('Proportion of observations in test set : ' + str(final_test_size / rows))
    print('Greatest label proportion in test set: ' + str(max(props)))
    print('Least label proportion in test set: ' + str(min(props)))

    # Prepare sets for return
    X_train = X[np.where(test != 1)[0]]
    X_test = X[np.where(test == 1)[0]]
    Y_train = Y[np.where(test != 1)[0]]
    Y_test = Y[np.where(test == 1)[0]]

    return X_train, X_test, Y_train, Y_test

def tokenize(text):
    """Replace urls with placeholders, split text into lemmatized tokens. Stopwords are
       not removed.

    Args:
    text: string. The text to be tokenized.
    
    Returns:
    lemmed: list of strings. The lemmatized tokens.
    """

    # Find any urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    
    # Replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
    
    # Normalize to lower case alphanumeric characters
    text = text.lower()
    text = re.sub("'", '', text)
    text = re.sub(r'[^a-z0-9]', ' ', text)
    
    # Tokenize
    words = word_tokenize(text)
        
    # Lemmatize nouns
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    # Lemmatize verbs
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    
    return lemmed

def build_model():
    """Build a multilabel gradient boosting classifier using SMOTE to compensate for
    the imbalanced nature of the data. Tune key hyperparameters using grid search.

    Args:
    None

    Returns:
    model: GridSearchCV. The configured model.
    """

    # Configure pipeline components
    vect = CountVectorizer(lowercase=False, tokenizer=tokenize, max_df=0.95,
                           min_df=25)
    tfidf = TfidfTransformer()
    smote = SMOTE()
    grad = GradientBoostingClassifier(max_features=None,
                                      n_iter_no_change=3)
    
    # Both smote and grad need to be run for each label individually to be effective
    clf = make_pipeline(smote, grad)
    multi_clf = MultiOutputClassifier(clf, n_jobs=-1)

    # Assemble full pipeline
    pipeline = make_pipeline(vect, tfidf, multi_clf)

    # Select hyperparameters to tune using grid search
    #parameters = dict(countvectorizer__ngram_range=[(1,1), (1,2)],
    #                  multioutputclassifier__estimator__gradientboostingclassifier__n_estimators=[10, 100],
    #                  multioutputclassifier__estimator__gradientboostingclassifier__max_depth=[2, 8],
    #                  multioutputclassifier__estimator__gradientboostingclassifier__subsample=[0.1, 0.5])

    parameters = dict(countvectorizer__ngram_range=[(1,1)],
                      multioutputclassifier__estimator__gradientboostingclassifier__n_estimators=[10],
                      multioutputclassifier__estimator__gradientboostingclassifier__max_depth=[2],
                      multioutputclassifier__estimator__gradientboostingclassifier__subsample=[0.1])

    model = GridSearchCV(pipeline, parameters, scoring='f1_micro')

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """Make predictions on test set and print out key metrics.

    Args:
    model: GridSearchCV. The configured model.
    X_test: ndarray. The array of features for the test set.
    Y_test: ndarray. The array of labels for the test set.
    category_names: list. The names of the labels.

    Returns:
    None
    """

    # Get predictions
    Y_pred = model.best_estimator_.predict(X_test)

    # Print key metrics for each label
    metrics = ['{:23}'.format('Label') + 
                '{:>9}'.format('Precision') + 
                '{:>9}'.format('Recall') + 
                '{:>9}'.format('F1 Score') + 
                '{:>9}'.format('Support')]
    for i in range(0, len(category_names)):
        report = classification_report(Y_test.T[i], Y_pred.T[i], output_dict=True)
        metrics.append('{:23}'.format(category_names[i]) + 
                        '{:>9.5f}'.format(report['1']['precision']) + 
                        '{:>9.5f}'.format(report['1']['recall']) + 
                        '{:>9.5f}'.format(report['1']['f1-score']) + 
                        '{:>9}'.format(int(report['1']['support'])))
    for line in metrics:
        print(line)

def save_model(model, model_filepath):
    """Save fitted model to disk.

    Args:
    model: GridSearchCV. The configured model.
    model_filepath: str. The filepath for the saved model.

    Returns:
    None
    """

    dump(model, model_filepath, compress=3)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('----')
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = multilabel_test_train_split(X, Y)

        print('----')
        print('Building model...')
        model = build_model()
        
        print('----')
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('----')
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('----')
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()