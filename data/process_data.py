# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load tweet messages and ground truth categories and merge on `id`.

    Args:
    messages_filepath: str. The filepath of the messages csv.
    categories_filepath: str. The filepath of the categories csv.

    Returns:
    df: Pandas dataframe. The merged dataframe.
    """

    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # return merged datasets
    df = messages.merge(categories, on='id')

    return df

def clean_data(df):
    """Convert categories column to one column per category (0/1) and
       remove duplicate rows.

    Args:
    df: Pandas dataframe. The dataframe to be cleaned.

    Returns:
    df: Pandas dataframe. The cleaned dataframe.
    """

    # create a dataframe of the individual category columns
    categories = df.categories.str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories
    category_colnames = row.str.split('-').str[0].tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str[1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # ensure maximum value for each category is 1
    categories[categories > 1] = 1

    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates and return
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filepath):
    """Save pandas dataframe as a sqlite database.

    Args:
    df: Pandas dataframe. The dataframe to be saved.
    database_filepath: str. The filepath to use for the saved database.
    """

    engine = create_engine(f'sqlite:///{database_filepath}')
    engine.execute('DROP TABLE IF EXISTS categorised_tweets;')
    df.to_sql('categorised_tweets', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()