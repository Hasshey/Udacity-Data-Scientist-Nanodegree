import sys
from sqlalchemy import create_engine
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    '''
        Load the messages and categories dataframes from the input files and merge into one dataframe

        Parameters:
                messages_filepath(string): Path for messages.csv file
                categories_filepath(string): Path for categories.csv file

        Returns:
                The merged dataframe 
    
    '''

    #Load the messages and categories dataframes
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #Merge the two dataframes into one dataframe
    df = messages.merge(categories, on = 'id')

    return df



def clean_data(df):
    '''
        Clean the merged dataframe so as to analyze the data

        Parameters:
                df: Merged dataframe of messages and categories dataframe

        Returns:
                The cleaned dataframe
    '''

    #Split the categories column for creating different columns
    categories = df['categories'].str.split(';', expand = True)

    #Select the first row of this dataframe and get the list of new columns
    row = categories.loc[0,:]
    category_colnames = row.apply(lambda x:x[:-2])
    categories.columns = category_colnames

    #Convert the category column values to 1 or 0
    for column in categories:
        categories[column] = categories[column].astype('str').apply(lambda x:x[-1])
        categories[column] = categories[column].astype('int64')

    categories['replaced'] = categories['related'].replace(2,1)

    #Replace the categories columns in the dataframe with new categories
    df = df.drop(columns = ['categories'])
    df = pd.concat([df,categories], axis = 1)
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    '''
        Save the dataframe into a SQL database

        Parameters:
                df: Dataframe to store into the database
                database_filename: Database filename to store the dataframe

        Returns:
                None
    '''

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False)

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