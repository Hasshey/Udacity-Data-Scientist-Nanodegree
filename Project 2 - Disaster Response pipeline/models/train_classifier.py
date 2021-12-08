import sys
import nltk
import pickle
nltk.download(['punkt', 'wordnet'])

import pandas as pd
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    '''
        Load the cleaned input data from the database and obtain the features/target for the model

        Parameters:
                database_filepath: The filepath in the database to retrieve the dataframe

        Returns:
                X: Features of the model
                Y: Target of the model
                category_names: Names of different categories
         
    '''

    #Create the SQL enginer connection and retrieve the cleaned up dataframe
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages', con=engine)

    #Create the Features and Target for the model
    X = df['message']
    Y = df.drop(columns = ['id', 'message', 'original', 'genre'])

    #Get the names of different categories
    category_names = Y.columns

    return X, Y, category_names



def tokenize(text):
    '''

        Tokenizer function for the input messages to be used in the model

        Parameters:
                text: Raw input text from the input dataframe

        Returns:
                clean_tokens: A list of clean tokens

    '''

    # Tokenize the input text 
    tokens = word_tokenize(text)

    # Create a lemmatizer for the tokens created
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    # Iterate through the tokens
    for token in tokens:

        # Lemmatize the tokens, normalize the case and remove leading/trailing spaces
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens


def build_model():
    '''

        Builds Classifier and tunes the model using GridSearchCV
        
        Parameters:
                None
        Returns:
                cv: Classifier
    
    '''

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize, analyzer = 'char')),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'clf__estimator__n_estimators' : [50, 100]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    
        Evaluates the model's performance and display classification report

        Parameters: 
            model: Classifier
            X_test: Test data for X
            Y_test: Test data for Y
            category_names: names for different categories

        Returns:
            None

    '''

    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))

def save_model(model, model_filepath):
    '''

        Export the model as a pickle file

        Parameters:
                model: Classifier
                model_filepath: Path of the pickle file

        Returns:
                None
    '''

    pickle.dump(model, open(model_filepath), 'wb')


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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