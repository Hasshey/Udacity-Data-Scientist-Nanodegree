# Disaster Response Pipeline Project

### Prerequisites:
The required modules that are required to run this project are outlined in the file - *requirements.txt*. Running this command will install the modules required - 

'''pip install -r requirements.txt''' 

### Introduction:
In this project, we will build a machine learning model in order to classify the disaster response messages into various categories. This complete task involves three major steps - 

#### 1. ETL pipeline:
 Here, we merge the data from the *data/messages.csv* with *data/categories.csv* and then cleanup the dataset. The cleaned up dataset is stored in the database file *data/DisasterResponse.db*

#### 2. ML pipeline:
In this step, we retrieve the data from the database file, and then use this data to train a machine learning model in order to predict the categories that each message belongs to. GridSearchCV is used to fine tune the parameters of the model and the model is saved into the pickle file *model/classifier.pkl*.

#### 3. Web app:
This is the final step where we deploy this machine learning model using a web application. In the home page, there are 2 visualisations showing some characteristics on the input data set and another option to type in custom message and know the categories.

### Files:

The input dataset is obtained from the two files 
1. *data/messages.csv* - This file contains all the messages (translated and original) and the genre of the message 
2. *data/categories.csv* - This file contains the categories that each message can fit into.

The file *data/process_data.py* merges the data from these two files and cleans it up. The cleaned up data is stored in the database file - *data/DisasterResponse.db*

The file *models/classifier.py* retrieves the data from the stored database file and trains the machine learning model. The trained machine learning model is stored in the file *models/classifier.pkl*

The file *app/run.py* contains the flask application to be run on the web. The web page design is present in the files *app/templates/master.html* and *app/templates/go.html*
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        '''python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'''
    - To run ML pipeline that trains classifier and saves
        '''python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'''

2. Run the following command in the app's directory to run your web app.
    '''python run.py'''

3. Go to http://0.0.0.0:3001/
