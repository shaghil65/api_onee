from io import StringIO
import numpy as np
#Importing Pandas Library 
import pandas as pd
import requests
import os
from flask import Flask
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
#Importing MultiClass Classifier ExtraTreeClassifier From SkLearn
from sklearn.tree import ExtraTreeClassifier

#Importing VarianceThreshold  From Features Selection


app = Flask(__name__)
@app.route('/getdata/<int:temp>/<int:fire>/<int:humidity>/<int:gas>')
def getdata(temp,fire,humidity,gas):
    URL = "https://drive.google.com/file/d/1VGB35TvJl2LMr4H2zdVObhPpT7mfJwrV/view?usp=sharing"
    path = 'https://drive.google.com/uc?export=download&id='+URL.split('/')[-2]
    train = pd.read_csv(path)



    data = {'Temperature':[temp],
        'Gas':[gas],
        'Fire':[fire],
        'Humidity':[humidity]
        }

    test = pd.DataFrame(data,columns=['Temperature','Gas','Fire','Humidity'])

#Firstly Training 3 Models For Train Data(Without Normalization)
#Spliting Our Train Data(Without Normalization) Into 80% And 20%

#For Labels
    Labels = train.Class

#For Features
    Features = train.drop('Class', axis=1)

#Splitting The Data Into 80%(For Training t_train) And 20%(For Testing t_test)
    t_train, t_test, y_train, y_test = train_test_split(Features, Labels,test_size=0.2)

#Here We Have Tweaked Our classifiers By  Adjusting The Parameters
    extra_tree = ExtraTreeClassifier(random_state=0,criterion ='entropy')
    cls = BaggingClassifier(extra_tree, random_state=0).fit(t_train, y_train)
    Class = cls.predict(t_test)
#Printing The Predicted Value Of Local Splitted Test Data
    print("The Predicted Values  Class Of Local Test Data",Class)
#Checking The  accuracy This Model Using The Local Test Data 
    ETree=cls.score(t_test,y_test)
#Printing The Accuracy Score Of The Decision Tree Classifier 
    print("The Accuracy Score Of Extra Tree Classifier On Local Test Data",ETree*100)

    Class2 = cls.predict(test)
    print("The Predicted Values  Class Of Local Test Data",Class2)
    py_list = Class2.tolist()
    c = py_list[0]


  
    return {
        "Temperature":temp, 
        "Gas":gas,
        "Class":c, 
        "Humidity": humidity,
        "Fire": fire,
        }

        
if __name__ == '__main__': 
        app.run(host="127.0.0.1", port=8000, debug=True)

# if __name__ == '__main__':
#    port = int(os.environ.get("PORT", 8000))
#    app.