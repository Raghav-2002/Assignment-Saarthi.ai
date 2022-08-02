import sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.metrics import f1_score
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
import tensorflow
from tensorflow import keras

path = sys.argv[1]
test_data = pd.read_csv(str(path))

test_data.drop("path",axis="columns",inplace=True)

test_actions_label = test_data.iloc[:,1] 
test_objects_label = test_data.iloc[:,2]
test_locations_label = test_data.iloc[:,3]
test_sentences = np.array(test_data.iloc[:,0])

print("Converting Sentences to Vectors")
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
test_sentence_embeddings = sbert_model.encode(test_sentences)

test_actions_label = np.array(test_actions_label)
test_objects_label = np.array(test_objects_label)
test_locations_label = np.array(test_locations_label)

test_actions_label = np.reshape(test_actions_label,(len(test_actions_label),1))
test_objects_label = np.reshape(test_objects_label,(len(test_objects_label),1))
test_locations_label = np.reshape(test_locations_label,(len(test_locations_label),1))

true_actions_label =le.fit_transform(test_actions_label)
true_locations_label = le.fit_transform(test_locations_label)
true_objects_label = le.fit_transform(test_objects_label)

ohe = OneHotEncoder(sparse=False)
test_actions_label =ohe.fit_transform(test_actions_label)
test_locations_label = ohe.fit_transform(test_locations_label)
test_objects_label = ohe.fit_transform(test_objects_label)

action_model = keras.models.load_model("Action")
location_model = keras.models.load_model("Location")
object_model = keras.models.load_model("Object")

predict_action=[]
predict_object=[]
predict_location=[]

print(len(test_sentences))
for i in range(len(test_sentences)):
  print(i)
  X=np.reshape(test_sentence_embeddings[i],(1,len(test_sentence_embeddings[i])))
  predict_action.append(np.argmax(action_model.predict(X)))
  predict_location.append(np.argmax(location_model.predict(X)))
  predict_object.append(np.argmax(object_model.predict(X)))

print("F1 Score for action classification",f1_score(true_actions_label,predict_action,average=None))
print("F1 Score for location classification",f1_score(true_locations_label,predict_location,average=None))
print("F1 Score for object classification",f1_score(true_objects_label,predict_object,average=None))