
import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
le = preprocessing.LabelEncoder()
from packaging import version
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import yaml
import os
import argparse
import yaml
import os

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--config', type=str)
args = vars(parser.parse_args())
path = args["config"]

def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config

config = load_config(str(path))

train_data = pd.read_csv(config["train_data_directory"])
valid_data = pd.read_csv(config["validation_data_directory"])

train_data.drop("path",axis="columns",inplace=True)
valid_data.drop("path",axis="columns",inplace=True)

train_actions_label = train_data.iloc[:,1] 
train_objects_label = train_data.iloc[:,2]
train_locations_label = train_data.iloc[:,3]
train_sentences = np.array(train_data.iloc[:,0])

from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

train_sentence_embeddings = sbert_model.encode(train_sentences)

train_actions_label = np.array(train_actions_label)
train_objects_label = np.array(train_objects_label)
train_locations_label = np.array(train_locations_label)

train_actions_label = np.reshape(train_actions_label,(len(train_actions_label),1))
train_objects_label = np.reshape(train_objects_label,(len(train_objects_label),1))
train_locations_label = np.reshape(train_locations_label,(len(train_locations_label),1))

ohe = OneHotEncoder(sparse=False)
train_actions_label = ohe.fit_transform(train_actions_label)
train_locations_label = ohe.fit_transform(train_locations_label)
train_objects_label = ohe.fit_transform(train_objects_label)


strategy = tf.distribute.MirroredStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

with strategy.scope():
 model_action = Sequential()
 model_action.add(Dense(12, input_shape=(768,), activation='relu'))
 model_action.add(Dense(8, activation='relu'))
 model_action.add(Dense(len(train_actions_label[0]), activation='sigmoid'))

 model_action.compile(loss= config["loss"], optimizer=config["optimizer"], metrics=config["metric"])

 model_object = Sequential()
 model_object.add(Dense(12, input_shape=(768,), activation='relu'))
 model_object.add(Dense(8, activation='relu'))
 model_object.add(Dense(len(train_objects_label[0]), activation='sigmoid'))

 model_object.compile(loss= config["loss"], optimizer=config["optimizer"], metrics=config["metric"])

 model_location = Sequential()
 model_location.add(Dense(12, input_shape=(768,), activation='relu'))
 model_location.add(Dense(8, activation='relu'))
 model_location.add(Dense(len(train_locations_label[0]), activation='sigmoid'))

 model_location.compile(loss= config["loss"], optimizer=config["optimizer"], metrics=config["metric"])

val_actions_label = valid_data.iloc[:,1] 
val_objects_label = valid_data.iloc[:,2]
val_locations_label = valid_data.iloc[:,3]
val_sentences = np.array(valid_data.iloc[:,0])

val_sentence_embeddings = sbert_model.encode(val_sentences)

val_actions_label = np.array(val_actions_label)
val_objects_label = np.array(val_objects_label)
val_locations_label = np.array(val_locations_label)

val_actions_label = np.reshape(val_actions_label,(len(val_actions_label),1))
val_objects_label = np.reshape(val_objects_label,(len(val_objects_label),1))
val_locations_label = np.reshape(val_locations_label,(len(val_locations_label),1))

ohe = OneHotEncoder(sparse=False)
val_actions_label =ohe.fit_transform(val_actions_label)
val_locations_label = ohe.fit_transform(val_locations_label)
val_objects_label = ohe.fit_transform(val_objects_label)

print("Training Model to find action in sentence\n")
model_action.fit(train_sentence_embeddings, train_actions_label, epochs=config["epochs"], verbose=0,batch_size=config["batch_size"],validation_data=(val_sentence_embeddings, val_actions_label))
model_action.save(config["action_model_path"])


print("\nTraining Model to find object in sentence\n")
model_object.fit(train_sentence_embeddings, train_objects_label, epochs=config["epochs"], verbose=0,batch_size=config["batch_size"],validation_data=(val_sentence_embeddings, val_objects_label))
model_object.save(config["object_model_path"])


print("\nTraining Model to find location in sentence\n")
model_location.fit(train_sentence_embeddings, train_locations_label, epochs=config["epochs"], verbose=0,batch_size=config["batch_size"],validation_data=(val_sentence_embeddings, val_locations_label))
model_location.save(config["location_model_path"])
