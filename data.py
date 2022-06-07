import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.models import Model

df = pd.read_csv("BadGuyGoodGuy/character.csv")
#print(df.head())
datasets = df.groupby("data set")
train_df = datasets.get_group("train")
test_df = datasets.get_group("test")
valid_df = datasets.get_group("valid")

train_df["filepaths"] = train_df["filepaths"].apply(lambda x: os.path.join("BadGuyGoodGuy", x))
test_df["filepaths"] = test_df["filepaths"].apply(lambda x: os.path.join("BadGuyGoodGuy", x))
valid_df["filepaths"] = valid_df["filepaths"].apply(lambda x: os.path.join("BadGuyGoodGuy", x))
# tao data
img_size = (300,300)
batch_size = 16
train_generator = ImageDataGenerator(horizontal_flip = True)
valid_test_generator = ImageDataGenerator()
train_data = train_generator.flow_from_dataframe(train_df,
						 x_col = "filepaths",
						 y_col = "labels",
						 target_size = img_size,
						 class_mode = "categorical",
						 color_mode = "rgb",
						 shuffle = True,
						 batch_size = batch_size)
						 
						
valid_data = valid_test_generator.flow_from_dataframe(valid_df,
						 x_col = "filepaths",
						 y_col = "labels",
						 target_size = img_size,
						 class_mode = "categorical",
						 color_mode = "rgb",
						 batch_size = batch_size)
						 
test_data = valid_test_generator.flow_from_dataframe(test_df,
						 x_col = "filepaths",
						 y_col = "labels",
						 target_size = img_size,
						 class_mode = "categorical",
						 color_mode = "rgb",
						 batch_size = batch_size)
						 
# build model
base_model = tf.keras.applications.efficientnet.EfficientNetB3(include_top = False,
								weights ="imagenet",
								input_shape = (img_size[0], img_size[1], 3),
								pooling = "max")
base_model.trainable = True
x = base_model.output
x = BatchNormalization() (x)
x = Dense(1024, activation = "relu") (x)
x = Dropout(0.3) (x)
x = Dense(512, activation = "relu") (x)
x = Dropout(0.3) (x)
x = Dense(128, activation = "relu") (x)
x = Dropout(0.3) (x)
outputs = Dense(2, activation = "softmax") (x)

model = Model(inputs =base_model.input, outputs = outputs)
lr = 0.001

model.compile(optimizer = Adamax(learning_rate = lr), loss = "categorical_crossentropy", metrics = ["accuracy"])


from tensorflow.keras.callbacks import ModelCheckpoint
ckpoint = ModelCheckpoint("best_weights_model.h5", monitor = "val_accuracy", save_best_only = True, mode = "auto")

n_epochs =10
model.fit(x = train_data,
	  epochs = n_epochs,
	  validation_data = valid_data,
	  callbacks = [ckpoint])
model = tf.keras.models.load_model("best_weights_model.h5")
