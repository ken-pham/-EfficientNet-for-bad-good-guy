from data import *
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

class_indices = list(train_data.calss_indices.values())
class_name = list(train_data.class_indices.keys())

preds = model.predict(test_dat)
labels = test_data.labels 

errors =0
for i,p in enumerate(preds):
	index = np.argmax(p)
	if class_indices[index] != labels[i]:
		errors += 1
acc = (1.0 - errors/len(preds)) * 100
print("Errors = ",errors," ACC = ",acc)
