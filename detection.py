# -*- coding: utf-8 -*-
"""
Created on Thu Mar 5 08:54:32 2020

@author: SATWIK RAM K
"""

# Importing the Librarries
import numpy as np
import pandas as pd

# Importing the tensorflow and keras
import tensorflow as tf
from tensorflow import keras


    
# Creating callback or checkpoint during training
class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if(logs.get('acc') >= 0.998):
                print("\n Reached 99.8% accuracy so cancelling training!")
                self.model.stop_training = True
    
                
callbacks = mycallback

 # Building the neural network
model = tf.keras.models.Sequential([
        
        # This is First Convolution Layer
        tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation = 'relu', input_shape = (150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.2),
                                 
        # Adding Second Convoluting Layer
        tf.keras.layers.Conv2D(64, (3,3), padding = 'same', activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.2),
        
        # Adding Third Convolution Layer
        tf.keras.layers.Conv2D(128, (3,3), padding = 'same', activation = 'relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.2),
        
        # Flatening the images
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.2),
        
        # Adding the Dense Layer of 512 hidden neurons
        tf.keras.layers.Dense(512, activation = 'relu'),
        tf.keras.layers.Dropout(0.2),
        
        # Adding last output layer of 5 classes
        tf.keras.layers.Dense(5, activation = 'softmax')])
    
# Compiling the model
from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer = RMSprop(lr = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    
# Fitting Images
from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip= True,
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        fill_mode = 'nearest')
    
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'my_data/training_set',
        target_size = (150, 150), # Setting target size of 64, 64 pixels
        batch_size = 50,
        class_mode = 'categorical') # As multiple class output
    
test_set = test_datagen.flow_from_directory(
        'my_data/test_set',
        target_size=(150, 150), # Setting target size of 64, 64 pixels
        batch_size = 32,
        class_mode='categorical') # As multiple class output

model.fit_generator(
        training_set,
        steps_per_epoch = 150,
        epochs = 5,
        validation_data = test_set,
        validation_steps = 45)
    
training_set.class_indices
    



#Making New Predictions
import numpy as np
from tensorflow.keras.preprocessing import image
test_image = image.load_img('my_data/single_predictions/IMG_20191011_130641.jpg',  target_size=(150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)

if result[0][0] == 1:
    prediction = 'pari'
    print('I think its parikshith')
    
elif result[0][1] == 1:
    prediction = 'pavan'
    print('I think its pavan')
    
elif result[0][2] == 1:
    prediction = 'Praveen'
    print('I think its praveen')
    
elif result[0][3] == 1:
    prediction = 'Satwik'
    print('I think its Satwik')
    
elif result[0][4] == 1:
    prediction = 'Sinch'
    print('I think its Sinchana')
    

    


  