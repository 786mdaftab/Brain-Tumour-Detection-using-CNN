# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/tr_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/te_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 230,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 23)

# Part 3 - Making new predictions

import numpy as np
from keras.preprocessing import image
address = 'dataset/single_prediction/no/Y12.jpg'
test_image = image.load_img(address, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'No Tumour Detected'
else:
    prediction = 'Brain Tumour Detected'
    
  #########################################################################################
#image plotting
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  

img= Image.open(address)
plt.imshow(img)


############################################################################################
#PLOTTING CURVE FOR LOSSES AND ACCURACY OF SYSTEM
# -*- coding: utf-8 -*-
losses =[0.4299,
0.2220,
0.0983,
0.0511,
0.0310,
0.0103,
0.0140,
0.0058,
0.0125,
0.0055,
9.1392e-04,
5.3858e-04,
0.0017,
0.0259,
0.0099,
0.0065,
7.4795e-04,
2.4068e-04,
1.5254e-04,
2.1131e-04,
6.8710e-05,
5.1168e-05,
5.8422e-05,
2.3587e-05,
3.3743e-05]
accuracy = [0.8111,0.9106,
0.9649,
0.9846,
0.9899,
0.9978,
0.9965,
0.9984,
0.9959,
0.9985,
1.0000,
1.0000,
0.9993,
0.9911,
0.9971,
0.9984,
1.0000,
1.0000,
1.0000,
1.0000,
1.0000,
1.0000,
1.0000,
1.0000,
1.0000]


import matplotlib.pyplot as plt
plt.plot(losses, color= 'red', label = "LOSSES")
plt.plot(accuracy, color = 'blue', label = "ACCURACY")
plt.xlabel('no. of epochs')
plt.legend()
plt.show()