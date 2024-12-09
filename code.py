import os

# Changer le répertoire de travail pour le dossier contenant le script
os.chdir(os.path.dirname(os.path.realpath(__file__)))

############################################################################################################
# Cellule 1

# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BS=16
datagenerator = ImageDataGenerator(rescale=1./255)

train_generator = datagenerator.flow_from_directory('./Flowers/Train',
                                          target_size=(128, 128),
                                          class_mode='categorical',
                                          batch_size=BS,
                                          shuffle='true',
                                          color_mode='rgb')

test_generator = datagenerator.flow_from_directory('./Flowers/Test',
                                          target_size=(128, 128),
                                          class_mode='categorical',
                                          color_mode='rgb'
                                          )


############################################################################################################
# Cellule 2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import SGD

model = Sequential()
model.add(Conv2D(4,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(len(train_generator.class_indices), activation="softmax"))
opt = SGD(learning_rate=1.0)
EPOCHS=10
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(train_generator, validation_data = test_generator, epochs=EPOCHS)
