import tensorflow as tf
import os
from custom_callbacks import CustomCallback
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Model:
    #PARENT DIRECTORY
    PARENT_DIRECTORY = os.getcwd()
    DATASET_PATH = os.path.join(PARENT_DIRECTORY,'cnn-lstm' , 'split-dataset')

    #TRAIN - TEST - DEV [DATASET_PATH]
    TRAIN_DATASET = os.path.join(DATASET_PATH ,'train')
    TEST_DATASET  = os.path.join(DATASET_PATH ,'test')
    DEV_DATASET   = os.path.join(DATASET_PATH ,'dev')
    
    #Create Labeling Data
    CLASS_LABELS = sorted(os.listdir(TRAIN_DATASET))
    LABELS_COUNT = len(CLASS_LABELS)
    
    def __init__(self):
        #EPOCH, BATCH_SIZE AND TARGET_SIZE VARIABLE 
        EPOCH = 50
        BATCH_SIZE = 32
        TARGET_SIZE = (250,200)

        #STEP VARIABLE
        STEPS_PER_EPOCH = 64
        VALIDATION_STEPS = 7
        
        #TARGET ACCURACY
        TARGET_ACCURACY = 0.95
        
        #IMAGE GENERATOR
        TRAIN_GEN_IMAGE , VALIDATION_GEN_IMAGE = self.image_generator(TARGET_SIZE , BATCH_SIZE)
        
        CNN_MODEL = self.cnn_model(num_label=self.LABELS_COUNT, input_shape=(250,150,3))
        CNN_MODEL.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        TRAINABLE_MODEL = CNN_MODEL.fit(
        TRAIN_GEN_IMAGE,
        steps_per_epoch = STEPS_PER_EPOCH,
        epochs = EPOCH,
        validation_data = VALIDATION_GEN_IMAGE,
        validation_steps = VALIDATION_STEPS,
        callbacks=[CustomCallback(TARGET_ACCURACY)]
        )

    def image_generator(self, target_size , batch_size):
        datagen_image = ImageDataGenerator(
                        rescale=1./255,
                        zoom_range=0.0,
                        horizontal_flip=True,
                        validation_split=0.2)

        train_gen_image = datagen_image.flow_from_directory(
                        self.TRAIN_DATASET, 
                        target_size=target_size,
                        shuffle=True,
                        batch_size=batch_size,
                        color_mode='rgb',
                        class_mode='categorical',
                        subset='training',)

        val_gen_image = datagen_image.flow_from_directory(
                        self.TEST_DATASET, 
                        target_size=target_size,
                        batch_size=batch_size,
                        color_mode='rgb',
                        class_mode='categorical',
                        subset='validation')
        
        return train_gen_image , val_gen_image
    
    def cnn_model(self,num_labels,input_shape):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.5))
        
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.5))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        # Dense atau Full Connection
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(units=num_labels, activation='softmax'))

        return model

Model()