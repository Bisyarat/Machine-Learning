from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.preprocessing import image
import keras.utils as image
import os
import numpy as np

class GenerateAugDataset:
    DATASET_PATH = 'new-dataset'
    SAVE_PATH = 'augmentation-dataset'
    
    def __init__(self):
        datagen = self.create_datagen()
        
        if os.path.exists(self.DATASET_PATH):

            if not os.path.exists(self.SAVE_PATH):
                os.mkdir(self.SAVE_PATH)

            for class_folder in os.listdir(self.DATASET_PATH):
                folders = os.path.join(self.DATASET_PATH , class_folder)
                for folder in os.listdir(folders):
                    images_files_path = os.path.join(folders, folder)
                    class_folder_name = os.path.join(self.SAVE_PATH, class_folder)

                    if not os.path.join(class_folder_name):
                        os.mkdir(class_folder_name)

                    list_images = []
                    
                    for image_file in os.listdir(images_files_path):
                        image_path = os.path.join(images_files_path , image_file)
                        img = image.load_img(image_path)
                        list_images.append(img)

                    images = np.array([image.img_to_array(img) for img in list_images])

                    folder_name = folder + '_'+ 'aug'
                    folder_augmentation_path = os.path.join(class_folder_name , folder_name)

                    if not os.path.exists(folder_augmentation_path):
                        os.makedirs(folder_augmentation_path)

                    for i, img in enumerate(images):
                        # Memperluas dimensi karena `flow` mengharapkan tensor 4D
                        img = np.expand_dims(img, axis=0)

                        # Membuat iterator untuk augmentasi
                        aug_iter = datagen.flow(img, save_to_dir=folder_augmentation_path, save_prefix=f"{class_folder}_aug_{i}", save_format='jpg')

                        # Melakukan augmentasi sebanyak 2 kali
                        next(aug_iter)
                    print('{} sudah berhasil di augmentasi'.format(folder))

    def create_datagen(self):
        datagen = ImageDataGenerator(
        rotation_range=15,
        brightness_range=[0.5, 1.5],
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
        )
        return datagen

GenerateAugDataset()