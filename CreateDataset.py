import os
import shutil
from sklearn.model_selection import train_test_split
from GenerateAugDataset import GenerateAugDataset

class CreateDataset:

    PARENT_DIRECTORY = os.getcwd()

    DATASET_PATH = 'dataset/extract-images' #PATH LOCATION DATASET FOLDER
    DATA_AUGMENATION_PATH = 'augmentation-dataset'
    TO_FOLDER = 'model-dataset' #PATH LOCATION CREATE SPLIT DATASET FOLDER   

    def __init__(self):
        
        if not os.path.exists(self.DATA_AUGMENATION_PATH):
            GenerateAugDataset()
        
        if os.path.exists(self.TO_FOLDER):
            #DELETE TO_FOLDER FOLDER IF EXITS FOR RE GENERATE SPLIT DATASET
            shutil.rmtree(self.TO_FOLDER)

        for subFolder in os.listdir(self.DATASET_PATH):

            train_path = os.path.join(self.TO_FOLDER, 'train', subFolder)
            test_path  = os.path.join(self.TO_FOLDER, 'test', subFolder)
            dev_path   = os.path.join(self.TO_FOLDER, 'dev', subFolder)

            subFolderPath = os.path.join(self.DATASET_PATH,subFolder)
            
            folder = os.listdir(subFolderPath)

            trainDataFolder, testDataFolder = train_test_split(folder, test_size=0.5)
            testDataFolder , devDataFolder  = train_test_split(testDataFolder, test_size=0.35)
            
            #move to folder split dataset
            self.move_folder_dataset(trainDataFolder, subFolderPath, train_path)
            self.move_folder_dataset(testDataFolder, subFolderPath, test_path)
            self.move_folder_dataset(devDataFolder, subFolderPath, dev_path)

    def move_file_dataset(self, files, sourcePath, toFolder):

        if not os.path.exists(toFolder):
            os.makedirs(toFolder)

        for file in files:
            sourceFilePath = os.path.join(sourcePath,file)
            toFolderPath   = os.path.join(toFolder,file)

            shutil.copy(sourceFilePath,toFolderPath)

    def move_folder_dataset(self, folders, sourcePath, toFolder):
        if not os.path.exists(toFolder):
            os.makedirs(toFolder)
        
        for folder in folders:
            sourceFilePath = os.path.join(sourcePath,folder)
            toFolderPath   = os.path.join(toFolder,folder)

            shutil.copytree(sourceFilePath,toFolderPath)