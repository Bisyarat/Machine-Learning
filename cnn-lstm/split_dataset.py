import os
import shutil
from sklearn.model_selection import train_test_split

class SplitDataset:

    THIS_FILE_PATH = os.path.abspath(__file__)
    PARENT_DIRECTORY = os.path.abspath(os.path.join(THIS_FILE_PATH, ".."))

    DATASET_PATH = os.path.join(PARENT_DIRECTORY, 'dataset') #PATH LOCATION DATASET FOLDER
    TO_FOLDER = os.path.join(PARENT_DIRECTORY, 'split-dataset') #PATH LOCATION CREATE SPLIT DATASET FOLDER   

    def __init__(self):
        if os.path.exists(self.TO_FOLDER):
            #DELETE TO_FOLDER FOLDER IF EXITS FOR RE GENERATE SPLIT DATASET
            shutil.rmtree(self.TO_FOLDER)

        for subFolder in os.listdir(self.DATASET_PATH):

            train_path = os.path.join(self.TO_FOLDER, 'train', subFolder)
            test_path  = os.path.join(self.TO_FOLDER, 'test', subFolder)
            dev_path   = os.path.join(self.TO_FOLDER, 'dev', subFolder)

            subFolderPath = os.path.join(self.DATASET_PATH,subFolder)

            for folder in os.listdir(subFolderPath):

                folderPath = os.path.join(subFolderPath,folder)

                trainDataFiles, testDataFiles = train_test_split(os.listdir(folderPath), test_size=0.3)
                testDataFiles , devDataFiles  = train_test_split(testDataFiles, test_size=0.35)

                self.move_file_dataset(trainDataFiles, folderPath, train_path)
                self.move_file_dataset(testDataFiles, folderPath, test_path)
                self.move_file_dataset(devDataFiles, folderPath, dev_path)

    def move_file_dataset(self, files, sourcePath, toFolder):

        if not os.path.exists(toFolder):
            os.makedirs(toFolder)

        for file in files:
            sourceFilePath = os.path.join(sourcePath,file)
            toFolderPath   = os.path.join(toFolder,file)

            shutil.copy(sourceFilePath,toFolderPath)