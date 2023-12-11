import cv2
import os
import re

class ExtractImage:
    
    def __init__(self,saveFolderPath,folderName,folderFiles):

        LOCATION_SAVE_FILE = os.path.join(saveFolderPath, folderName)
        
        for subFolder in os.listdir(folderFiles):

            subFolderPath = os.path.join(folderFiles,subFolder)

            for file in os.listdir(subFolderPath):

                try:
                    filePath = os.path.join(subFolderPath,file)

                    folderFileName = self.create_folder_name(file)
                    fileName = file.split('.')[0]

                    videoFile = cv2.VideoCapture(filePath)

                    if not videoFile.isOpened():
                        with open('file_error.txt', 'a') as errorLog:
                            errorLog.write('error when open video file: {}\n'.format(file))

                    saveFolder = os.path.join(LOCATION_SAVE_FILE,folderFileName)

                    if not os.path.exists(saveFolder):
                        os.makedirs(saveFolder)

                    numFrame = 1

                    while videoFile.isOpened():
                        condition,frame = videoFile.read()

                        if condition is False:
                            with open('file_error.txt', 'a') as errorLog:
                                errorLog.write('error file: {} when create frame {}\n'.format(file,numFrame))
                                break

                        saveFrame = os.path.join(saveFolder, f"{fileName}_{numFrame}.jpg")
                        cv2.imwrite(saveFrame, frame)

                        numFrame += 1

                    videoFile.release()

                except Exception as error:
                    with open('exception_error.txt', 'a') as errorLog:
                        errorLog.write('error file: {} - {}\n'.format(file,error))


    def create_folder_name(self, filename):
        pattern = r'\b[A-Za-z0-9]*' 
        match = re.search(pattern, filename)
        
        if match is not None:
            return match.group()
        return False
    
    def create_file_name(self, filename):
        pattern = r'\b[A-Za-z0-9]+_[A-Za-z0-9]*' 
        match = re.search(pattern, filename)
        
        if match is not None:
            return match.group()
        return False