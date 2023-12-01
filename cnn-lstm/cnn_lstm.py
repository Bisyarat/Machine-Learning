from split_dataset import SplitDataset
import os

THIS_FILE_PATH = os.path.abspath(__file__)
PARENT_DIRECTORY = os.path.abspath(os.path.join(THIS_FILE_PATH, ".."))

#PATH DATASET FOLDER TRAIN, TEST AND DEV SIGN LANGUAGE
SPLIT_DATASET_PATH = os.path.join(PARENT_DIRECTORY,'split-dataset')

# CHECK IF THE SPLIT DATASET FOLDER NOT EXITS OR NOT HAVE DIR
if not os.path.exists(SPLIT_DATASET_PATH) or os.path.isdir(SPLIT_DATASET_PATH):
    #GENERATE TRAIN, TEST AND DEV DATASET
    SplitDataset()

TRAIN_DATASET = os.path.join(SPLIT_DATASET_PATH, 'train')
TEST_DATASET  = os.path.join(SPLIT_DATASET_PATH, 'test')
DEV_DATASET   = os.path.join(SPLIT_DATASET_PATH, 'dev')

print(os.path.exists(TRAIN_DATASET))