import os
import tensorflow as tf
from tensorflow import keras
from dotenv import load_dotenv
import data.load_data as load_data
import models.cnn as classifier
import settings

load_dotenv()
settings.set_env()

DATASET_URL = os.environ.get("DATASET_URL")
DATA_DIR = os.environ.get("DATA_DIR")
RAW_DIR = os.environ.get("RAW_DIR")
#get the filename of the zipped file: 
ZIP_FILE = os.environ.get("ZIP_FILE")
PETIMAGES_DIR = os.environ.get("PETIMAGES_DIR")
#print("PETIMAGES_DIR: ", os.path.exists(PETIMAGES_DIR))
IMAGE_SIZE = tuple(map(int, os.environ.get("IMAGE_SIZE").split(",")))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
NUM_CLASSES = int(os.environ.get("NUM_CLASSES"))
EPOCHS = int(os.environ.get("EPOCHS"))
BUFFER_SIZE = int(os.environ.get("BUFFER_SIZE"))

#create data dir, if it does not exist
if not (os.path.exists(DATA_DIR)): 
  # Create a new directory because it does not exist 
  os.makedirs(DATA_DIR)
  #print("The new directory is created!")

#download dataset
load_data.download_data(DATASET_URL, ZIP_FILE)

#check if zip file exists:
if (os.path.exists(ZIP_FILE)):
    #unzip file
    load_data.unzip_file(ZIP_FILE, RAW_DIR)

#check if PetImages folder exists
if (os.path.exists(PETIMAGES_DIR)):
    #filter corrupted images
    load_data.filter_corrupted_images(PETIMAGES_DIR)
    #generate dataset
    train_ds, val_ds = load_data.generate_dataset(PETIMAGES_DIR, IMAGE_SIZE, BATCH_SIZE)

    #create the model
    model = classifier.make_model(input_shape=IMAGE_SIZE + (3,), num_classes=NUM_CLASSES)
    train_ds = train_ds.prefetch(buffer_size=BUFFER_SIZE)
    val_ds = val_ds.prefetch(buffer_size=BUFFER_SIZE)
    classifier.train(model, train_ds, val_ds, EPOCHS)