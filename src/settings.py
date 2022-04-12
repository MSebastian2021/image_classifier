import dotenv
import os

def set_env():
    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    dotenv_path = os.path.join(project_dir, '.env')
    raw_dir = os.path.join(project_dir, 'data','raw')
    data_set_url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'

    dotenv.load_dotenv(dotenv_path)
    dotenv.set_key(dotenv_path, "DATASET_URL",data_set_url )
    
    #get the filename of the zipped file: 
    zip_file = os.path.join(raw_dir, data_set_url.split("/")[-1])
    dotenv.set_key(dotenv_path, "RAW_DIR", raw_dir)
    dotenv.set_key(dotenv_path, "ZIP_FILE", zip_file)

    petimages_dir = os.path.join(raw_dir, "PetImages")
    dotenv.set_key(dotenv_path, "PETIMAGES_DIR", petimages_dir)
    
    dotenv.set_key(dotenv_path, "IMAGE_SIZE", "180,180")

    dotenv.set_key(dotenv_path, "BATCH_SIZE", "32")

    dotenv.set_key(dotenv_path, "NUM_CLASSES", "2")
    dotenv.set_key(dotenv_path, "EPOCHS", "50")
    dotenv.set_key(dotenv_path, "BUFFER_SIZE", "32")
    