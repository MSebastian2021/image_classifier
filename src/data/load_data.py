import urllib
import zipfile
import os
import tensorflow as tf

def download_data(src, dest):
    #get the filename
    #filename = src.split("/")[-1] 
    urllib.request.urlretrieve(src, dest)
    print("downloading ", src, " ...")

def unzip_file(path_to_zip_file, extract_dir):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        print("extracting file... ")

def filter_corrupted_images(petimages_dir):
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        #folder_path = os.path.join("PetImages", folder_name)
        folder_path = os.path.join(petimages_dir, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)
    print("Deleted %d images" % num_skipped)

def generate_dataset(petimages_dir, image_size, batch_size):
    #image_size = (180, 180)
    #batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        petimages_dir,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        petimages_dir,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    return(train_ds,val_ds)