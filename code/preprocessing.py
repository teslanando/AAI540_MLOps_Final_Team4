import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def resize_images(img_path, output_path, size=(128, 128)):
    img = Image.open(img_path)
    img = img.resize(size, Image.ANTIALIAS)
    img.save(output_path)

def preprocess_dataset(images_path, labels_file, output_path):
    df = pd.read_csv(labels_file)
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(images_path, x + '.jpg'))
    df['label'] = df['diagnosis'].apply(lambda x: 1 if x == 'malignant' else 0)
    
    # Resize images and split dataset
    df['resized_path'] = df['image_path'].apply(lambda x: os.path.join(output_path, os.path.basename(x)))
    df['image_path'].apply(lambda x: resize_images(x, os.path.join(output_path, os.path.basename(x))))
    
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'])
    train_df, val_df = train_test_split(train_df, test_size=0.25, stratify=train_df['label'])  # 0.25 * 0.8 = 0.2

    # Data augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Save datasets
    train_df.to_csv(os.path.join(output_path, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_path, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test.csv'), index=False)

    return datagen, train_df, val_df, test_df

base_dir = "/opt/ml/processing"
images_path = f"{base_dir}/input/images"
labels_file = f"{base_dir}/input/labels.csv"
output_path = f"{base_dir}/output"

datagen, train_df, val_df, test_df = preprocess_dataset(images_path, labels_file, output_path)
