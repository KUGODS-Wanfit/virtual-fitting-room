import os
import cv2
import numpy as np
import shutil
import random

# 데이터 폴더 설정
data_folder = "data/"
sub_datasets = [
    "in_shop_clothes_retrieval",
    "3d_body_scans",
    "ecommerce_fit_data",
]
target_size = (128, 128)  # 목표 이미지 크기
train_ratio = 0.7
valid_ratio = 0.15
test_ratio = 0.15

# 데이터 전처리 및 분할
for sub_dataset in sub_datasets:
    dataset_folder = os.path.join(data_folder, sub_dataset)
    preprocessed_folder = os.path.join(data_folder, "preprocessed", sub_dataset)
    os.makedirs(preprocessed_folder, exist_ok=True)

    image_files = os.listdir(dataset_folder)
    random.shuffle(image_files)
    num_samples = len(image_files)
    num_train = int(num_samples * train_ratio)
    num_valid = int(num_samples * valid_ratio)

    train_images = image_files[:num_train]
    valid_images = image_files[num_train : num_train + num_valid]
    test_images = image_files[num_train + num_valid :]

    train_folder = os.path.join(data_folder, "train", sub_dataset)
    valid_folder = os.path.join(data_folder, "valid", sub_dataset)
    test_folder = os.path.join(data_folder, "test", sub_dataset)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(valid_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for image_name in train_images:
        image_path = os.path.join(dataset_folder, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, target_size)
        image = image / 255.0
        preprocessed_image_path = os.path.join(preprocessed_folder, image_name)
        cv2.imwrite(preprocessed_image_path, image)
        shutil.copy(
            os.path.join(dataset_folder, image_name),
            os.path.join(train_folder, image_name),
        )

    for image_name in valid_images:
        shutil.copy(
            os.path.join(dataset_folder, image_name),
            os.path.join(valid_folder, image_name),
        )

    for image_name in test_images:
        shutil.copy(
            os.path.join(dataset_folder, image_name),
            os.path.join(test_folder, image_name),
        )

print("Data preprocessing and splitting completed.")
