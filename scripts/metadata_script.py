import os
import pandas as pd

data_folder = "data/"
sub_datasets = []
metadata = []

for sub_dataset in sub_datasets:
    dataset_folder = os.path.join(data_folder, sub_dataset)

    for image_name in os.listdir(dataset_folder):
        # 이미지 파일명에서 정보 추출 (예: 카테고리)
        category = sub_dataset
        image_path = os.path.join(dataset_folder, image_name)

        # 메타데이터로 추가할 정보를 추출하거나 가공하여 metadata 리스트에 추가
        metadata.append(
            {
                "image_name": image_name,
                "category": category,
                "image_path": image_path,
            }
        )

# metadata 리스트를 데이터프레임으로 변환
metadata_df = pd.DataFrame(metadata)

# 데이터프레임을 CSV 파일로 저장
metadata_df.to_csv(os.path.join(data_folder, "metadata.csv"), index=False)
