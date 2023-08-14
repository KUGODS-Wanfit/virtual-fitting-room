import tensorflow as tf

from tensorflow.python.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Concatenate,
)
from tensorflow.python.keras.models import Model


def create_fitting_room_model(input_shape, num_body_features):
    # 이미지 입력
    image_input = Input(shape=input_shape)

    # CNN 레이어 (옷 스타일 추출)
    conv_layer = Conv2D(32, (3, 3), activation="relu")(image_input)
    maxpool_layer = MaxPooling2D((2, 2))(conv_layer)
    flatten_layer = Flatten()(maxpool_layer)
    cnn_output = Dense(128, activation="relu")(flatten_layer)

    # 체형 데이터 입력
    body_type_input = Input(shape=(num_body_features,))

    # MLP 레이어 (체형 데이터 처리)
    mlp_output = Dense(64, activation="relu")(body_type_input)

    # CNN과 MLP의 출력을 병합
    merged_layer = Concatenate()([cnn_output, mlp_output])

    # 출력 레이어 (맞춤성 예측)
    output_layer = Dense(1, activation="sigmoid")(merged_layer)

    # 모델 정의
    model = Model(inputs=[image_input, body_type_input], outputs=output_layer)
    return model


# 모델 생성
input_shape = (128, 128, 3)  # 이미지 크기
num_body_features = 10  # 체형 데이터의 특성 수
fitting_room_model = create_fitting_room_model(input_shape, num_body_features)

# 모델 요약 출력
fitting_room_model.summary()
