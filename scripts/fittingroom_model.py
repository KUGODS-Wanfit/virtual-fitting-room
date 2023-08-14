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
from tensorflow.python.keras.optimizers import Adam


# 이미지 모델
def create_image_model(input_shape):
    inputs = Input(shape=input_shape)
    # 이미지 처리에 적합한 레이어들을 구성
    # 예: Conv2D, MaxPooling2D 등을 사용
    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    outputs = Dense(128, activation="relu")(x)
    return Model(inputs, outputs, name="image_model")


# 체형 데이터 모델
def create_body_model(num_body_features):
    inputs = Input(shape=(num_body_features,))
    # 체형 데이터를 처리하는 레이어들을 구성
    # 예: Dense 레이어 등을 사용
    x = Dense(64, activation="relu")(inputs)
    outputs = Dense(32, activation="relu")(x)
    return Model(inputs, outputs, name="body_model")


# 메인 가상 피팅룸 모델
def create_fitting_room_model(input_shape, num_body_features):
    image_input = Input(shape=input_shape, name="image_input")
    body_input = Input(shape=(num_body_features,), name="body_input")

    image_model = create_image_model(input_shape)
    body_model = create_body_model(num_body_features)

    image_features = image_model(image_input)
    body_features = body_model(body_input)

    concatenated = Concatenate()([image_features, body_features])
    x = Dense(256, activation="relu")(concatenated)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)  # 예측 클래스 수에 맞게 설정

    model = Model(
        inputs=[image_input, body_input],
        outputs=outputs,
        name="virtual_fitting_room",
    )
    return model


# 모델 생성
input_shape = (128, 128, 3)  # 이미지 크기
num_body_features = 10  # 체형 데이터의 특성 수
num_classes = 10  # 예측 클래스 수 (의류 스타일 등)
fitting_room_model = create_fitting_room_model(input_shape, num_body_features)

# 모델 컴파일
fitting_room_model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# 모델 구조 출력
fitting_room_model.summary()
