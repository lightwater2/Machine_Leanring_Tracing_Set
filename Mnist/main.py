# 참고 : https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6
# 1. 자체 라이브러리
from lib import HistoryManager

time = HistoryManager.TimeChecker()  # 모델 및 결과 이름 작성에 사용
import os
HistoryManager.SaveToCode("../Mnist/archive/", fileName=os.path.abspath(__file__), time=time)  # 컴파일 시점 코드저장
# 2. 이미지 로드 및 전처리 단계

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


def TestSet():
    return (pd.read_csv("input/test.csv")/255.0).values.reshape(-1, 28, 28, 1)


def TrainSet():
    random_seed = 2
    train = pd.read_csv("input/train.csv")
    Y_t = to_categorical(train["label"], num_classes=10)
    X_t = (train.drop(labels=["label"], axis=1) / 255.0).values.reshape(-1, 28, 28, 1)
    return train_test_split(X_t, Y_t, test_size=0.1, random_state=random_seed)


test = TestSet()
X_train, X_val, Y_train, Y_val = TrainSet()

# 3. 트레이닝 단계
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import History


# 3-1. 모델링 함수
def setModel():
    md = Sequential()

    md.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                  activation='relu', input_shape=(28, 28, 1)))
    md.add(Conv2D(filters=32, kernel_size=(5, 5), padding='Same',
                  activation='relu'))
    md.add(MaxPooling2D(pool_size=(2, 2)))
    md.add(Dropout(0.25))

    md.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                  activation='relu'))
    md.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same',
                  activation='relu'))
    md.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    md.add(Dropout(0.25))

    md.add(Flatten())
    md.add(Dense(256, activation="relu"))
    md.add(Dropout(0.5))
    md.add(Dense(10, activation="softmax"))
    return md


# 3-2. 트레이닝 세팅
rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=0.00001)
epochs = 1  # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images
datagen.fit(X_train)
model = HistoryManager.LoadToModel(setModel)  # 두번째 매개변수로 파일명(.h5) 입력시 이미 학습된 모델 로드 default는 새 학습

model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
History()  # keras history 셋업
history = model.fit(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    epochs=epochs, validation_data=(X_val, Y_val),
                    verbose=2, steps_per_epoch=X_train.shape[0] // batch_size,
                    callbacks=[learning_rate_reduction])

HistoryManager.LoggingHistory(history=history, time=time)


HistoryManager.SaveToModel(model, directory="../Mnist/models/", time=time)
pred = model.predict(test)


# 정답에 맞는 포매팅, 저장함수에 콜백으로 전달
def Submission(prediction):
    # thresHold = 0.5  # 임계값 필요시 사용하여 정답 작성
    df = pd.DataFrame(prediction)
    df2 = pd.DataFrame(columns=['ImageId', 'Label'])
    df2['ImageId'] = df.index + 1
    df2['Label'] = df.idxmax(axis="columns")
    return df2


HistoryManager.SaveToPredict(pred=pred, callback=Submission, directory="../Mnist/predicts/", time=time)
