# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow

import tarfile

#%% 압축해제

# # tar 파일 경로
# tar_file_path = "C:/Users/user/Desktop/capstone/089.차량 내 탑승자 상황 인식 영상 데이터/01.데이터/2.Validation/라벨링데이터/abnormal_230303_add/VL1.tar"

# # tar 파일 객체 생성
# tar = tarfile.open(tar_file_path)

# # 압축 해제
# tar.extractall('C:/Users/user/Desktop/capstone/089.차량 내 탑승자 상황 인식 영상 데이터/01.데이터/2.Validation/라벨링데이터/abnormal_230303_add/')

# # tar 파일 객체 닫기
# tar.close()




# # tar 파일 경로
# tar_file_path = "C:/Users/user/Desktop/capstone/089.차량 내 탑승자 상황 인식 영상 데이터/01.데이터/2.Validation/원천데이터/abnormal_230303_add/VS1.tar"

# # tar 파일 객체 생성
# tar = tarfile.open(tar_file_path)

# # 압축 해제
# tar.extractall('C:/Users/user/Desktop/capstone/089.차량 내 탑승자 상황 인식 영상 데이터/01.데이터/2.Validation/라벨링데이터/abnormal_230303_add/')

# # tar 파일 객체 닫기
# tar.close()


#%% 라벨 데이터 프레임 생성

#SGA2100920 <- 이폴더 경로 잘못들어가있어서 바꿨음 (SGA2101523 이폴더 안에 들어가 있길래 밖으로 뺐음(날짜보니까 잘못 들어간 데이터 같기도하고) (SGA2100920S0108 여기엔 라벨도 없음))
import json
import os
from glob import glob
import pandas as pd

os.chdir('C:\\Users\\user\\Desktop\\capstone\\089.차량 내 탑승자 상황 인식 영상 데이터\\01.데이터\\2.Validation\\라벨링데이터\\abnormal_230303_add\\')

#상위파일목록
path = glob("*")
path.remove('VL1.tar')
#path.remove("SGA2100920") #<- 얘 하위 폴더에 라벨 없는 데이터 몇개있음

#빈 df 생성
result_df = pd.DataFrame(columns=["상위파일", "id", "label"]) 
new_df = pd.DataFrame(columns=["상위파일", "ECG", "EEG_0", "EEG_1", "PPG", "SPO2", "emotion"]) 

for i in path:
    #중간파일 목록
    new_path = os.listdir("C:\\Users\\user\\Desktop\\capstone\\089.차량 내 탑승자 상황 인식 영상 데이터\\01.데이터\\2.Validation\\라벨링데이터\\abnormal_230303_add\\" + i)
    
    for j in new_path:
        label_path = os.listdir("C:\\Users\\user\\Desktop\\capstone\\089.차량 내 탑승자 상황 인식 영상 데이터\\01.데이터\\2.Validation\\라벨링데이터\\abnormal_230303_add\\" + i +"\\"+ j + "\\label")
        
        with open(("C:\\Users\\user\\Desktop\\capstone\\089.차량 내 탑승자 상황 인식 영상 데이터\\01.데이터\\2.Validation\\라벨링데이터\\abnormal_230303_add\\" + i +"\\"+ j + "\\label\\"+ label_path[0]), 'r', encoding='UTF8') as f:
            json_data = json.load(f)
            
            print("상위파일:", i, "id :" , json_data["scene_info"]['scene_id'] , "라벨명:",json_data["scene_info"]["category_name"])
            
            #result_df
            result_list = []
            result_list.append([i,
                                json_data["scene_info"]['scene_id'],
                                json_data["scene_info"]["category_name"]])
            
            df = pd.DataFrame(result_list, columns=["상위파일", "id", "label"] )
            result_df = pd.concat([result_df, df])

            #new_df 
            new_list = []
            new_list.append([i,
                             json_data["scene"]["sensor"][0]["ECG"], 
                             json_data["scene"]["sensor"][0]["EEG"][0],
                             json_data["scene"]["sensor"][0]["EEG"][1],
                             json_data["scene"]["sensor"][0]["PPG"],
                             json_data["scene"]["sensor"][0]["SPO2"],
                             json_data["scene"]["data"][0]["occupant"][0]["emotion"]
                             ])
            df_2 = pd.DataFrame(new_list,columns=["상위파일", "ECG", "EEG_0", "EEG_1", "PPG", "SPO2", "emotion"] )
            new_df = pd.concat([new_df,df_2])


result_df.info()
result_df.describe()
result_df["label"].unique()

json_data["scene"]["sensor"][0]["ECG"]
json_data["scene"]["sensor"][0]["EEG"][0]
json_data["scene"]["sensor"][0]["EEG"][1]
json_data["scene"]["sensor"][0]["PPG"]
json_data["scene"]["sensor"][0]["SPO2"]
json_data["scene"]["data"][0]["occupant"][0]["emotion"]


#%% 

#train / test split

image_path = result_df['상위파일'] +"/"+ result_df['id']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(image_path, result_df['label'], test_size=0.2, random_state=42)

#%%
import os

x_train = 'C:/Users/user/Desktop/capstone/089.차량 내 탑승자 상황 인식 영상 데이터/01.데이터/2.Validation/라벨링데이터/abnormal_230303_add/' + x_train + "/" +"img/"

x_train = x_train.reset_index(drop=True)

y_train = y_train.reset_index(drop=True)


# %%
# 검색하고자 하는 경로

import os
import glob

# 이미지 파일들이 있는 폴더 경로

# 폴더 내 모든 이미지 파일 경로 불러오기
total_x_train =pd.DataFrame()

for i in range(len(x_train)):
    folder_path = x_train[i]
    image_paths = glob.glob(os.path.join(folder_path, "*.jpg"))
    x_train_image_path = pd.DataFrame(image_paths)
    x_train_image_path['label'] = y_train[i]
    total_x_train = pd.concat([total_x_train,x_train_image_path])
    print(i)
    
total_x_train = total_x_train.reset_index(drop=True)

total_x_train.columns =['image_path','label']


# %%
# 이미지 데이터 전처리

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,validation_split =0.2)


train_generator = train_datagen.flow_from_dataframe(
    dataframe=total_x_train,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

#%%

# pre-trained ResNet50 모델 불러오기
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# ResNet50 모델 불러오기 (include_top=False 로 설정하여 fully-connected layer 제외)
resnet50 = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

# ResNet50의 출력에 fully-connected layer 추가하기
x = resnet50.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)

# 새로운 모델 구성하기
model = Model(inputs=resnet50.input, outputs=predictions)

# 모든 레이어를 동결하고 마지막 레이어 2개만 학습하도록 설정
for layer in resnet50.layers:
    layer.trainable = False
model.layers[-1].trainable = True
model.layers[-2].trainable = True

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


import tensorflow as tf
# 모델 학습
with tf.device('/device:GPU:0'): 
    history = model.fit(train_generator, epochs=100)
