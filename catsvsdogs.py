import cv2
import numpy as np 
import os 
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
Dir_train = 'C:/Users/OmarDASSER/Desktop/cvd test/train/'
Dir_test = r'C:\Users\OmarDASSER\Desktop\cvd test\test'
size_img = 50

LR = 0.001

MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')

def get_label(img):
	img_label = img.split('.')[0]
	if img_label == 'cat' : return [1,0]
	elif img_label == 'dog' : return [0,1]

def create_train_data():
	train_data = []
	for img in tqdm(os.listdir(Dir_train)):
		label = get_label(img)
		path = os.path.join(Dir_train,img)
		img = cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), (size_img,size_img))
		train_data.append([np.reshape(img,(-1,1)),np.array(label)])
		#labels = np.reshape(labels, (-1, 1))
	shuffle(train_data)
	np.save('train_data.npy',train_data)
	return  train_data

def process_test_data():
	test_data = []
	for img in tqdm(os.listdir(Dir_test)):
		path = os.path.join(Dir_test,img)
		img_num = img.split('.')[0]
		img = cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(size_img,size_img))
		test_data.append([np.array(img),np.array(img_num)])
	np.save('test_data.npy',test_data)
	return test_data

#tr_data = create_train_data()
tr_data = np.load('train_data.npy')



convnet = input_data(shape=[None, size_img, size_img, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
	model.load(MODEL_NAME)
	print('MODEL LOADED!')

train = tr_data[:-500]
test = tr_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1,size_img,size_img,1)
Y = [i[1] for i in train]

x_test = np.array([i[0] for i in test]).reshape(-1,size_img,size_img,1)
y_test = [i[1] for i in test]



model.fit({'input': X}, {'targets': Y}, n_epoch=5, validation_set=({'input': x_test}, {'targets': y_test}),
 snapshot_step=None, show_metric=True, run_id=MODEL_NAME)


model.save(MODEL_NAME)

import matplotlib.pyplot as plt 

#ts_data = process_test_data()
ts_data = np.load('test_data.npy')
fig = plt.figure()
shuffle(ts_data)
for num,data in enumerate(ts_data[:12]):
    # cat: [1,0]
    # dog: [0,1]
    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(size_img,size_img,1)
    
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 1: str_label='Dog'
    else: str_label='Cat'
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()