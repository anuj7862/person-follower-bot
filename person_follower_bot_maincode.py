# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
##################################
import serial
#**********ser = serial.Serial("/dev/ttyACM1",9600,timeout=0)
##################################
# Create a pipeline
pipeline = rs.pipeline()

#Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)
# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D , Flatten , Dense , BatchNormalization , MaxPooling2D, Dropout
from keras import optimizers
import numpy as np
import cv2
import os
import collections
import copy
from random import randint
import matplotlib.pyplot as plt
#keras.backend.set_floatx('float64')



def cm():
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image
        aligned_frames = align.process(frames)
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # 640x480 depth image
        color_frame = aligned_frames.get_color_frame()
        
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        #print(depth_image[240,360])
        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        #bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)
         
        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))
        #cv2.namedWindow('Align Example', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('Align Example', images)
        np.putmask(depth_image, depth_image>6000,6000)
        #print(depth_image)
        #plt.imshow(color_image)
        #plt.show()
        depth_image = (depth_image/6000)*255
        depth_img = np.zeros((480,640,1))
        depth_img[:,:,0] = depth_image
        print(depth_img.shape,color_image.shape)
        #break
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        #if key & 0xFF == ord('q') or key == 27:
        #    cv2.destroyAllWindows()
        break
    return depth_img,color_image


def random_patch(x1,y1,ht,wd,img):
    d=80
    y2 = y1
    x2 = x1
    while((y2 < y1+3 and y2 > y1-3) or (y2> 640) or (y2 < 0) ):
      y2=np.random.randint(y1-d,y1+d)
    while((x2 < x1+3 and x2 > x1-3) or (x2 >480) or (x2 <0) ):
      x2=np.random.randint(x1-d,x1+d)
    #if(x1-d > 0):
    #x2=np.random.randint(x1-d,x1+d)
    #else:
    #  x2 = np.random.randint(0, x1+d)
    patch = img[x2:x2+ht, y2:y2+wd, :]
    #print(x2,y2)
    return patch



def alpha_value(img, x1, y1, pre_depth):
    crop_img = copy.copy(img[int(x1):int(x1+h), int(y1):int(y1+w)])
    crop_img = crop_img*6.0/255
    pixels_within_range = ((crop_img<pre_depth + .25) & (crop_img>pre_depth - .25))
    total_pixels = pixels_within_range.sum()
    pixels_sum = sum(crop_img[pixels_within_range].T)
    alpha = pixels_sum/total_pixels
    if(total_pixels == 0):
        alpha = pre_depth
    return alpha

def in_bounding_box_right(img, x1, y1, count, alpha):
    lower_range = alpha - .25
    upper_range = alpha + .25
    img_sub = copy.copy(img[x1:x1+h, y1-sliding_h:y1])
    img_add = copy.copy(img[x1:x1+h, y1-sliding_h+w:y1+w])
    count_to_sub = ((img_sub > lower_range) & (img_sub < upper_range)).sum()
    count_to_add = ((img_add > lower_range) & (img_add < upper_range)).sum()
    count = count - count_to_sub + count_to_add
    ratio = (count*1.0/(h*w))
    return [ratio, count]

def in_bounding_box_left(img, x1, y1, count, alpha):
    lower_range = alpha - .25
    upper_range = alpha + .25
    img_sub = copy.copy(img[x1:x1+h, y1+w:y1+sliding_h+w])
    img_add = copy.copy(img[x1:x1+h, y1:y1+sliding_h])
    count_to_sub = ((img_sub > lower_range) & (img_sub < upper_range)).sum()
    count_to_add = ((img_add > lower_range) & (img_add < upper_range)).sum()
    count = count - count_to_sub + count_to_add
    ratio = (count*1.0/(h*w))
    return [ratio, count]


def initial_count(img, x1, y1, alpha):
    lower_range = alpha - .25
    upper_range = alpha + .25
    patch = copy.copy(img[x1:x1+h, y1:y1+w])
    count = ((patch > lower_range) & (patch < upper_range)).sum()
    return count


def right_sweep(img, x1, y1, alpha, count):
    c = 0
    ratio = 1
    while ratio > .7 or c < 5 :
        y1 = y1 + sliding_w
        if(y1 > 640):
            break
        [ratio, count] = in_bounding_box_right(img, x1, y1, count, alpha)
        list_x.append([x1, y1])
        c = c +1
    return c


def left_sweep(img, x1, y1, alpha, count):
    c = 0
    ratio = 1
    while ratio > .7 or c < 5:
        y1 = y1 - sliding_w
        if(y1 < 0):
            break
        [ratio, count] = in_bounding_box_left(img, x1, y1, count, alpha)
        list_x.append([x1, y1])
        c = c + 1
    return c


def one_in_all(image, x1, y1, alpha):
    img = copy.copy(image)
    img = img*6.0/255
    total_bounding_box = 0
    x2 = x1
    count = initial_count(img, x1, y1, alpha)
    print(count)
    ratio = count*1.0/(h*w)
    print(ratio)
    i = 0
    while (ratio > .7 or i < 5) and x1 >= sliding_h:
        list_x.append([x1, y1])
        c = right_sweep(img, x1, y1, alpha, count)
        total_bounding_box = total_bounding_box + c
        c = left_sweep(img,x1, y1, alpha, count)
        total_bounding_box = total_bounding_box + c
        x1 = x1 - sliding_h
        count = initial_count(img, x1, y1, alpha)
        ratio = count*1.0/(h*w)
        i += 1

    i = 0
    x1 = x2
    count = initial_count(img, x1, y1, alpha)
    ratio = count*1.0/(h*w)
    while (ratio> .7 or i < 5) and x1<84:
        list_x.append([x1, y1])
        x1 = x1 + sliding_h
        c = right_sweep(img, x1, y1, alpha, count)
        total_bounding_box = total_bounding_box + c
        c = left_sweep(img,x1,y1, alpha, count)
        total_bounding_box = total_bounding_box + c
        count = initial_count(img, x1, y1, alpha)
        ratio = count*1.0/(h*w)
        i += 1

from collections import deque
sliding_h = 8
sliding_w = 8
patch_1_global = deque(maxlen = 17)
images = deque(maxlen = 40)
cordis = deque(maxlen = 40)
w = 100
h = 390

#image_files = os.listdir("lecture_hall/left")
#image_files = sorted([f.lower() for f in image_files])
#depth_files = os.listdir("lecture_hall/depth")
#depth_files = sorted([f.lower() for f in depth_files])
#ground_truth = open("lecture_hall/GroundTruth.txt")
lines = []

#depth_img = cv2.imread("depth00000000.jpg")
#color_img = cv2.imread("left00000000.jpg")
while True:
    depth_img,color_img = cm()
    cv2.rectangle(color_img,(276,130),(276+100,130+390),(255,0,0),2)
    cv2.imshow('Image0',color_img)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('\n') or key == 27:
        cv2.destroyAllWindows()
        break
    

img = np.concatenate([color_img, depth_img], axis = 2)
img = img[:, :, :4]
patch_1 = img[32:32+390, 276:276+100, :]
patch_1_global.append(patch_1)
images.append(img)
box = [32, 276]
cordis.append(box)

x_train = []
for i in range(0,50):
    patch_0 = random_patch(box[0],box[1],h,w,img)
    patch_0 = cv2.resize(patch_0 ,(28, 28))
    imgtoarray = np.asarray(patch_0)
    x_train.append(imgtoarray)
for i in range(0,50):
    x_train.append(cv2.resize(patch_1, (28, 28)))

x_train = np.array(x_train)

y_train = np.zeros(100)
y_train[50:] = np.ones(50)
print(x_train.shape)

cnn_v1 = Sequential()
cnn_v1.add(Conv2D(32 , kernel_size = 3 , strides = 1 , input_shape = (28,28,4) , padding = 'same', activation = 'tanh'))
cnn_v1.add(Dropout(.2))
cnn_v1.add(MaxPooling2D())
cnn_v1.add(Conv2D(64 , kernel_size = 3 , strides = 1 , padding = 'same', activation = 'tanh'))
cnn_v1.add(Dropout(.2))
cnn_v1.add(MaxPooling2D())
cnn_v1.add(Flatten())
cnn_v1.add(Dense(128 , activation = 'tanh'))
cnn_v1.add(Dense(1 , activation = 'sigmoid'))
cnn_v1.summary()

opt = optimizers.Adam(lr = .001)
cnn_v1.compile(optimizer = opt , loss = 'binary_crossentropy' , metrics = ['accuracy'])
cnn_v1.fit(x_train , y_train , epochs = 15, batch_size = 50, shuffle = True )

# = cv2.imread("lecture_hall/left/" + image_files[1])
#depth_img = cv2.imread("lecture_hall/depth/" + depth_files[1])
depth_img,color_img = cm()
img = np.concatenate([color_img, depth_img], axis = 2)
img = img[:, :, :4]
patch_1 = img[32:32+390, 276:276+100, :]
patch_1 = cv2.resize(patch_1, (28, 28))
np.resize(patch_1, (1,28, 28,4))
print(cnn_v1.predict(np.resize(patch_1, (1,28, 28,4))))

#depth_img = cv2.imread("depth00000000.jpg")[:, :, 0]
#depth_img = depth_img*6.0/255
initial_alpha = np.sum(depth_img[box[0]+h//2-5 : box[0]+h//2+5, box[1]+w//2-5 : box[1]+w//2+5])/100   ####
pre_depth = initial_alpha
print("#####################",pre_depth)
pre_depth = pre_depth*6.0/255

new_cordis = cordis[0]
while(1):
    print(i)
    #color_img = cv2.imread("lecture_hall/left/" + image_files[i])
    #depth_img = cv2.imread("lecture_hall/depth/" + depth_files[i])
    depth_img,color_img = cm()
    #depth_img = depth_img*6.0/255
    img = np.concatenate([color_img, depth_img], axis = 2)
    img = img[:, :, :4]
    depth = alpha_value(img[:,:,3],new_cordis[0],new_cordis[1], pre_depth)
    print("depth of current image ", depth)
    print("depth of previous image ", pre_depth)
    if(depth < pre_depth + .3 and depth > pre_depth - .3):
        pre_depth = depth
    list_x = []
    one_in_all(img[:, :, 3],new_cordis[0],new_cordis[1],pre_depth)
    #print(list_x)
    if(list_x == []):
        print("\n\n\n")
        continue
    predictions = []
    for u in range(len(list_x)):
        x1 = list_x[u][0]
        y1 = list_x[u][1]
        if(x1>479 or x1 < 0):
            print("###############################################",x1)    
        if(y1>639 or y1 < 0):
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",y1)
        p = img[x1:x1+h, y1:y1+w, :]
        p = np.asarray(p)
        p = cv2.resize(p, (28, 28))
        p = np.resize(p , (1,28,28,4))
        predictions.append(cnn_v1.predict(p)[0])
    #print("predictions ", predictions)
    #print("number of sliding_window ", len(list_x))
    index = predictions.index(max(predictions))
    print("predicted_value ", predictions[index])
    if(predictions[index] > 0.5):
        new_cordis = list_x[index]
        print("cordis are ", new_cordis)
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@arduino send code
        arduino_angle_dir = 1
        arduino_depth_dir = 1
        arduino_angle = new_cordis[1] - 320
        arduino_depth = (depth -1.5)*100
        if(arduino_angle < 0):
            arduino_angle_dir = 0
            arduino_angle = -1*arduino_angle
        else :
            arduino_angel_dir = 1
        if(arduino_depth < 0):
            arduino_depth_dir = 0
            arduino_depth = -1*arduino_depth
        else:
            arduino_depth_dir = 1
        arduino_depth = int(arduino_depth)    
        arduino_angle = format(arduino_angle,'03d')
        arduino_depth = format(arduino_depth,'03d')
        #print("arduino_depth,arduino_angle :",arduino_depth,arduino_angle)
        send_str = str(arduino_angle_dir) + str(arduino_angle) + str(arduino_depth_dir)+str(arduino_depth) +"\n"
        #*************ser.write(send_str.encode())
        print("string :" ,send_str)
        #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@arduino code
    
        images.append(img)
        cordis.append(new_cordis)
        patch_1 = img[new_cordis[0]:new_cordis[0]+h, new_cordis[1]:new_cordis[1]+w, :]
        color_image = color_img[new_cordis[0]:new_cordis[0]+h, new_cordis[1]:new_cordis[1]+w, :]
        patch_1_global.append(patch_1)
        patch_rev = patch_1_global
        patch_rev.reverse()
        #f, axrr = plt.subplots(1, 2)
        #axrr[0].imshow(patch_1[:, :, 0])
        #axrr[1].imshow(img[:, :, 0])
        print("fdsf")
        cv2.imshow('Image + str(i)',color_image)
        cv2.imshow('Image1',color_img)
        cv2.waitKey(2)
        x_train = []
        for i in range(10):
            #x = randint(0, len(cordis)-1)
            #x_train.append(np.asarray(cv2.resize(random_patch(cordis[x][0], cordis[x][1], h, w, images[x]), (28, 28))))
            x_train.append(np.asarray(cv2.resize(random_patch(new_cordis[0], new_cordis[1], h, w, img), (28, 28))))

        for i in range(10):
            x = 0
            if(len(patch_rev) < 16):
                x = randint(0, len(patch_rev)-1)
            else:
                x = randint(0, 15)
            x_train.append(np.asarray(cv2.resize(patch_rev[x], (28, 28))))
        x_train = np.asarray(x_train)
        y_train = np.zeros(20)
        y_train[10:] = np.ones(10)
        cnn_v1.fit(x_train, y_train, epochs = 1, batch_size = 20, shuffle = True)
    
pipeline.stop()
