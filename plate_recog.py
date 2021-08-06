
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
plt.style.use('dark_background')

#1. Read Input Image
img_ori = cv2.imread('images/skyblue4.jpeg')
cv2.imshow("input",img_ori)
height, width, channel = img_ori.shape
plt.show()
#2. Gray Level Scale -> RGB반대인 것에 주의.
img_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
cv2.imshow("img_gray", img_gray)
#3. GaussianBlur -> Gaussian을 씌우고 threshold를 해주면 노이즈를 더 줄일 수 있음.
#7 By 7을 하여 최대한 줄였음. -> odd matrix로 해야함.
img_gaussian = cv2.GaussianBlur(img_gray, ksize=(7, 7), sigmaX=0)
cv2.imshow("img_gaussian",img_gaussian)
#4. Threshold
img_threshold = cv2.adaptiveThreshold(img_gaussian, maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, 
    blockSize=19, C=9)
cv2.imshow("threshold", img_threshold)
#5. contour
img_contour = img_threshold.copy()
#Image Contour Info
#cv2.CHAIN_APPROX_NONE : 외곽 점 일일이 다찍음
contours, _ = cv2.findContours(img_threshold, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
#All zero 다차원 배열
img_contour =  np.zeros((height, width, channel), dtype=np.uint8)
#Contour Info로 비트맵 image draw 
#-1 모든 값을 연결
cv2.drawContours(img_contour, contours=contours, contourIdx=-1, color=(255, 255, 255))

#6. Rectangle contour
img_rectangle = img_contour.copy()
#Data를 List로 저장 후 Contour Ractangle의 길이, 넓이 등을 저장.
contours_dict = [] 

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(img_rectangle, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
    
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),         #Rectangle의 Center값
        'cy': y + (h / 2)
    })

#7. Rectangle Candidates -> Recognize하기 위해 임의의 Contour의 넓비 등을 가정.
img_candidate = img_rectangle.copy()
MIN_AREA = 80
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 1.0
#모든 정보를 저장
possible_contours = []     

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h']          #비율을 계산
    ratio = d['w'] / d['h']
    
    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO :     #맞는지 아닌지 확률로 계산
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)     #Index로 append저장

#8. Rectagle만 남기고 contour는 날림
img_candidate = np.zeros((height, width, channel), dtype=np.uint8) 

for d in possible_contours:
    cv2.rectangle(img_candidate, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

#9. Select Candidate ->확률적으로 Ract의 면적
img_select = img_candidate.copy()   
MAX_DIAG_MULTIPLYER = 5 # 5
MAX_ANGLE_DIFF = 12.2 # 12.0
MAX_AREA_DIFF = 0.5 # 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 3 # 3

def find_chars(contour_list):
    matched_result_idx = []
    
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        # append this contour
        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
        
        # recursive
        recursive_contour_list = find_chars(unmatched_contour)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx
    
result_idx = find_chars(possible_contours)

matched_result = []

for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

# visualize possible contours
img_select = np.zeros((height, width, channel), dtype=np.uint8)
for r in matched_result:
    for d in r:
        cv2.rectangle(img_select, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

#10. Select Rotate
PLATE_WIDTH_PADDING = 1.3 # 1.3     1.5
PLATE_HEIGHT_PADDING = 1.5 # 1.5     2.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 7

plate_imgs = []
plate_infos = []

for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
    
    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
    
    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
    
    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']]))
    
    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
    
    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
    
    img_rotate_rectangle = cv2.warpAffine(img_select, M=rotation_matrix, dsize=(width, height))
    #cv2.imshow("img_rotate_rectangle",img_rotate_rectangle)
    # Crop
    img_rotate_ori = cv2.warpAffine(img_ori, M=rotation_matrix, dsize=(0, 0))
    #cv2.imshow("img_rotate_ori",img_rotate_ori)
    img_crop_ori = cv2.getRectSubPix(img_rotate_ori, patchSize=(int(plate_width), int(plate_height)), 
        center=(int(plate_cx), int(plate_cy)))

cv2.imwrite('plate.jpg', img_crop_ori)

img_crop_ori = cv2.imread('plate.jpg')\

plt.figure("license plate capture")
plt.imshow(img_crop_ori) #11
plt.show()
#!/usr/bin/python
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
# --- Author         : Ahmet Ozlu
# --- Mail           : ahmetozlu93@gmail.com
# --- Date           : 8th July 2018 - before Google inside look 2018 :)
# -------------------------------------------------------------------------


import cv2
from color_recognition_api import color_histogram_feature_extraction
from color_recognition_api import knn_classifier
import os
import os.path
import sys

# read the test image
try:
    source_image = cv2.imread('plate.jpg')
except:
    source_image = cv2.imread('plate.jpg')
prediction = 'n.a.'

# checking whether the training data is ready
cv2.imshow("source_image", source_image)
PATH = './training2.data'

if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print ('training data is ready, classifier is loading...')
else:
    print ('training data is being created...')
    open('training2.data', 'w')
    color_histogram_feature_extraction.training()
    print ('training data is ready, classifier is loading...')

# get the prediction
color_histogram_feature_extraction.color_histogram_of_test_image(source_image)
prediction = knn_classifier.main('training2.data', 'test2.data')
print(prediction)
if prediction == 'skyblue':
    print('Electric car')
elif prediction == 'blue':
    print('dipolmat car')
else:
    print('Not Electric Car')
# cv2.putText(
#     source_image,
#     'Prediction: ' + prediction,
#     (15, 45),
#     cv2.FONT_HERSHEY_PLAIN,
#     3,
#     200,
#     )

# Display the resulting frame
# cv2.imshow('color classifier', source_image)
# cv2.waitKey(0)	







