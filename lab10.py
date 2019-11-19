import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('image.jpg')

imgOrig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(imgOrig, cv2.COLOR_BGR2GRAY)
	
plt.subplot(3,2,1),plt.imshow(imgOrig,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3,2,2),plt.imshow(gray,cmap = 'gray')
plt.title('Gray'), plt.xticks([]), plt.yticks([])

dst = cv2.cornerHarris(gray, 2, 3, 0.01)

plt.subplot(3,2,3),plt.imshow(dst,cmap = 'gray')
plt.title('DST'), plt.xticks([]), plt.yticks([])

imgHarris = imgOrig.copy()

threshold = 0.01;
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris, (j,i), 3, (255, 1, 1), -1)

plt.subplot(3,2,4),plt.imshow(imgHarris,cmap = 'gray')
plt.title('DST COPY'), plt.xticks([]), plt.yticks([])

corners = cv2.goodFeaturesToTrack(gray, 1000, 0.05, 0.1)

imgShiTomasi = imgOrig.copy()

for i in corners:
    x, y = i.ravel()
    cv2.circle(imgShiTomasi, (x, y), 3, (255, 1, 1), -1)

plt.subplot(3,2,5),plt.imshow(imgShiTomasi,cmap = 'gray')
plt.title('Tomasi Algo Copy'), plt.xticks([]), plt.yticks([])

sift = cv2.xfeatures2d.SIFT_create(50)
(kps, descs) = sift.detectAndCompute(gray, None)
print("# kps: {}, descriptors: {}".format(len(kps), descs.shape))

imgSift = cv2.drawKeypoints(imgOrig, kps, outImage=None, color=(255, 1, 1), flags=4)

plt.subplot(3,2,6),plt.imshow(imgSift,cmap = 'gray')
plt.title('Sift Algo'), plt.xticks([]), plt.yticks([])

plt.show()



