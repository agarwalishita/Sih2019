import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread('hi.jpg') 
firebase = firebase.FirebaseApplication('https://ishita-b18d9.firebaseio.com/', None)
grayscaled=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)     ###loading an image
threshold=0
max_value=255
ret,o1=cv2.threshold(grayscaled,threshold,max_value,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
ret,o2=cv2.threshold(grayscaled,threshold,max_value,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
ret,o3=cv2.threshold(grayscaled,threshold,max_value,cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
ret,o4=cv2.threshold(grayscaled,threshold,max_value,cv2.THRESH_TOZERO_INV+cv2.THRESH_OTSU)
ret,o5=cv2.threshold(grayscaled,threshold,max_value,cv2.THRESH_TRUNC+cv2.THRESH_OTSU)

output=[grayscaled,o1,o2,o3,o4,o5]
titles=['original','binary','bnary inv','zero','zero inv','trunc']
for i in range(6): 
    plt.subplot(2,3,i+1)
    plt.imshow(output[i],cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()
result = firebase.post('/users',{'Scratch': '1'})
print result
{u'name': u'-Io26123nDHkfybDIGl7'}

#result = firebase.post('/users',{'X_FANCY_HEADER': 'VERY FANCY'})
print result == None
True
