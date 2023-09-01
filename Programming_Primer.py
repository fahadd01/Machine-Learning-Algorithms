#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib import image as mpimg
from matplotlib.pyplot import figure
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import cv2


# In[11]:


#Task 1

#Lists
#Make another list named cubes and append the cubes of the given list in this list and print it
nums = [3, 4, 7, 8, 15]
cubes = [1, 8]
for i in range (5):
    cubes.append((nums[i])**3)
print(cubes)

#Dictionaries
#Add the following data to the dictionary: ‘person’: 2, ‘cat’: 4, ‘spider’: 8, ‘horse’: 4 as key value pairs
dic = {'person': 2, 'cat': 4, 'spider': 8, 'horse': 4}

#Use the ‘items’ method to loop over the dictionary and print the animals and their corresponding legs
print(dic.items())

#Sum the legs of each animal, and print the total at the end
sum_legs = dic['person'] + dic['cat'] + dic ['spider'] + dic['horse']
print(sum_legs)

#Tuples
#Change the value in the list from ‘5’ to ‘3’
D = (1,15,4,[5,10])
L = list(D)
L[3] = [3,10]
D = tuple(L)

#Delete the tuple D
del D

#Print the number of occurences of ‘p’ in tuple E
E = ('a','p','p','l','e')
print(E.count('p'))

#Print the index of ‘l’ in tuple E
print(E.index('l'))


# In[13]:


#Task 2

#Convert matrix M into numpy array
M = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12]])

#Use slicing to pull out the subarray consisting of the first 2 rows and columns 1 and 2
#Store it in b which is a numpy array of shape (2, 2)
b = M[:2,:2]

#Create an empty matrix ‘y’ with the same shape as ‘M
y = np.empty((3,4))

#Add the vector z to each column of the matrix M with an explicit loop and store it in y
z = np.array([1, 0, 1])
for i in range (3):
    for j in range (4):
        y[i][j]=M[i][j]+z[i]

#Add the two matrices A and B
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
v = np.array([9,10])
add = np.add(A,B)

#Multiply the two matrices A and B
multiply = np.multiply(A,B)

#Take the element wise square root of matrix A
sqareroot = np.sqrt(A)

#Take the dot product of the matrix A and vector v
dotproduct = np.inner(A,v)

#Compute sum of each column of A
sumcolumn = np.sum(A,0)

#Print the transpose of B
print(np.transpose(B))


# In[7]:


#Task 3

#Functions
#Declare a function Compute that takes two arguments: distance and time, and use it to calculate velocity
def compute(distance,time):
    velocity = distance/time
    return velocity

#Forloops
#Declare a list even that contains all even numbers up till 16
#Declare a function sum that takes the list as an argument and calculates the sum of all entries using a for loop
even = [2, 4, 6, 8, 10, 12, 14, 16]
def sum (arr):
    csum = 0
    for i in range (np.size(arr)):
        csum = csum + arr[i]
    return csum


# In[10]:


#Task 4

#Plotting a single line
#Compute the x and y coordinates for points on a sine curve and plot the points using matplotlib
x_0 = 0
y_0 = math.sin(x_0)
x = []
x.append(x_0)
y = []
y.append(y_0)
for i in range (1,32):
    x.append(i*(2*np.pi)/32)
    y.append(math.sin(x[i]))
plt.plot(x,y)

#Use the function plt.show()
plt.show()

#Plotting multiple lines
#Compute the x and y coordinates for points on sine and cosine curves and plot them on the same graph using matplotlib
x_0 = 0
ysin_0 = math.sin(x_0)
ycos_0 = math.cos(x_0)
x = []
x.append(x_0)
ysin = []
ysin.append(ysin_0)
ycos = []
ycos.append(ycos_0)
for i in range (1,32):
    x.append(i*(2*np.pi)/32)
    ysin.append(math.sin(x[i]))
    ycos.append(math.cos(x[i]))
plt.plot(x, ysin)
plt.plot(x, ycos)

#Add x and y labels to the graph as well
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#Subplots
#Compute the x and y coordinates for points on sine and cosine curves
x_0 = 0
ysin_0 = math.sin(x_0)
ycos_0 = math.cos(x_0)
x = []
x.append(x_0)
ysin = []
ysin.append(ysin_0)
ycos = []
ycos.append(ycos_0)
for i in range (1,32):
    x.append(i*(2*np.pi)/32)
    ysin.append(math.sin(x[i]))
    ycos.append(math.cos(x[i]))
#Set up a subplot grid that has height 2 and width 1, and set the first such subplot as active
#Plot the sine and cosine graphs
plt.subplot(1, 2, 1)
plt.plot(x,ysin)
plt.subplot(1, 2, 2)
plt.plot(x,ycos)
plt.show()


# In[9]:


#Task 5

#Create a dataframe pd that contains 5 rows and 4 columns
data = {'Col1':[1, 2, 3, 4, 5], 
        'Col2':[6, 7, 5, 5, 8], 
        'Col3':[7, 78, 78, 18, 88], 
        'Col4':[7, 5, 707, 60, 4]} 
pdd = pd.DataFrame(data)

#Print only the first two rows of the dataframe
print(pdd.iloc[0:2,:])

#Print the second column
print(pdd.iloc[:,1])

#Change the name of the third column from “Col3” to “XYZ”
pd = pdd.rename({'Col3': 'XYZ'}, axis=1)

#Add a new column to the dataframe and name it “Sum”
Sum = pdd.sum(axis=1)

#Sum the entries of each row and add the result in the column “Sum”
pdd.insert(4, 'Sum', Sum)


# In[23]:


# Task 6

#Display your image using the plt.imshow function
img = np.array(mpimg.imread('IMG-20170505-WA0004.jpg'))
img.setflags(write=1)
plt.imshow(img)

#Crop the image
cropped_img = img[100:300, 100:300, :]

#Create 50 randomly placed markers on your image
coor_1 = np.random.randint(570, size = 50)
coor_2 = np.random.randint(570, size = 50)
marker = np.random.randint(0, len(Line2D.markers), 50)
for x,y in enumerate(Line2D.markers):
    i = (marker == x)
    plt.scatter(coor_1[i], coor_2[i], marker = 'x', c = 'pink')


# In[8]:


#Carry out color analysis by accessing the RGB values and plotting them
#Use the seaborn library
red = img[:,:,0]
green = img[:,:,1]
blue = img[:,:,2]
sns.distplot(red, hist = False, label = 'Red', color = 'Red')
sns.distplot(green, hist = False, label = 'Green', color = 'Green')
sns.distplot(blue, hist = False, label = 'Blue', color = 'Blue')
plt.show()

#View only the red values
img[:,:,1]=0
img[:,:,2]=0
plt.imshow(img)


# In[18]:


#Task 7

#Check if the camera is open
video = cv2.VideoCapture(0)
ret, frame = video.read()
if video.isOpened():
    print("Webcam online.")
video.release()
cv2.destroyAllWindows()

#Read the video from camera feed
#Write the video file into memory using the videoWriter function
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('ML_A1_Task7.avi', fourcc, 20.0, size)
while(True):
    _, frame = cap.read()
    cv2.imshow('Recording...', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('S'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




