#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
from numpy.linalg import eig
from numpy.linalg import svd
import matplotlib.pyplot as plt
import imageio.v3 as imageio
from PIL import Image
import time
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# 1.2
# A : square random matrix with size n
n = 3
A = np.random.rand(n, n)
def QR_eignvalue_eignvector(A, iterations=5000):
    Ak = np.copy(A)
    n = A.shape[0]
    QQ = np.eye(n)
    for k in range(iterations):
        Q, R = np.linalg.qr(Ak)
        Ak = np.dot(R,Q)
        QQ = np.dot(QQ,Q)
    return Ak, QQ
w,v=eig(QR_eignvalue_eignvector(A))
print('E-value:', (w))
print('E-vector', (v))


# In[3]:


#2.1
def svd_compress(black_and_white_image , k ):
    A = np.array(black_and_white_image)
    # Perform SVD on the image matrix
    U, s, V = np.linalg.svd(A)
    # Construct a compressed image using only the k largest singular values
    Ak = U[:, :k] @ np.diag(s[:k]) @ V[:k, :]
    return Ak

def psnr(original, compressed ):
    # Calculate the maximum possible pixel value
    max_pixel_value = np.max(original)
    # Calculate the MSE between the original and compressed images
    mse = np.mean((original - compressed) ** 2)
    # Calculate the PSNR value       
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    return psnr

color_image = Image.open("q2_pic.bmp")
# Convert the image to black and white
black_and_white_image = color_image.convert("L")

psnra = []  
k = range(30,70,10)
for i in k:
    img_array = svd_compress(black_and_white_image, i)
    # Calculate the PSNR value between the original and compressed images
    psnra.append(psnr(black_and_white_image,img_array))

print(psnra)
plt.plot(k,psnra)
plt.show()

compressed_image = Image.fromarray(svd_compress(black_and_white_image, 40))
plt.figure()
plt.imshow(compressed_image, cmap='gray')
plt.title('A Sample with k = 40(psnr = 30.53)');

compressed_image = Image.fromarray(svd_compress(black_and_white_image, 60))
plt.figure()
plt.imshow(compressed_image, cmap='gray')
plt.title('A Sample with k = 60(psnr = 31.68)');


# In[4]:


#2.2  


# In[5]:



# Generate Gaussian noise
img_arr = np.array(black_and_white_image)
mean = 0
variance = 50
sigma = np.sqrt(variance)
gaussian_noise = np.random.normal(mean, sigma, img_arr.shape)

# Add Gaussian noise to the image
noisy_img_arr = img_arr + gaussian_noise.astype(np.uint8)

# Convert the noisy image array back to PIL Image
noisy_img = Image.fromarray(noisy_img_arr)

# Display the noisy images
plt.figure()
plt.imshow(noisy_img, cmap='gray')
plt.title('Noisy Image White Gauusian Noise used');
print(psnr(black_and_white_image,noisy_img_arr))

psnrb = []  
k = range(30,70,10)
for i in k:
    img_array = svd_compress(noisy_img_arr, i)
    # Calculate the PSNR value between the original and compressed images
    psnrb.append(psnr(noisy_img_arr,img_array))


# In[6]:


#plot for different psnr
print(psnrb)
plt.plot(k,psnrb)
plt.show()
plt.show()


# In[7]:


# A Sample for gauusian noise noise when svd Applied(psnr=24.6)
compressed_image = Image.fromarray(svd_compress(noisy_img_arr, 40))
plt.figure()
plt.imshow(compressed_image, cmap='gray')
plt.title('A Sample with k = 40(psnr= 24.60)');

compressed_image = Image.fromarray(svd_compress(noisy_img_arr, 60))
plt.figure()
plt.imshow(compressed_image, cmap='gray')
plt.title('A Sample with k = 60(psnr= 25.08)');


# In[8]:


# Add salt and pepper noise
img_array = np.array(black_and_white_image)
noise = np.random.randint(0, 100, img_array.shape)
salt = noise > 90
pepper = noise < 10
img_array[salt] = 255
img_array[pepper] = 0
noisy_img_salt_and_pepper = Image.fromarray(img_array)
plt.figure()
plt.imshow(noisy_img_salt_and_pepper, cmap='gray')
plt.title('Noisy Image Salt and Pepper used');
print(psnr(black_and_white_image,img_array))

psnrc = []  
k = range(30,70,10)
for i in k:
    img_array = svd_compress(noisy_img_salt_and_pepper, i)
    # Calculate the PSNR value between the original and compressed images
    psnrc.append(psnr(noisy_img_salt_and_pepper,img_array))


# In[9]:


#plot for different psnr
print(psnrc)
plt.plot(k,psnrc)
plt.show()
plt.show()


# In[10]:


# A Sample for salt and pepper noise when svd Applied(psnr = 12.78)
compressed_image = Image.fromarray(svd_compress(img_array, 40))
plt.figure()
plt.imshow(compressed_image, cmap='gray')
plt.title('A Sample with k = 40(psnr = 12.79)');

compressed_image = Image.fromarray(svd_compress(img_array, 60))
plt.figure()
plt.imshow(compressed_image, cmap='gray')
plt.title('A Sample with k = 60(psnr = 12.94)');


# In[13]:


#3.5
# load the data from csv file
df = pd.read_csv('iris.csv')

X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# calculate the covariance matrix
covariance_matrix = np.cov(X.T)

# calculate the eigenvectors and eigenvalues of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# sort the eigenvectors in descending order of their corresponding eigenvalues
sorted_indexes = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_indexes]

# select the first two eigenvectors to form the transformation matrix
transformation_matrix = sorted_eigenvectors[:, :2]

# transform the data into the new space
X_transformed = X.dot(transformation_matrix)

# plot the transformed data
fc = df['type'].tolist()
type1 = X_transformed[0:50,:]
plt.scatter(type1[:, 0], type1[:, 1],color = 'green')
type2 = X_transformed[51:100,:]
plt.scatter(type2[:, 0], type2[:, 1],color = 'blue')
type3 = X_transformed[101:150,:]
plt.scatter(type3[:, 0], type3[:, 1],color = 'red')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()


# In[ ]:





# In[ ]:




