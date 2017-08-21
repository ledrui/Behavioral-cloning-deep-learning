import cv2
import numpy as np

#Histogram Equalization
def eq_Hist(img):
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    return img

#Compute linear image transformation img*s+m
def lin_img(img,s=1.0,m=0.0):
    img2=cv2.multiply(img, np.array([s]))
    return cv2.add(img2, np.array([m]))

#Change image contrast; s>1 - increase
def contr_img(img, s=1.0):
    m=127.0*(1.0-s)
    return lin_img(img, s, m)

#Random brightness 
def bright_img(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img[:,:,2] = img[:,:,2]*(0.2+np.random.uniform())
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img

#Image normalization into [-0.5, 0.5]
def norm_img(img):
	img=np.asfarray(img)
	img[:, :, 0] = img[:, :, 0]/255.0-0.5
	img[:, :, 1] = img[:, :, 1]/255.0-0.5
	img[:, :, 2] = img[:, :, 2]/255.0-0.5
	return img

#Prepare augmented image
def augment_img(img_p, x_size, y_size, f):
	img=cv2.imread(img_p, cv2.IMREAD_COLOR)
	img = preproc_img(img)
	a = np.random.randint(30)-15
	b = np.random.randint(10)
	img = img[50+a-b:160-22-b,:,:] #Take bottom part of the image without the sky
	img = cv2.resize(img, (x_size, y_size), interpolation = cv2.INTER_CUBIC)
	img = bright_img(img)
	if f == 1: #flip vertically, if f==1
		img = cv2.flip(img,1)
	return img


def val_img(img_p, x_size, y_size):
	img=cv2.imread(img_p, cv2.IMREAD_COLOR)
	img=pred_img(img, x_size, y_size)
	return img

#Image process for validation or prediction
def pred_img(img, x_size, y_size):
	img = preproc_img(img)
	img = img[50:160-22,:,:] #Take bottom part of the image without the sky
	img = cv2.resize(img, (x_size, y_size), interpolation = cv2.INTER_CUBIC)
	return img

#Image preprocessor
def preproc_img(img):
	img = eq_Hist(img)
	img = cv2.GaussianBlur(img, (5,5), 0)
	img = contr_img(img, 1.2)
	return img
