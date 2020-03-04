import cv2;
import math;
import numpy as np;
# im : input image
# Nous devrons trouver la valeur de input image
def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark
# ATMLIGHT : Atmosphere light
def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz,1);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A
#medium transmission est en fonction de dark channel de input image
def TransmissionEstimate(im,A,sz):
    omega = 0.95;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission
# Apres obtenire la valeur maticielle de te ,A ont peut calculer output image
def Recover(im,te,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

if __name__ == '__main__':
    import sys
    

src = cv2.imread('/inputimage.png');

I = src.astype('float64')/255;
 
dark = DarkChannel(I,15);
A = AtmLight(I,dark);
te = TransmissionEstimate(I,A,15);
t = TransmissionRefine(src,te);
J = Recover(I,t,A,0.1);

cv2.imshow("dark",dark);
cv2.imshow("Medium transmission t",te);
cv2.imshow('Input Image I',src);
cv2.imshow('Enhanced Image J ',J);
cv2.imwrite("/outputimage.png",J*255);
cv2.waitKey();
    

