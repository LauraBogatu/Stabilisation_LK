import numpy as np
import matplotlib.pyplot as plt

import cv2
%matplotlib inline

def show_gray(img, title, **kwargs):
    plt.figure()
    plt.imshow(img, cmap="gray",  **kwargs)
    plt.axis("off")
    plt.title(title)
    
def apply_color(img, fn):
    return np.dstack((fn(img[:,:,0]),fn(img[:,:,1]),fn(img[:,:,2])))

def load_and_normalise(fname):
    return skimage.io.imread(fname) / 256.0

def load_gray(f):
    return skimage.color.rgb2gray(load_and_normalise(f))


def load_gray_int(f):
    return (skimage.color.rgb2gray(load_and_normalise(f)) * 256.0).astype(np.uint8)




print cv2.__version__

cap = cv2.VideoCapture(0)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 40,
                       qualityLevel = 0.1,
                       minDistance = 9,
                       blockSize = 9 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (25,25),
                  maxLevel = 8,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
#color = np.random.randint(0,255,(100,3))
#print color

# Take first frame and find corners in it
ret, old_frame = cap.read()
preview_size = (450,450)
old_frame = cv2.resize(old_frame, preview_size, interpolation=cv2.INTER_AREA)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

off_x = 0
off_y = 0
homography = False
hom_M = np.eye(3)

side_by_side = np.hstack((np.zeros_like(old_frame),np.zeros_like(old_frame)))
while(1):
    ret,frame = cap.read()
    preview_size = (450,450)
    frame = cv2.resize(frame, preview_size, interpolation=cv2.INTER_AREA)
    if frame is None:
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,0)
        ret,frame = cap.read()
            
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    offset = p0 - p1
   
    offset = np.squeeze(offset)
    med_offset = np.median(offset, axis=0)    
    
    if True :
        if(off_x + med_offset[0]>30):     
            off_x = +30
            
        elif(off_x + med_offset[0]<-30):
            off_x = -30
        else:    
            off_x += med_offset[0]
        if(off_y + med_offset[1]>30):
            off_y = +30
        elif(off_y + med_offset[1]<-30):
            off_y = -30
        else:    
            off_y += med_offset[1]
        #to add try catch  
        hom = cv2.findHomography(p0,p1)
    
        hom_M =np.dot(hom_M,hom[0])
    
        # Select good points
        good_new = p1[st==1]
       
        good_old = p0[st==1]

    
        img = cv2.add(frame,mask)
        # gradually relax offsets (so we can still pan around slowly)
        off_x *= 0.95
        off_y *= 0.95
        M = np.float32([[1,0,off_x],[0,1,off_y]])    
        hom_M = 0.99*hom_M + 0.01 *np.eye(3)
    
    
        if homography:
            aligned = cv2.warpPerspective(src=img, M=hom_M,dsize=(img.shape[1],img.shape[0]), flags=4|cv2.WARP_INVERSE_MAP)
        else:
            aligned = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]), flags=4)
    
    cv2.line(aligned,(170,0),(170,img.shape[0]), (30,185,30),2)
    cv2.line(aligned,(0,140),(img.shape[1],140), (30,185,30),2)
    
    cv2.line(img,(170,0),(170,img.shape[0]), (30,185,30),2)
    cv2.line(img,(0,140),(img.shape[1],140), (30,185,30),2)
    
    side_by_side[:img.shape[0],img.shape[1]:,:] = aligned
    side_by_side[:img.shape[0],:img.shape[1],:] = img
    #old_frame[img.shape[1],img.shape[0],:] = img
    cv2.imshow('frame',side_by_side)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    if(good_new.shape[0]<4):
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
    else:
        p0 = good_new.reshape(-1,1,2)
    
cv2.destroyAllWindows()
cap.release()