from __future__ import division
import cv2
import time
import sys
import imutils
import dlib
import cv2
import glob
import numpy as np
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from imutils import face_utils
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt
from sklearn.preprocessing import minmax_scale

ForeHeadXDim = 200 # length of forehead bounding box
ForeHeadYDim = 75 # hight of forehead bounding box

dim = (ForeHeadXDim,ForeHeadYDim) #forehead image and hist dimensions

y_hist_range = 15000
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
configFile = "deploy.prototxt"
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
conf_threshold = 0.7
source = 0
indexToDelete = 0
print("[INFO] sampling THREADED frames from webcam...")
DelayTimer = 0
thresh = 111 
FramesToDelay = 3
FrameCounter = 0
FrameCounterLimit = 10
# nasolabial fold right indices [3,4,28,49]
select_indexRightFold = [3,4,28]
# nasolabial fold left indices [13,14,28,53]
select_indexLeftFold = [13,14,28]
# forehead
forehead_index = [0]
right_eye_corner_index = [39]
left_eye_corner_index = [42]

indcnt=-1
IntensityTreshhold = 10
deltaIntensity = 0
ForeheadStackFramesValue = np.zeros((FrameCounterLimit,ForeHeadXDim)) # Default Baseline calculations array

def detectFaceOpenCVDnn(net, frame, gray):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    showFolds = False
    roiRight = np.empty(1)
    roiL = np.empty(1)
    roiForehead = np.empty(1)
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            indcnt=-1
            showFolds = True
            # top and bottom right coordinates of a face
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            
            faceWidth = (abs(x2 - x1))
            faceHight = (abs(y2 - y1))        

            # transform coordinates of dnn box to dlib rectanlge for landmark predictor
            rectFace = dlib.rectangle(x1, y1, x1+faceWidth, y1+faceHight) 
            
            shapeX = predictor(gray, rectFace)
            shape = face_utils.shape_to_np(shapeX)  
            
            ##### FOLDS DETECTION #####

            # draw indecies numbers
            '''
            for (x, y) in shape:
                indcnt=indcnt+1 # count and display indexes of shapes
                cv2.circle(frameOpencvDnn, (x, y), 1, (0, 0, 255), -1)
                #cv2.putText(frameOpencvDnn, (str(indcnt)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            '''
            # get desired points from the main shape
            filtered_indices = np.take(shape, select_indexRightFold, axis=0) 
       
            # extract the ROI of the right fold
            (x, y, w, h) = cv2.boundingRect(np.array([filtered_indices[:]]))

            # Display regions of interest - right nasolabial fold
            roiRight = gray[y:y + h, x:x + w]
            roiRight = imutils.resize(roiRight, width=256, inter=cv2.INTER_LINEAR)            
            
            cv2.rectangle(frameOpencvDnn,(x,y),(x+w,y+h),(0,255,0),1)
         
            # get bottom line forehead coordinates            
            forehead_filtered_indices = np.take(shape, forehead_index, axis=0) 
            (x1_forehead, y1_forehead, x2_forehead, y2_forehead) = cv2.boundingRect(np.array([forehead_filtered_indices[:]]))
            
            forehead_hight = int(y1_forehead - y1)
            forehead_width = int(x2 - x1)
            
            # box around forehead
            cv2.rectangle(frameOpencvDnn,(x1,y1),(x2,y1_forehead),(255,0,0),1)

            # Display regions of interest
            roiForehead = gray[y1:y1 + forehead_hight, x1:x1 + forehead_width]            
            roiForehead = imutils.resize(roiForehead, width=256, inter=cv2.INTER_LINEAR)            

            # get nasolabial left points from the main shape
            filtered_indicesLeftFold = np.take(shape, select_indexLeftFold, axis=0) 
       
            # extract the ROI of the face region as a separate image
            (x_LeftFold, y_LeftFold, w_LeftFold, h_LeftFold) = cv2.boundingRect(np.array([filtered_indicesLeftFold[:]]))
            
            # Display regions of interest
            roiL = gray[y_LeftFold:y_LeftFold + h_LeftFold, x_LeftFold:x_LeftFold + w_LeftFold]            
            roiL = imutils.resize(roiL, width=256, inter=cv2.INTER_LINEAR)            
            cv2.rectangle(frameOpencvDnn,(x_LeftFold,y_LeftFold),(x_LeftFold+w_LeftFold,y_LeftFold+h_LeftFold),(0,255,0),1)
            
            ### TOP OF THE NOSE ###

            # get left eye corner coordinates            
            left_eye_corner_indices = np.take(shape, left_eye_corner_index, axis=0) 
            (x1_left_eye_corner, y1_left_eye_corner, x2_left_eye_corner, y2_left_eye_corner) = cv2.boundingRect(np.array([left_eye_corner_indices[:]]))

            # get right eye corner coordinates            
            right_eye_corner_indices = np.take(shape, right_eye_corner_index, axis=0) 
            (x1_right_eye_corner, y1_right_eye_corner, x2_right_eye_corner, y2_right_eye_corner) = cv2.boundingRect(np.array([right_eye_corner_indices[:]]))

            # get distance between eyes (region width)
            eyes_w = x1_right_eye_corner - x1_left_eye_corner
            # top of the nose region hight
            eyes_h = int(forehead_hight / 3)
            # get coordinates of the top of this region
            nose_top_x1 = x1_right_eye_corner
            nose_top_x2 = x1_left_eye_corner
            nose_top_y1 = y1_right_eye_corner - eyes_h
            nose_top_y2 = y1_left_eye_corner - eyes_h

            cv2.rectangle(frameOpencvDnn,(nose_top_x2,nose_top_y2),(nose_top_x2+eyes_w,nose_top_y2+eyes_h),(0,255,0),1)
            
            ### END FOLDS DETECTION ###

        else:
            pass
    return frameOpencvDnn, roiRight, roiL, roiForehead, showFolds 

def PosterizeMe (imToPost):

    n = 4    # Number of levels of quantization

    indices = np.arange(0,256)   # List of all colors 

    divider = np.linspace(0,255,n+1)[1] # we get a divider

    quantiz = np.int0(np.linspace(0,255,n)) # we get quantization colors

    color_levels = np.clip(np.int0(indices/divider),0,n-1) # color levels 0,1,2..

    palette = quantiz[color_levels] # Creating the palette

    im2 = palette[imToPost]  # Applying palette on image

    imPosterized = cv2.convertScaleAbs(im2) # Converting image back to uint8

    return imPosterized


def TrailBaseLine (ForeheadBlurredResizedx, ForeheadStackFramesValue): # Function takes array of intensities for N frames and current frame, outputs mean, median, max and min values for a time period
    
    a_del = np.delete(ForeheadStackFramesValue, 0, 0) # Dropping old frame values
    
    stacked = np.vstack((a_del,ForeheadBlurredResizedx)) # Adding most recent
    
    BaselineMax = np.amax(stacked, axis=0)
    BaselineMin = np.amin(stacked, axis=0)
    BaselineMean = np.mean(stacked, axis=0)
    BaselineMedian = np.median(stacked, axis=0)
    ForeheadStackFramesValue = stacked
    
    return ForeheadStackFramesValue, BaselineMax, BaselineMin, BaselineMean, BaselineMedian

if __name__ == "__main__" :
    
    #Defining histogram parameters:
    fig, ax = plt.subplots()
    ax.axis([0, dim[0], 0, y_hist_range])
    ax.set_title('Custom pixel intensities histogram')
    ax.set_xlabel('Image X pixels')
    ax.set_ylabel('Intensity')
    lw = 3
    alpha = 0.5
    #Default values for the histogram
    ForeheadBlurredResizedx, BaseLinePlot, BaselineMax, BaselineMin, BaselineMedian, BaselineMean = np.zeros(dim[0]), np.zeros(dim[0]), np.zeros(dim[0]), np.zeros(dim[0]), np.zeros(dim[0]), np.zeros(dim[0])
    x = [x for x in range(dim[0])]
    ax.plot(x, ForeheadBlurredResizedx)
    lineGray, = ax.plot(x, ForeheadBlurredResizedx) #current state of the signal
    lineBlue, = ax.plot(x, BaselineMax)
    lineRed, = ax.plot(x, BaselineMin)
    lineBlack, = ax.plot(x, BaselineMean)
    linePink, = ax.plot(x, BaselineMedian)
    
    ax.legend()
    #End of histogram parameters


    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    
    plt.ion()
    
    plt.show()
     
    if len(sys.argv) > 1:
        source = sys.argv[1]
 
    while(1):
        frame = vs.read()
        
        # convert to gray frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        #ForeheadIntensityArray = np.array([0])
        frame = imutils.resize(frame, width=600)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = clahe.apply(gray)
        
        # send frame, get forehead and folds regions, as well as detection status
        outOpencvDnn, roiret, roiLret, roiForeheadret, showFoldsRet = detectFaceOpenCVDnn (net, frame, gray)
        
        if showFoldsRet == True:
            if FrameCounter >= FrameCounterLimit:
                FrameCounter = 0
            else:
                pass
             
            # Forehead #########################
            roiForeheadret = PosterizeMe(roiForeheadret)
            ForeheadBlurred = cv2.GaussianBlur(roiForeheadret, (11, 11), 0)
            ForeheadBlurredResized = cv2.resize(ForeheadBlurred, dim, interpolation = cv2.INTER_LINEAR)
            roiretBlurred = cv2.GaussianBlur(roiret, (11, 11), 0)
            roiretLBlurred = cv2.GaussianBlur(roiLret, (11, 11), 0)
            ForeheadBlurredResized = cv2.bitwise_not(ForeheadBlurredResized)
            
            
            ForeheadBlurredResizedx = ForeheadBlurredResized.sum(axis=0) #sum of intensity values per collumn (actual state of the histogram)
            
            # invert grayscale                        
            lineGray.set_ydata(ForeheadBlurredResizedx) #current forehead intensity hist state
            cv2.putText(outOpencvDnn, (str(FrameCounter)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            FrameCounter += 1
            
            
            ForeheadStackFramesValue, BaselineMax, BaselineMin, BaselineMean, BaselineMedian = TrailBaseLine(ForeheadBlurredResizedx, ForeheadStackFramesValue)       
            
            lineBlack.set_ydata(BaselineMean)
            lineBlue.set_ydata(BaselineMax)
            lineRed.set_ydata(BaselineMin)
            linePink.set_ydata(BaselineMedian)
            
            '''
            if FrameCounter == (FrameCounterLimit - 1):
                lineBlue.set_ydata(ForeheadBlurredResizedx) #previous forehead intensity hist state
            else:
                pass
            '''
        else:
            pass  
        fig.canvas.draw()
        cv2.imshow("Folds", outOpencvDnn)
        cv2.imshow("Forehead", ForeheadBlurredResized)
        
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        fps.update()
    
    fps.stop()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows() 
    vs.stop
