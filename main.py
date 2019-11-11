# -*- coding: utf-8 -*-


import cv2
from ImageProcessing import ImageProcess



def putTextOnFrame(frame,carforwardCount,bikeforwardCount,carreverseCount,bikereverseCount ):
     text1 = "Cars in forward direction : " + str(carforwardCount)
     text2 = "Bikes in forward direction: " + str(bikeforwardCount)
     text3 = "Cars in reverse direction : " + str(carreverseCount)
     text4 = "Bikes in reverse direction: " + str(bikereverseCount)
     cv2.putText(frame,text1, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
     cv2.putText(frame,text2, (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
     cv2.putText(frame,text3, (10,110), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)
     cv2.putText(frame,text4, (10,140), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2)


def drawBoundingBoxesOnFrame(frame,frame_height,frame_width, matches):
    for (i,match) in enumerate(matches):
            contour, centroid = match
            #print(extent)
            x, y, w, h = contour
            cv2.rectangle(frame, (x, y), (x + w - 1, y + h - 1), (255,0,0), 1)

    cv2.line(frame, (0,int(frame_height/2)-10), (int(frame_width/2)+50,int(frame_height/2)-10), (0,0,255), 2)
    cv2.line(frame, (int(frame_width/2)+100,int(frame_height/2)+50), (int(frame_width),int(frame_height/2)+50), (0,255,0), 2)

def getVideoDetails(cap):
      frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
      frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
      frameRate  = cap.get(cv2.CAP_PROP_FPS)
      frameSize = (int(frame_width),int(frame_height))
      return frame_width,frame_height,frameRate,frameSize

def main():
    
    #counters for cars and vehicles in forward and reverse direction
    carforwardCount = 0
    carreverseCount = 0
    bikeforwardCount = 0
    bikereverseCount = 0
    
    #read input video
    videoPath = './Input/video.mp4'
    print(cv2.__version__)
    cap = cv2.VideoCapture(videoPath)   #read video file
    frame_width, frame_height,frameRate, frameSize = getVideoDetails(cap)
  
    
    
    #create object to write output video
    outputVideoPath = './Output/output.mp4'
    out = cv2.VideoWriter(outputVideoPath,cv2.VideoWriter_fourcc(*'DIVX'), int(frameRate), frameSize)

    
    
    #create background subtractor
    backgroundSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG(history=1, nmixtures=0, backgroundRatio=0.0001)
    
   
 
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        
        #process the frames to detect the cars and vehicles in image
        imgProcess = ImageProcess(frame, frame_height, frame_width, backgroundSubtractor)
        matches, count= imgProcess.process()
        
        #update car and bike counts
        carforwardCount += count[0]
        bikeforwardCount += count[1]
        carreverseCount += count[2]
        bikereverseCount += count[3]
        
        drawBoundingBoxesOnFrame(frame,int(frame_height),int(frame_width),matches)
        
        putTextOnFrame(frame,carforwardCount,bikeforwardCount,carreverseCount,bikereverseCount)
       
        cv2.imshow("Image", frame)
        out.write(frame)
        keyboard = cv2.waitKey(30)
        if keyboard == 'q' or keyboard == 27:
            break

    cv2.destroyAllWindows() 
    out.release()

if __name__ == "__main__":
    main()