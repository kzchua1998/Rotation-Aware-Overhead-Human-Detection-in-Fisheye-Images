from api import Detector

# Initialize detector
detector = Detector(model_name='rapid',
                    weights_path='./weights/rapid_pL1_dark53_COCO608_Jun16_2000,ckpt',
                    use_cuda=False)

import cv2
import numpy as np
from PIL import Image
import pdb
# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
cap = cv2.VideoCapture(r'D:\visual_studio_code\yolov5_drone_test\yolov5\try.mp4')

# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# Read until video is completed
i = 0
while(cap.isOpened()):
  # Capture frame-by-frame
  i+=1
  ret, frame = cap.read()
  if ret == True:
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    scale_percent = 50 # of original image size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width,height)
    frame = cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)
    #pdb.set_trace()
    im_pil = Image.fromarray(frame)
    im = detector.detect_one(pil_img=im_pil, return_img=True, conf_thres=0.5, test_aug=None, input_size=1024)
    cv2.putText(im, str(i), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow("image",im)
    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
