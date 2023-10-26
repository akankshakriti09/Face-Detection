import cv2
import os #os library to create directroy and to check directory

#create folder dataset
datasets = "dataset"
#subfolder name
sub_data = "pic"

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path): #if directory not present
    os.mkdir(path)    #make this directory

(width, height) = (130, 100)
alg = "HaarCascadeFrontalFaceAlgo.xml" #initialize algo in variable
haar_cascade = cv2.CascadeClassifier(alg) #load algorithm inside computer vision
cam = cv2.VideoCapture(0) #device cam
#cam = cv2.VideoCapture(1) #external cam

count = 1
while count < 31:  #capture 30 img
    print(count)
    _,img = cam.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4)
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        faceOnly = grayImg[y:y+h, x:x+w]
        resizeImg = cv2.resize(faceOnly, (width,height))
        cv2.imwrite("%s/%s.jpg" %(path,count), faceOnly)   #%S%S for saving img as 1.jpg,2.jpg ... n.jpg
        count+=1
    cv2.imshow("FaceDetection", img)
    key = cv2.waitKey(1)
    if key == 27: # 27 is pressing an esc key
        break
print("Image Captured successfully")
cam.release()
cv2.destroyAllWindows()
 