import torch
from torchvision.transforms import *
from torchvision.models import resnet18,regnet_x_1_6gf
from torch import nn
import cv2
import glob
import pyperclip
import pyautogui
import win32clipboard
from TranslationServices import translate
import mouse
import time
import win32api
import numpy as np
import mediapipe as mp
from threading import Thread
from PIL import Image
import sys

sys.setrecursionlimit(64*55*100)


mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh


transform = Compose([
    Resize((64,55)),
    ToTensor()
])

testTransforms = Compose([
    Grayscale(),
    Resize((50,50)),
    ToTensor()
])

testTransformsEyeCsf = Compose([
    Grayscale(),
    Resize((100,200)),
    ToTensor()
])

def samePosition(preds,amount):
    if(amount>len(preds)):
        return False
    prev = preds[-1]
    for i in range(amount):
        if(preds[-(i+1)]==prev):
            continue
        else:
            return False
    return True

def scroll(amount=10,speed=10,sleepTime=0.1,directionDown=True):
    speed = speed if directionDown else -1*speed
    print("SCROLLING")
    for i in range(amount):
        mouse.wheel(int(speed))
        time.sleep(int(sleepTime))

def selectText(sentenceBased):
    clicks = 2 if sentenceBased else 3
    pyautogui.click(button="left",clicks=3,interval=0.25)
def getText():
    selectText(False)
    pyautogui.hotkey("ctrl","c")
    time.sleep(0.01)
    pyautogui.click(button="left")
    return pyperclip.paste()
def getEyeMapping(indexArr):
    mapping = {
        (0,0):'Closed',
        (1,1):'Open',
        (1,0):'Left eye closed',
        (0,1):'Right eye closed'
    }
    predictions = []
    for val in indexArr:
        predictions.append(val.item())
    movement = mapping[tuple(predictions)]
    return movement
def getEye(results,image):
    image = Image.fromarray(image)
    predictions = ""
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

                landmarks = face_landmarks.landmark
                rightEyeTopLeftX = landmarks[55].x*w
                rightEyeTopLeftY = landmarks[55].y*h
                
                rightEyeTopRightX = landmarks[46].x*w
                rightEyeTopRightY = landmarks[46].y*h
                
                rightEyeTopY = landmarks[188].y*h
                
                rightEye = image.crop((rightEyeTopRightX,rightEyeTopRightY,rightEyeTopLeftX,rightEyeTopY))
                
                
                leftEyeTopRightX = landmarks[285].x*w
                leftEyeTopRightY = landmarks[285].y*h
                
                leftEyeTopLeftX = landmarks[276].x*w
                leftEyeTopLeftY = landmarks[276].y*h
                
                leftEyeTopY = landmarks[412].y*h
                
                leftEye = image.crop((leftEyeTopRightX,leftEyeTopRightY,leftEyeTopLeftX,leftEyeTopY))
                
                eyes = torch.tensor([testTransformsEyeCsf(rightEye).tolist(),testTransformsEyeCsf(leftEye).tolist()])
                
                rightEyeOpen = eyeClassifier(eyes)
                preds = torch.round(rightEyeOpen)
                
                predictions = getEyeMapping(preds)
                
    return predictions

class ClassifierReg(nn.Module):
  def __init__(self):
    super(ClassifierReg,self).__init__()
    self.pretrained = regnet_x_1_6gf()
    self.pretrained.fc = nn.Linear(912,2)

  def forward(self,data):
    return self.pretrained(data)
  
  def setTrain(self):
    self.pretrained.fc.requires_grad = True
  def setEval(self):
    self.pretrained.fc.requires_grad = False

class ClassifierRes(nn.Module):
  def __init__(self):
    super(ClassifierRes,self).__init__()
    self.pretrained = resnet18()
    self.pretrained.fc = nn.Linear(512,2) # have to change to only 2 outputs


  def forward(self,data):
    return self.pretrained(data)
  def setTrain(self):
    self.pretrained.fc.requires_grad = True
  def setEval(self):
    self.pretrained.fc.requires_grad = False
  
class EyeClassifier(nn.Module):
  def __init__(self):
    super(EyeClassifier,self).__init__()

    self.network = nn.Sequential(
        nn.Conv2d(1,3,5,2),
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.Conv2d(3,6,5,2),
        nn.BatchNorm2d(6),
        nn.ReLU(),
        nn.Conv2d(6,9,5,2),
        nn.BatchNorm2d(9),
        nn.ReLU(),
        nn.Conv2d(9,12,5),
        nn.BatchNorm2d(12),
        nn.ReLU(),
        nn.Conv2d(12,18,5),
        nn.Flatten(),
        nn.Linear(252,1),
        nn.Sigmoid()
    )
  def forward(self,data):
    return self.network(data)



csf = ClassifierReg()
checkpoint = torch.load(r"C:\Users\16177\Documents\Machine Learning\EyeTracking\Reg22941.1622.pth")
csf.load_state_dict(checkpoint)
csf.eval()

eyeClassifier = EyeClassifier()
eyeClassifier.load_state_dict(torch.load(r"C:\Users\16177\Downloads\EyeOpenClassifierSmaller.pth",map_location=torch.device('cpu')))
eyeClassifier.eval()


count = 0

cap = cv2.VideoCapture(0)
prevPoint = np.array([-100,-100])
closedCount = 0
movement = True
shouldTranslate = False

target_ln = "en"


    
        
        
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
  with mp_face_mesh.FaceMesh(
      static_image_mode=True,
      max_num_faces=1,
      refine_landmarks=True,
      min_detection_confidence=0.5) as face_mesh:

      success,image = cap.read()      
      while success: 
          if(count%3==0):
              img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
              results = face_mesh.process(img)
              h, w, c = image.shape
              
              image = Image.fromarray(img)
              
              if results.multi_face_landmarks:
                  for face_landmarks in results.multi_face_landmarks:


                          eyePred = getEye(results,img)
                          
                          if(eyePred=="Left eye closed"):
                            closedCount = 0
                            shouldTranslate = True
                          elif(eyePred=="Closed"):
                            closedCount+=1
                            if(closedCount>10):
                              print("closed")
                              movement = not movement
                              closedCount = 0
                          elif(movement!=False):
                            closedCount = 0
                            landmarks = face_landmarks.landmark
                            rightEyeTopLeftX = landmarks[55].x*w
                            rightEyeTopLeftY = landmarks[55].y*h
                            
                            rightEyeTopRightX = landmarks[46].x*w
                            rightEyeTopRightY = landmarks[46].y*h
                            
                            rightEyeTopY = landmarks[188].y*h
                            
                            rightEye = image.crop((rightEyeTopRightX,rightEyeTopRightY,rightEyeTopLeftX,rightEyeTopY))
                            
                            
                            leftEyeTopRightX = landmarks[285].x*w
                            leftEyeTopRightY = landmarks[285].y*h
                            
                            leftEyeTopLeftX = landmarks[276].x*w
                            leftEyeTopLeftY = landmarks[276].y*h
                            
                            leftEyeTopY = landmarks[412].y*h
                            
                            leftEye = image.crop((leftEyeTopRightX,leftEyeTopRightY,leftEyeTopLeftX,leftEyeTopY))
                            

                            rightEye = rightEye.resize((55,32))
                            leftEye = leftEye.resize((55,32))

                            right = np.array(rightEye)
                            left = np.array(leftEye)
                            stacked = np.concatenate([left,right],axis=0)
                            stacked = Image.fromarray(stacked)
                            
                            image = transform(stacked)
                            point = csf(image.unsqueeze(dim=0)).detach().numpy()
                            
                            if(shouldTranslate):
                                text = getText()
                                if(text!=""):
                                  translated = translate(text,target_ln)
                                  print("text has been written")
                                  with open(r"C:\Users\16177\Documents\HackTheCastle\Communication.txt","w") as f:
                                    f.write(translated)
                                shouldTranslate = False
                                
                            if((point-prevPoint).sum()>70 and (point-prevPoint).sum()<500):
                              pyautogui.moveTo(*point.tolist()[0],0.5)
                            prevPoint = point
          success,image = cap.read()
          count += 1