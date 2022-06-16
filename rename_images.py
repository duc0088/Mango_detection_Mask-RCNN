import numpy as np
import cv2
import math
import os
import xml.etree.ElementTree as ET
i=1
for filename in os.listdir("E:\\project_duc\\data_xoaiii"):
     filen=os.path.splitext(os.path.basename(filename))[0]
     print(filen)
     extension=os.path.splitext(os.path.basename(filename))[1]
     print(extension)
     if extension=='.jpg':
       img = cv2.imread("E:\\project_duc\\data_xoaiii\\"+filen + ".jpg")
       cv2.imwrite("E:\\project_duc\\data_xoaiii\\"+str(i) + ".jpg",img)
       filen1=(str(i))
       print(filen1)
       if filen!=filen1:
         os.remove("E:\\project_duc\\data_xoaiii\\"+filen + ".jpg")
       i+=1