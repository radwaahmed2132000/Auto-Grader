import os
import cv2
import imutils
from CellScanner import make_square

for i, filename in enumerate(os.listdir("./training_data/9/")):
    if filename.endswith(".png"): 
         img_path = os.path.join("./training_data/9/", filename)
         digit = cv2.imread(img_path)
         digit = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
         T, digit= cv2.threshold(digit, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
         digit = cv2.resize(make_square(digit), (28, 28))
         cv2.imwrite(f'new_dataset/nine/nine_{i}.jpg', digit ) 
