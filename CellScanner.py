import cv2
import imutils
import numpy as np
from PIL import Image
import pytesseract
from Hexlassifier import predict_hex
from PrintClassifier import predict_print
from CellExtractor import get_cells
from PageExtractor import getPageWarped
# makes the character into a square by padding the contour


def make_square(char, p):
    s = max(char.shape)
    f = np.zeros((s, s), np.uint8)
    ax, ay = (s - char.shape[1])//2, (s - char.shape[0])//2
    f[ay:char.shape[0]+ay, ax:ax+char.shape[1]] = char
    image = cv2.copyMakeBorder(f, p, p, p, p, cv2.BORDER_CONSTANT)
    return image

def cell_processing(cell):
    #Finding contours (only need the external contour):
    cnts, _ = cv2.findContours(cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    characters = []
    for i, c in enumerate(cnts):
        x,y,w,h = cv2.boundingRect(c)
        character = cell[y:y+h, x:x+w]
        character = cv2.resize(make_square(character, 6), (28, 28))
        characters.append((character, x, w * h))
        cv2.rectangle(cell, (x, y), (x + w, y + h), (100,0,0), 2)
    if(len(characters) != 0):
        max_area = max(characters,key=lambda by_area:by_area[2])[2]          # the area of the larget contour
        characters = [c for c in characters if c[2] > 0.2 * max_area]        #make 0.2 bigger to be more restrictive.
        characters.sort(key=lambda by_x: by_x[1])
        characters = [c[0] for c in characters]

    return characters

# Reading and resizing the cell
table_with_background = cv2.imread('TestCases/4_3.jpeg')
table = getPageWarped(table_with_background)[5]
cells = get_cells(table)

# the following should be a loop.
rows = len(cells)
columns = len(cells[0])
excel = []

for i in range(0, rows):
    excel_row = []
    for j in range(0, columns):
        cell = cells[i][j]
        cell = imutils.resize(cell, width=500)
        characters = cell_processing(cell)

        #Writing the found contour into a folder
        #for i, char in enumerate(characters):
        #    cv2.imwrite(f'ScannedCharacters/Char{i}{j}.jpg', char )
        cv2.imwrite(f'ScannedCells/Cell{i}_{j}.jpg', cell)

        handwritten = False
        if(handwritten):
         cell_content = predict_hex(characters, saved=True) if characters else ''
        else:
         cell_content = predict_print(characters, saved=True) if characters else ''
        #Google = pytesseract.image_to_string(cell)
        #print(Google)

        excel_row.append(cell_content)
    excel.append(excel_row)

       
print(np.array(excel))