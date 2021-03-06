import cv2
import imutils
import numpy as np
from PIL import Image
import pytesseract
from Hexlassifier import predict_hex
from PrintClassifier import predict_print
from CellExtractor import get_cells
from PageExtractor import getPageWarped
import xlwt
# makes the character into a square by padding the contour

def make_square(char, p):
    s = max(char.shape)
    f = np.zeros((s, s), np.uint8)
    ax, ay = (s - char.shape[1])//2, (s - char.shape[0])//2
    f[ay:char.shape[0]+ay, ax:ax+char.shape[1]] = char
    image = cv2.copyMakeBorder(f, p, p, p, p, cv2.BORDER_CONSTANT)
    return image

def cell_processing(cell):
    cell_area = len(cell) * len(cell[0])
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
        characters = [c for c in characters if c[2] > 0.2 * max_area and c[2] > 0.005 * cell_area]        #make 0.2 bigger to be more restrictive.
        characters.sort(key=lambda by_x: by_x[1])
        characters = [c[0] for c in characters]
    return characters

# the following should be a loop.
def get_excel(cells, cells_for_google,google=False):
    rows, columns = len(cells), len(cells[0])
    excel, excel_google = [], []
    for i in range(0, rows):
        excel_row, excel_google_row = [], []
        for j in range(0, columns):
            cell = cells[i][j]
            cell = imutils.resize(cell, width=500)
            characters = cell_processing(cell)
            cv2.imwrite(f'ScannedCells/Cell{i}_{j}.jpg', cell)
            handwritten = True if (j > 3 ) else False
            if(handwritten):
                cell_content = predict_hex(characters, saved=True) 
            else:
                cell_content = predict_print(characters, saved=True)
            #pytesseract.pytesseract.tesseract_cmd =r'C:\Users\mohamed saad\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
            cell_content_google = pytesseract.image_to_string(cells_for_google[i][j]) if google else ''
            excel_row.append(cell_content+'\n')
            excel_google_row.append(cell_content_google+'\n')
        excel_google.append(excel_google_row)
        excel.append(excel_row)

    #create excel file from the 2d array excel
    rows=len(excel)
    columns=len(excel[0])
    excel_file = xlwt.Workbook()
    sheet = excel_file.add_sheet('Ours')
    sheet_google = excel_file.add_sheet('Google')

    for i in range(0, rows):
        for j in range(0, columns):
            sheet.write(i, j, excel[i][j])
            sheet_google.write(i, j, excel_google[i][j])

    excel_file.save('table.xls')

# Reading and resizing the cell
#table_with_background = cv2.imread('./TestCases/Perfection.jpg')

def tableToExcel(filename):
    bad_img = True
    table_with_background = Image.open(filename)
    table_with_background = cv2.cvtColor(np.array(table_with_background), cv2.COLOR_RGB2BGR)
    cv2.imwrite('Table1.jpg', table_with_background)
    table_with_background = getPageWarped(table_with_background)[0][5] if bad_img else table_with_background
    table_with_background = cv2.rotate(table_with_background, -cv2.ROTATE_90_CLOCKWISE) if bad_img else table_with_background
    cv2.imwrite('Table.jpg', table_with_background)
    cells,cells_for_google = get_cells(table_with_background)
    get_excel(cells,cells_for_google,google=True)
    return 'Its a GGWP Scenario'
