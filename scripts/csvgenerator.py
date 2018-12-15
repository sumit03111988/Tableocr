import sys
import os
import PyPDF2
import re
import cv2
import imutils
from pdf2image import convert_from_path
import numpy as np
import camelot as cm
import pandas as pd
from PyPDF2 import PdfFileReader,PdfFileWriter
import fnmatch
import logging
import logging.config
from datetime import datetime

#def mergeFiles(self,output_dir,filename):
def mergeFiles(input_directory_Path,output_dir, filename,logger):
    logger.info("Merging the multiple tables in one")
    output_final_temp = input_directory_Path[:input_directory_Path.rindex('/')]
    output_final_dir = output_final_temp[:output_final_temp.rindex('/')]
    print(filename)
    finalOutputCSV = output_final_dir + "/FinalCSVs/"
    if not os.path.exists(finalOutputCSV):
        os.makedirs(finalOutputCSV)
    finalCSVName = os.path.splitext(filename)[0]+'.csv'
    finalCSV = open(finalOutputCSV+finalCSVName,"ab")
    line = 1
    for eachFile in os.listdir(output_dir):
        if fnmatch.fnmatch(eachFile,'*-page*'):
            fin = open((output_dir+'/'+eachFile),"rb")
            data1 = fin.read()
            finalCSV.write(b"\n")
            finalCSV.write(b"\n")
            finalCSV.write(data1)
            os.remove(output_dir+'/'+eachFile)
        line = line+1
    finalCSV.close()
    logger.info("Merging is completed")
    

#def findIndex(self,mappingDF,columnNames,value,flag):
def findIndex(mappingDF, columnNames, value, flag,logger,mappingDF_Yaxis,mappingDF_Xaxis):

    logger.info("Mapping the image points and pdf points")
    index_found = 'N'
    while index_found == 'N':
        list_empty = mappingDF.index[mappingDF[columnNames[0]] == value].tolist()
        if  list_empty:
            final_index = list_empty[0]
            index_found = 'Y'
            break
        else:
            if flag == 'Y':
                value = value -1
            elif flag =='X':
                value = value +1
    #self.logger.info("Mapping of points of image and pdf is done")
    return final_index

#def convertToCV(self,pdfFilePath,xmin,xmax,ymin,ymax,output_dir,y,filename):
def convertToCV(pdfFilePath, xmin, xmax, ymin, ymax, output_dir, y, filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img):

    logger.info("Applying camelot to save pdf to csv format")
    if ymin < 51:
        ymin = 51
    if ymax>1121:
        ymax=1121
    if xmin < 98:
        xmin = 98
    if xmax > 1050:
        xmax = 1050

    if flag_img == 'Y':
        if ymin < 133:
            ymin =133
        if ymax > 1888:
            ymax = 1888
        if xmin < 156:
            xmin = 156
        if xmax > 2293:
            xmax = 2293

    columnNames = list(mappingDF_Yaxis.head(0))
    index_ymin = findIndex(mappingDF_Yaxis,columnNames,ymin,'Y',logger,mappingDF_Yaxis,mappingDF_Xaxis)
    ymin_new = mappingDF_Yaxis.iloc[index_ymin][1]
    index_ymax = findIndex(mappingDF_Yaxis,columnNames,ymax,'Y',logger,mappingDF_Yaxis,mappingDF_Xaxis)
    ymax_new = mappingDF_Yaxis.iloc[index_ymax][1]

    index_xmin = findIndex(mappingDF_Xaxis, columnNames, xmin,'X',logger,mappingDF_Yaxis,mappingDF_Xaxis)
    xmin_new = mappingDF_Xaxis.iloc[index_xmin][1]

    index_xmax = findIndex(mappingDF_Xaxis, columnNames, xmax,'X',logger,mappingDF_Yaxis,mappingDF_Xaxis)
    xmax_new = mappingDF_Xaxis.iloc[index_xmax][1]
    table_areas_string = str(xmin_new)+','+str(ymin_new)+','+str(xmax_new)+','+str(ymax_new)
    print(table_areas_string)
    #tables = cm.read_pdf(pdfFilePath,flavor='stream', pages='1', table_areas=[table_areas_string])
    tables = cm.read_pdf(pdfFilePath,flavor='stream', pages='1', table_areas=['33.82,802,540,200'])
    output_file_name = output_dir+str(y)+'.csv'
    tables.export(output_file_name, f='csv')
    logger.info("PDF is converted to csv successfully")

#def fetchTable(self,pdfFilePath,pdfPageImage,pdfPageImage_Mask,xTopLeft,yTopLeft,xBottomRight,yBottomRight,intermediate_Image_Path,i,output_dir,y,filename):
def fetchTable(pdfFilePath, pdfPageImage, pdfPageImage_Mask, xTopLeft, yTopLeft, xBottomRight, yBottomRight,
                   intermediate_Image_Path, i, output_dir, y, filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img):

    logger.info("Fetching the table from in PDF")
    pdfPageImage_table = cv2.rectangle(pdfPageImage, (xTopLeft, yTopLeft-60), (xBottomRight, yBottomRight), (0, 0, 0), 1)
    print(xTopLeft, xBottomRight, yTopLeft, yBottomRight)
    pdfPageImage_crop = pdfPageImage_table[yTopLeft-60:yBottomRight, xTopLeft:xBottomRight]
    #self.logger.info("Table coordinates fetched")
    convertToCV(pdfFilePath,xTopLeft, xBottomRight, yBottomRight, yTopLeft,output_dir,y,filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img)

#def extractTableCoordinates_streamWithoutTable(self,pdfFilePath,pdfPageImage,pdfPageImage_Mask,contours,PDF_Page_Tables,intermediate_Image_Path,output_dir,y,filename):
def extractTableCoordinates_streamWithoutTable(pdfFilePath, pdfPageImage, pdfPageImage_Mask, contours,
                                                   PDF_Page_Tables, intermediate_Image_Path, output_dir, y, filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img):

    logger.info("Extracting table coordinates")
    yPosPrevious = 0
    xBottomRight = []
    yBottomRight = []
    contour_rows = len(contours)
    flag_table = 'N'
    for i in range(0, contour_rows):
        rect = cv2.minAreaRect(contours[i])
        xPos_of_contour= (rect[0][0])
        yPos_of_contour= (rect[0][1])
        width_of_contour=(rect[1][0])
        height_of_contour=(rect[1][1])
        angle_of_contour=(rect[2])
        xArr = []
        yArr = []
        contour_columns = len(contours[i])
        yPosCurrent = contours[i][contour_columns - 1][0][1]
        if flag_table == 'N':
            xBottomRight = contours[i][contour_columns - 1][0][0]
            yBottomRight = contours[i][contour_columns - 1][0][1]
        for j in range(0, contour_columns):
            xArr.append((contours[i][j][0][0]))
            yArr.append((contours[i][j][0][1]))

        xTopLeft = ((contours[i][0][0][0]))
        yTopLeft = ((contours[i][0])[0][1])
        if ((yPosPrevious - yPosCurrent > 80 or i == contour_rows - 1)):
            xTopLeft = ((contours[i-1][0][0][0]))
            yTopLeft = ((contours[i-1][0])[0][1])
            if (((xBottomRight - xTopLeft) >= 200) and ((yBottomRight-yTopLeft) >= 200)):
                fetchTable(pdfFilePath,pdfPageImage,pdfPageImage_Mask,xTopLeft,yTopLeft,xBottomRight,yBottomRight,intermediate_Image_Path,i,output_dir,y,filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img)
                flag_table = 'N'
                yPosPrevious = yPosCurrent
        else:
            xTopLeft= ((contours[i][0][0][0]))
            yTopLeft= ((contours[i][0])[0][1])
            yPosPrevious = yPosCurrent
            flag_table = 'Y'

#def extractTableCoordinates_stream(self,pdfFilePath,pdfPageImage,pdfPageImage_Mask,contours,PDF_Page_Tables,intermediate_Image_Path,output_dir,y,filename):
def extractTableCoordinates_stream(pdfFilePath, pdfPageImage, pdfPageImage_Mask, contours, PDF_Page_Tables,
                                       intermediate_Image_Path, output_dir, y, filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img):

    logger.info("Extracting table coordinates")
    xTopLeft = []
    yTopLeft = []
    xBottomRight = []
    yBottomRight = []
    contour_rows = len(contours)
    for i in range(0,contour_rows):
        contour_columns= len(contours[i])
        for j in range(0,contour_columns):
            xTopLeft.append((contours[i][j][0][0]))
            xBottomRight.append((contours[i][contour_columns-1][0][0]))
            yTopLeft.append((contours[i][j])[0][1])
            yBottomRight.append(contours[i][contour_columns-1][0][1])
    xmin = np.min(xTopLeft,axis=0)
    xmax = np.max(xBottomRight, axis=0)
    ymin = np.min(yTopLeft, axis=0)
    ymax = np.max(yBottomRight, axis=0)
    pdfPageImage_table = cv2.rectangle(pdfPageImage, (xmin, ymin), (xmax, ymax), (0, 0, 0), 1)
    pdfPageImage_crop = pdfPageImage_table[ymin:ymax,xmin:xmax]
    convertToCV(pdfFilePath, xmin, xmax, ymin, ymax,output_dir,y,filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img)

## function to get the tabular part cropped from the original image
#def extractTableCoordinates_Grid(self,pdfFilePath,pdfPageImage,pdfPageImage_Mask,contours,index_for_table_contour,intermediate_Image_Path,output_dir,y,filename):
def extractTableCoordinates_Grid(pdfFilePath, pdfPageImage, pdfPageImage_Mask, contours,
                                     index_for_table_contour, intermediate_Image_Path, output_dir, y, filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img):

    logger.info("Extracting table coordinates")
    xTopLeft = []
    yTopLeft = []
    contour_rows = len(contours)
    contour_columns = len(contours[index_for_table_contour])
    for j in range(0,contour_columns-1):
        xTopLeft.append((contours[index_for_table_contour][j][0][0]))
        yTopLeft.append((contours[index_for_table_contour][j])[0][1])
    xmin = np.min(xTopLeft,axis=0)
    xmax = np.max(xTopLeft, axis=0)
    ymin = np.min(yTopLeft, axis=0)
    ymax = np.max(yTopLeft, axis=0)
    pdfPageImage_table = cv2.rectangle(pdfPageImage, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
    #cv2.imshow("win1",pdfPageImage_table)
    #cv2.waitKey()
    print(xmin,xmax,ymin,ymax)
    pdfPageImage_crop = pdfPageImage_table[ymin:ymax,xmin:xmax]
    #cv2.imshow("win2", pdfPageImage_crop)
    #cv2.waitKey()
    convertToCV(pdfFilePath,xmin,xmax,ymin,ymax,output_dir,y,filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img)

## Function to perform contour processing on image
#def imageContoursProcessing(self,pdfFilePath,pdfPageImage,pdfPageImage_Mask,PDF_Page_Tables,intermediate_Image_Path,output_dir,total_number_of_Figures,filename):
def imageContoursProcessing(pdfFilePath, pdfPageImage, pdfPageImage_Mask, PDF_Page_Tables,
                                intermediate_Image_Path, output_dir, total_number_of_Figures, filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img):

    logger.info("Processing the image contours")
    pdfPageImage_Mask_contours,contours,hierarchy=cv2.findContours(pdfPageImage_Mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    xPos_of_contour =[]
    yPos_of_contour = []
    width_of_contour = []
    height_of_contour = []
    angle_of_contour = []
    count_of_tabular_contours = 0
    number_Of_Contours = len(contours)
    yFlag=' '
    angFlag=' '
    for eachContour in range(0,number_Of_Contours):
        rect = cv2.minAreaRect(contours[eachContour])
        xPos_of_contour.append(rect[0][0])
        yPos_of_contour.append(rect[0][1])
        width_of_contour.append(rect[1][0])
        height_of_contour.append(rect[1][1])
        angle_of_contour.append(rect[2])
        box = cv2.boxPoints(cv2.minAreaRect(contours[eachContour]))
        box = np.int0(box)
        img_drwa_contours = cv2.drawContours(pdfPageImage, [box], 0, (0, 0, 255), 3)

    for y in range(0,len(height_of_contour)-1):
        if height_of_contour[y] == 0.0:
            yFlag = 'Y'
        else:
            yFlag = 'N'
            break
    for ang in range(0,len(angle_of_contour)):
        if angle_of_contour[ang] == 0.0:
            angFlag = 'Y'
        else:
            angFlag = 'N'
            break
    try:
        if yFlag =='Y' and angFlag =='Y':
            streamFlag = 'Y'
        else:
            streamFlag = 'N'
        contour_required = []
        if streamFlag == 'N':
            for y in range(0,len(height_of_contour)):
                if height_of_contour[y] > 10 and width_of_contour[y] > 10:
                    contour_required.append(y)
                '''
                if total_number_of_Figures > 0 and PDF_Page_Tables > 0:
                    try:
                        contours = imutils.grab_contours(contours)
                        contours = sorted(contours,key=cv2.contourArea,reverse=True)
                    except:
                        print("oops!", sys.exc_info()[0], "occured.")
                '''
            if len(contour_required) > 0:
               if len(contour_required) >= total_number_of_Figures:
                   for y in range(0,len(contour_required)):
                        index_for_table_contour = contour_required[y]
                        count_of_tabular_contours = count_of_tabular_contours + 1
                        extractTableCoordinates_Grid(pdfFilePath, pdfPageImage, pdfPageImage_Mask, contours,index_for_table_contour, intermediate_Image_Path,output_dir,y,filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img)
            else:
                extractTableCoordinates_streamWithoutTable(pdfFilePath, pdfPageImage, pdfPageImage_Mask, contours,PDF_Page_Tables, intermediate_Image_Path,output_dir,y,filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img)
        elif streamFlag =='Y':
            extractTableCoordinates_stream(pdfFilePath,pdfPageImage,pdfPageImage_Mask,contours,PDF_Page_Tables,intermediate_Image_Path,output_dir,y,filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img)
    except:
         print("oops!", sys.exc_info()[0], "occured.")

# Function to preprocess image using opencv
#def imagePreProcess(self,pdfFilePath,intermediate_Image_Path,PDF_Page_Tables,output_dir,total_number_of_Figures,filename):
def imagePreProcess(pdfFilePath, intermediate_Image_Path, PDF_Page_Tables, output_dir,
                        total_number_of_Figures, filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img):

    logger.info("Pre-process the image")
    pdfPageImage = cv2.imread(intermediate_Image_Path)
    pdfPageImage_gray = cv2.cvtColor(pdfPageImage,cv2.COLOR_BGR2GRAY)
    pdfPageImage_gray_threshold = cv2.adaptiveThreshold(~pdfPageImage_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,-2)
    pdfPageImage_gray_threshold_horizontal = pdfPageImage_gray_threshold.copy()
    pdfPageImage_gray_threshold_vertical = pdfPageImage_gray_threshold.copy()
    rows,columns = pdfPageImage_gray_threshold.shape
    scale = 15
    horizontalSize = columns/scale
    verticalSize = rows/scale

    ##Preparing horizontal structure

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(int(horizontalSize),1))
    pdfPageImage_gray_threshold_horizontal = cv2.erode(pdfPageImage_gray_threshold_horizontal,horizontalStructure,(-1,-1))
    pdfPageImage_gray_threshold_horizontal = cv2.dilate(pdfPageImage_gray_threshold_horizontal,horizontalStructure,(-1,-1))

    ##Preparing vertical structure

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(1,int(verticalSize)))
    pdfPageImage_gray_threshold_vertical = cv2.erode(pdfPageImage_gray_threshold_vertical, verticalStructure,(-1, -1))
    pdfPageImage_gray_threshold_vertical = cv2.dilate(pdfPageImage_gray_threshold_vertical, verticalStructure,(-1, -1))
    pdfPageImage_Mask = pdfPageImage_gray_threshold_horizontal + pdfPageImage_gray_threshold_vertical
    imageContoursProcessing(pdfFilePath,pdfPageImage,pdfPageImage_Mask,PDF_Page_Tables,intermediate_Image_Path,output_dir,total_number_of_Figures,filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img)

# Function to convert PDFs to images
#def convertPDFToImage(self,pdfFilePath,PDF_pages,PDF_Page_Tables,intermediate_Directory,output_dir,total_number_of_Figures,filename):
def convertPDFToImage(pdfFilePath, PDF_pages, PDF_Page_Tables, intermediate_Directory, output_dir,
                          total_number_of_Figures, filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img):
    logger.info("Converting PDF to image")
    pageImages = convert_from_path(pdfFilePath,100)
    intermediate_Image_Path = intermediate_Directory+str(PDF_pages)+'.jpg'
    pageImages[PDF_pages-1].save(intermediate_Image_Path,'JPEG')
    imagePreProcess(pdfFilePath,intermediate_Image_Path,PDF_Page_Tables,output_dir,total_number_of_Figures,filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img)

# Function to convert PDF to text to determine if there is any word like 'Table'
#def convertPDFToText(self,pdfFile,intermediate_Directory,output_dir,filename):
def convertPDFToText(pdfFile, intermediate_Directory, output_dir, filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img):
    logger.info("Converting PDF to text")
    pdfReader = PyPDF2.PdfFileReader(pdfFile)
    number_Of_Pages = pdfReader.numPages
    for page in range(1,number_Of_Pages+1):
        pageText = pdfReader.getPage(page-1).extractText()
        tableList1 = re.findall(r'\s+Table \d+' ,pageText)
        tableList2 = (re.findall(r'following table',pageText))
        tableList3 = (re.findall(r'\d+Table \d+',pageText))
        tableList4 = (re.findall(r'Table [A-Z]', pageText))
        tableList1_set = set(tableList1)
        tableList2_set = set(tableList2)
        tableList3_set = set(tableList3)
        tableList4_set = set(tableList4)
        number_Of_Tables1 = len(list(tableList1_set))
        number_Of_Tables2 = len(list(tableList2_set))
        number_Of_Tables3 = len(list(tableList3_set))
        number_Of_Tables4 = len(list(tableList4_set))
        total_number_of_tables = number_Of_Tables1+ number_Of_Tables2+number_Of_Tables3+ number_Of_Tables4
        figList1 = re.findall(r'Figure \d+', pageText)
        figList2 = (re.findall(r'following figure', pageText))
        figList3 = (re.findall(r'\d+Figure \d+', pageText))
        figList4 = (re.findall(r'Figure [A-Z]', pageText))
        figList1_set = set(figList1)
        figList2_set = set(figList2)
        figList3_set = set(figList3)
        figList4_set = set(figList4)
        number_Of_Figures1 = len(list(figList1_set))
        number_Of_Figures2 = len(list(figList2_set))
        number_Of_Figures3 = len(list(figList3_set))
        number_Of_Figures4 = len(list(figList4_set))
        total_number_of_Figures = number_Of_Figures1 + number_Of_Figures2 + number_Of_Figures3 + number_Of_Figures4
        convertPDFToImage(pdfFile,page,total_number_of_tables,intermediate_Directory,output_dir,total_number_of_Figures,filename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img)

#def splitPDF(self,input_directory_path,filename,splitPDF_dir):
def splitPDF(input_directory_path, filename, splitPDF_dir,logger):
    logger.info("Splitting the PDF into single page PDF")
    pdf = PdfFileReader(open((input_directory_path+"/"+filename), "rb"))
    for i in range(pdf.numPages):
        output = PdfFileWriter()
        output.addPage(pdf.getPage(i))
        with open(splitPDF_dir+"_%s_"  % (i+1)+filename,"wb") as outputStream:
            output.write(outputStream)


# Function to determine the document type
def documentType(input_directory_path,output_dir,logger,mappingDF_Yaxis,mappingDF_Xaxis):
#def documentType(self, input_directory_path, output_dir):
    logger.info("Get the document type")
    for filename in os.listdir(input_directory_path):
        intermediate_Directory = input_directory_path +"/"+ "IntermediateResults/"+filename+"/"
        output_dir = input_directory_path+"/" + "Outputs/"
        splitPDF_dir = input_directory_path +"/"+ "splitPDFS/"
        if not os.path.exists(intermediate_Directory):
            os.makedirs(intermediate_Directory)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(splitPDF_dir):
            os.makedirs(splitPDF_dir)
        if filename.endswith('.pdf'):
            splitPDF(input_directory_path,filename,splitPDF_dir,logger)
            file_jpg=(filename[:filename.rindex('.')]+".jpg")
            print("File_jpg",file_jpg)
            if os.path.isfile(input_directory_path+"/"+file_jpg):
                flag_img = 'Y'
                mappingDF_Yaxis = pd.read_excel(input_directory_path + "/" + "Plots_Y_Image.xls")
                mappingDF_Xaxis = pd.read_excel(input_directory_path + "/" + "Plots_X_Image.xls")

    try:
        for splitfilename in os.listdir((splitPDF_dir)):
            convertPDFToText((splitPDF_dir + splitfilename), intermediate_Directory, (output_dir + "/"),splitfilename,logger,mappingDF_Yaxis,mappingDF_Xaxis,flag_img)
            deleteExtraCSVs(output_dir,logger)
            finalFile = re.sub('_.*?_','',splitfilename)
            mergeFiles(input_directory_path,output_dir,finalFile,logger)

    except:
        print("oops!", sys.exc_info()[0], "occured.")
    return output_dir

#def deleteExtraCSVs(self,output_directory):
def deleteExtraCSVs(output_directory,logger):
    logger.info("Deleting invalid CSVs")
    for file in os.listdir(output_directory):
        pd_csv = pd.read_csv(output_directory+file)
        num_of_cols = len(pd_csv.columns)
        cnt_of_text_lines = 1
        first_col = pd_csv.iloc[:,0]
        lines_count = 0
        delete_flag = ' '
        for rows in first_col:
            str1 = str(rows)
            wordCount=len(str1.split())
            if wordCount >=10:

                lines_count = lines_count+1
            if lines_count >10:
                delete_flag = 'Y'
                break
        if num_of_cols ==1 or delete_flag =='Y':
            os.remove(output_directory+file)

#if __name__ == '__main__':
def mainPDF(input_dir, output_dir):
    abs_path = ("../logs/")
    os.chdir(abs_path)
    path = os.getcwd()  # type: str
    logging.config.fileConfig("../conf/logging.conf")
    logger = logging.getLogger('MainLogger')

    fh = logging.FileHandler(path + "/logger_{:%Y-%m-%d}.log".format(datetime.now()), mode='a')
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | '
                                  '%(filename)s-%(funcName)s-%(lineno)04d | %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("Starting the appication")
    input_directory = input_dir
    output_directory = ''
    mappingDF_Yaxis = pd.read_excel(input_directory+"/"+"Plots_Y.xls")
    mappingDF_Xaxis= pd.read_excel(input_directory+"/"+"Plots_X.xls")
    output_directory= documentType(input_directory,output_directory,logger,mappingDF_Yaxis,mappingDF_Xaxis)





