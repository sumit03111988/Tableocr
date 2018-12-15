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
from PyPDF2 import PdfFileReader, PdfFileWriter
import xlrd
import logging
import logging.config
import datetime

import fnmatch


class PdfOutput:

    def __init__(self,input_directory,output_directory):
        self.mappingDF_Yaxis = pd.read_excel(input_directory + "Plots_Y.xls")
        self.mappingDF_Xaxis = pd.read_excel(input_directory + "Plots_X.xls")
        abs_path = ("../logs/")
        os.chdir(abs_path)
        path = os.getcwd()  # type: str
        logging.config.fileConfig("../conf/logging.conf")
        self.logger = logging.getLogger('MainLogger')

        fh = logging.FileHandler(path + "/logger_{:%Y-%m-%d}.log".format(datetime.now()), mode='a')
        formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | '
                                      '%(filename)s-%(funcName)s-%(lineno)04d | %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.input_directory=input_directory
        self.output_directory=output_directory

    def mergeFiles(self,output_dir, filename):
        finalOutputCSV = "/home/impadmin/Divya/PersonalDocs/AxisUseCase/TestDataset/TestFolder" + "/FinalCSVs/"
        if not os.path.exists(finalOutputCSV):
            os.makedirs(finalOutputCSV)
        finalCSVName = os.path.splitext(filename)[0] + '.csv'
        finalCSV = open(finalOutputCSV + finalCSVName, "ab")
        # finalCSV.write("HIII")
        # fin = open("/home/impadmin/Divya/PersonalDocs/AxisUseCase/TestDataset/TestFolder/Outputs/1-page-1-table-1.csv","rb")
        # data1 = fin.read()
        # print(type(data1))
        # finalCSV.write(data1)
        line = 1
        for eachFile in os.listdir(output_dir):
            if fnmatch.fnmatch(eachFile, '*-page*'):
                print("EachFilename", eachFile)
                fin = open((output_dir + '/' + eachFile), "rb")
                data1 = fin.read()
                if line != 1:
                    finalCSV.write(b"\n")
                    finalCSV.write(b"\n")
                finalCSV.write(data1)
                os.remove(output_dir + '/' + eachFile)
            line = line + 1

        finalCSV.close()

    def findIndex(self,mappingDF, columnNames, value, flag):
        index_found = 'N'
        while index_found == 'N':
            list_empty = mappingDF.index[mappingDF[columnNames[0]] == value].tolist()
            # print(index_found)
            # print(type(list_empty))
            # print("new List",list_empty)
            if list_empty:
                # print("list",list_empty)
                # print("length",len(list_empty))
                final_index = list_empty[0]
                # print("final index",final_index)
                index_found = 'Y'
                break
            else:
                if flag == 'Y':
                    value = value - 1
                elif flag == 'X':
                    value = value + 1
        return final_index

    def convertToCV(self,pdfFilePath, xmin, xmax, ymin, ymax, output_dir, y, filename):
        xConversionfactor = 0.7373353937
        xmin_new = xmin * xConversionfactor
        xmax_new = xmax * xConversionfactor
        columnNames = list(self.mappingDF_Yaxis.head(0))
        index_ymin = self.findIndex(self.mappingDF_Yaxis, columnNames, ymin, 'Y')
        ymin_new = self.mappingDF_Yaxis.iloc[index_ymin][1]
        index_ymax = self.findIndex(self.mappingDF_Yaxis, columnNames, ymax, 'Y')
        ymax_new = self.mappingDF_Yaxis.iloc[index_ymax][1]
        if xmin < 98:
            xmin = 98
        index_xmin = self.findIndex(self.mappingDF_Xaxis, columnNames, xmin, 'X')
        print('index_xmin ', index_xmin)
        xmin_new = self.mappingDF_Xaxis.iloc[index_xmin][1]
        print(xmin_new)
        if xmax > 800:
            xmax = 800
        index_xmax = self.findIndex(self.mappingDF_Xaxis, columnNames, xmax, 'X')
        print("index_xmax ", index_xmax)
        xmax_new = self.mappingDF_Xaxis.iloc[index_xmax][1]

        print(xmin_new, ymin_new, xmax_new, ymax_new)
        table_areas_string = str(xmin_new) + ',' + str(ymin_new) + ',' + str(xmax_new) + ',' + str(ymax_new)
        tables = cm.read_pdf(pdfFilePath, flavor='stream', pages='1', table_areas=[table_areas_string])
        print(tables)
        output_file_name = output_dir + str(y) + '.csv'
        print(output_dir)
        print(output_file_name)
        tables.export(output_file_name, f='csv')
        # deleteExtraCSVs(output_dir)
        # mergeFiles(output_dir,filename)

    def fetchTable(self,pdfFilePath, pdfPageImage, pdfPageImage_Mask, xTopLeft, yTopLeft, xBottomRight, yBottomRight,
                   intermediate_Image_Path, i, output_dir, y, filename):
        print("in fetch function")
        pdfPageImage_table = cv2.rectangle(pdfPageImage, (xTopLeft, yTopLeft - 60), (xBottomRight, yBottomRight),
                                           (0, 0, 0), 1)
        print(xTopLeft, xBottomRight, yTopLeft, yBottomRight)
        pdfPageImage_crop = pdfPageImage_table[yTopLeft - 60:yBottomRight, xTopLeft:xBottomRight]
        self.convertToCV(pdfFilePath, xTopLeft, xBottomRight, yBottomRight, yTopLeft, output_dir, y, filename)

    def extractTableCoordinates_streamWithoutTable(self,pdfFilePath, pdfPageImage, pdfPageImage_Mask, contours,
                                                   PDF_Page_Tables, intermediate_Image_Path, output_dir, y, filename):
        print("Without table keyword")
        yPosPrevious = 0

        xBottomRight = []
        yBottomRight = []

        contour_rows = len(contours)
        print(contours)
        flag_table = 'N'
        for i in range(0, contour_rows):
            # rect = cv2.minAreaRect(contours[contour_rows-1])
            rect = cv2.minAreaRect(contours[i])
            xPos_of_contour = (rect[0][0])
            yPos_of_contour = (rect[0][1])
            width_of_contour = (rect[1][0])
            height_of_contour = (rect[1][1])
            angle_of_contour = (rect[2])
            xArr = []
            yArr = []
            # xBottomRight = (contours[i][0][0][0])
            # yBottomRight = (contours[i][0][0][1])
            # yPosCurrent = contours[i][0][0][1]
            contour_columns = len(contours[i])
            yPosCurrent = contours[i][contour_columns - 1][0][1]
            # xPosCurrent = contours[i][contour_columns - 1][0][0]

            if flag_table == 'N':
                xBottomRight = contours[i][contour_columns - 1][0][0]
                yBottomRight = contours[i][contour_columns - 1][0][1]

            print(yPosCurrent)
            for j in range(0, contour_columns):
                print("Current", yPosCurrent)
                print("previou", yPosPrevious)
                print("i ", i)
                print("j ", j)
                print(width_of_contour)
                print(height_of_contour)
                print(contour_columns)
                print(contour_rows)
                xArr.append((contours[i][j][0][0]))
                yArr.append((contours[i][j][0][1]))

            xTopLeft = ((contours[i][0][0][0]))
            yTopLeft = ((contours[i][0])[0][1])
            # if ((yPosPrevious - yPosCurrent > 80 or i == contour_rows - 1) and (width_of_contour < 100 and height_of_contour < 100)):
            if ((yPosPrevious - yPosCurrent > 80 or i == contour_rows - 1)):
                xTopLeft = ((contours[i - 1][0][0][0]))
                yTopLeft = ((contours[i - 1][0])[0][1])
                print("Difference x ", xBottomRight - xTopLeft)
                print("Difference y ", yTopLeft - yBottomRight)
                if (((xBottomRight - xTopLeft) >= 200) and ((yBottomRight - yTopLeft) >= 200)):
                    self.fetchTable(pdfFilePath, pdfPageImage, pdfPageImage_Mask, xTopLeft, yTopLeft, xBottomRight,
                               yBottomRight, intermediate_Image_Path, i, output_dir, y, filename)
                    flag_table = 'N'
                    yPosPrevious = yPosCurrent
            else:
                xTopLeft = ((contours[i][0][0][0]))
                yTopLeft = ((contours[i][0])[0][1])
                yPosPrevious = yPosCurrent
                flag_table = 'Y'

        print("In Last function")
        print(xBottomRight)
        print(yBottomRight)
        print(xTopLeft)
        print(yTopLeft)

    def extractTableCoordinates_stream(self,pdfFilePath, pdfPageImage, pdfPageImage_Mask, contours, PDF_Page_Tables,
                                       intermediate_Image_Path, output_dir, y, filename):
        print("Calling stream")
        xTopLeft = []
        yTopLeft = []
        xBottomRight = []
        yBottomRight = []
        contour_rows = len(contours)
        print("rows", contour_rows)
        print(contours)
        for i in range(0, contour_rows):
            contour_columns = len(contours[i])
            for j in range(0, contour_columns):
                xTopLeft.append((contours[i][j][0][0]))
                xBottomRight.append((contours[i][contour_columns - 1][0][0]))
                yTopLeft.append((contours[i][j])[0][1])
                yBottomRight.append(contours[i][contour_columns - 1][0][1])
        print("XtopLEft")
        print(xTopLeft)
        xmin = np.min(xTopLeft, axis=0)
        xmax = np.max(xBottomRight, axis=0)
        ymin = np.min(yTopLeft, axis=0)
        ymax = np.max(yBottomRight, axis=0)
        pdfPageImage_table = cv2.rectangle(pdfPageImage, (xmin, ymin), (xmax, ymax), (0, 0, 0), 1)
        # cv2.imshow("win8",pdfPageImage_table)
        # cv2.waitKey()
        print(xmin, xmax, ymin, ymax)
        pdfPageImage_crop = pdfPageImage_table[ymin:ymax, xmin:xmax]
        # cv2.imshow("win9",pdfPageImage_crop)
        # cv2.waitKey()
        self.convertToCV(pdfFilePath, xmin, xmax, ymin, ymax, output_dir, y, filename)
        # table_areas_string = str(xmin) + ',' + str(ymax) + ',' + str(xmax) + ',' + str(ymin)
        # tables = cm.read_pdf(pdfFilePath, flavor='stream', pages='1', table_areas=[table_areas_string])
        # tables.export(output_dir + '.csv', f='csv')
        # cv2.imwrite("/home/impadmin/Divya/PersonalDocs/AxisUseCase/TestDataset/TestFolder/Outputs/1.jpg",pdfPageImage_crop)

    ## function to get the tabular part cropped from the original image
    def extractTableCoordinates_Grid(self,pdfFilePath, pdfPageImage, pdfPageImage_Mask, contours, index_for_table_contour,
                                     intermediate_Image_Path, output_dir, y, filename):
        print("Calling grid")
        xTopLeft = []
        yTopLeft = []
        print(contours)
        contour_rows = len(contours)
        contour_columns = len(contours[index_for_table_contour])
        print("columns", contour_columns)
        print("rows", contour_rows)
        print("printing contours in new fun"
              "ction", contours[index_for_table_contour])
        print(contours[index_for_table_contour][0][0][0])
        for j in range(0, contour_columns - 1):
            xTopLeft.append((contours[index_for_table_contour][j][0][0]))
            yTopLeft.append((contours[index_for_table_contour][j])[0][1])
        xmin = np.min(xTopLeft, axis=0)
        xmax = np.max(xTopLeft, axis=0)
        ymin = np.min(yTopLeft, axis=0)
        ymax = np.max(yTopLeft, axis=0)
        pdfPageImage_table = cv2.rectangle(pdfPageImage, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        # cv2.imshow("win8",pdfPageImage_table)
        # cv2.waitKey()
        print(xmin, xmax, ymin, ymax)
        pdfPageImage_crop = pdfPageImage_table[ymin:ymax, xmin:xmax]
        # cv2.imshow("win9",pdfPageImage_crop)
        # cv2.waitKey()
        # table_areas_string = str(xmin) + ',' + str(ymax) + ',' + str(xmax) + ',' + str(ymin)
        # tables = cm.read_pdf(pdfFilePath, flavor='stream', pages='1', table_areas=[table_areas_string])
        # tables = cm.read_pdf(pdfFilePath, flavor='stream', pages='1', table_areas=['72.79,708.814,508.529,459.391'])
        # tables.export(output_dir + '.csv', f='csv')
        self.convertToCV(pdfFilePath, xmin, xmax, ymin, ymax, output_dir, y, filename)
        # cv2.imwrite("/home/impadmin/Divya/PersonalDocs/AxisUseCase/TestDataset/TestFolder/Outputs/004.jpg", pdfPageImage_crop)

    ## Function to perform contour processing on image
    def imageContoursProcessing(self,pdfFilePath, pdfPageImage, pdfPageImage_Mask, PDF_Page_Tables, intermediate_Image_Path,
                                output_dir, total_number_of_Figures, filename):
        pdfPageImage_Mask_contours, contours, hierarchy = cv2.findContours(pdfPageImage_Mask, cv2.RETR_EXTERNAL,
                                                                           cv2.CHAIN_APPROX_SIMPLE)
        # print(pdfPageImage_Mask_contours)
        # print(contours)
        # print(hierarchy)

        xPos_of_contour = []
        yPos_of_contour = []
        width_of_contour = []
        height_of_contour = []
        angle_of_contour = []
        # cv2.imshow("win6",pdfPageImage_Mask_contours)
        # cv2.waitKey()
        count_of_tabular_contours = 0
        number_Of_Contours = len(contours)
        print("number Of Contours", number_Of_Contours)
        for eachContour in range(0, number_Of_Contours):
            rect = cv2.minAreaRect(contours[eachContour])
            print("Min area of contour ", eachContour, "is ", cv2.minAreaRect(contours[eachContour]))
            xPos_of_contour.append(rect[0][0])
            yPos_of_contour.append(rect[0][1])
            width_of_contour.append(rect[1][0])
            height_of_contour.append(rect[1][1])
            angle_of_contour.append(rect[2])

            box = cv2.boxPoints(cv2.minAreaRect(contours[eachContour]))
            box = np.int0(box)

            img_drwa_contours = cv2.drawContours(pdfPageImage, [box], 0, (0, 0, 255), 3)
            # cv2.imshow("win9", img_drwa_contours)
            # cv2.waitKey()

        for eachContour in range(0, number_Of_Contours):
            print("X value ", xPos_of_contour[eachContour])
            print("Y value ", yPos_of_contour[eachContour])
            print("W value ", width_of_contour[eachContour])
            print("H value ", height_of_contour[eachContour])
            print("Angle value ", angle_of_contour[eachContour])

        for y in range(0, len(height_of_contour) - 1):
            if height_of_contour[y] == 0.0:
                yFlag = 'Y'
            else:
                yFlag = 'N'
                break
            print(yFlag)
        for ang in range(0, len(angle_of_contour)):
            if angle_of_contour[ang] == 0.0:
                angFlag = 'Y'
            else:
                angFlag = 'N'
                break
            print(angFlag)
        try:
            if yFlag == 'Y' and angFlag == 'Y':
                streamFlag = 'Y'
            else:
                streamFlag = 'N'
            # print("Streaming Flag ",streamFlag)
            contour_required = []
            # if height_of_contour > 10 and width_of_contour > 10 and angle_of_contour == 0.0:
            if streamFlag == 'N':
                for y in range(0, len(height_of_contour)):
                    if height_of_contour[y] > 10 and width_of_contour[y] > 10:
                        contour_required.append(y)

                    if total_number_of_Figures > 0 and PDF_Page_Tables > 0:
                        print("Before")
                        print(contours)
                        try:
                            contours = imutils.grab_contours(contours)

                            contours = sorted(contours, key=cv2.contourArea, reverse=True)
                        except:
                            print("oops!", sys.exc_info()[0], "occured.")
                        print("After")
                        print(contours)

                print("contours_required", contour_required)
                print("length", len(contour_required))

                print("New contour", contours)
                # if len(contour_required)>0:

                if len(contour_required) > 0:
                    if len(contour_required) > total_number_of_Figures:
                        for y in range(0, len(contour_required)):
                            index_for_table_contour = contour_required[y]
                            count_of_tabular_contours = count_of_tabular_contours + 1
                            self.extractTableCoordinates_Grid(pdfFilePath, pdfPageImage, pdfPageImage_Mask, contours,
                                                         index_for_table_contour, intermediate_Image_Path, output_dir,
                                                         y, filename)
                else:
                    self.extractTableCoordinates_streamWithoutTable(pdfFilePath, pdfPageImage, pdfPageImage_Mask, contours,
                                                               PDF_Page_Tables, intermediate_Image_Path, output_dir, y,
                                                               filename)
            elif streamFlag == 'Y':
                self.extractTableCoordinates_stream(pdfFilePath, pdfPageImage, pdfPageImage_Mask, contours, PDF_Page_Tables,
                                               intermediate_Image_Path, output_dir, y, filename)
        except:
            print("oops!", sys.exc_info()[0], "occured.")

    # Function to preprocess image using opencv
    def imagePreProcess(self,pdfFilePath, intermediate_Image_Path, PDF_Page_Tables, output_dir, total_number_of_Figures,
                        filename):
        pdfPageImage = cv2.imread(intermediate_Image_Path)
        print("Image size", pdfPageImage.size)
        # cv2.imshow("win1",pdfPageImage)
        # cv2.waitKey()
        pdfPageImage_gray = cv2.cvtColor(pdfPageImage, cv2.COLOR_BGR2GRAY)
        # cv2.imshow("win2",pdfPageImage_gray)
        # cv2.waitKey()
        pdfPageImage_gray_threshold = cv2.adaptiveThreshold(~pdfPageImage_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                            cv2.THRESH_BINARY, 15, -2)
        pdfPageImage_gray_threshold_horizontal = pdfPageImage_gray_threshold.copy()
        pdfPageImage_gray_threshold_vertical = pdfPageImage_gray_threshold.copy()
        rows, columns = pdfPageImage_gray_threshold.shape
        scale = 15
        horizontalSize = columns / scale
        verticalSize = rows / scale

        ##Preparing horizontal structure

        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (int(horizontalSize), 1))
        pdfPageImage_gray_threshold_horizontal = cv2.erode(pdfPageImage_gray_threshold_horizontal, horizontalStructure,
                                                           (-1, -1))
        pdfPageImage_gray_threshold_horizontal = cv2.dilate(pdfPageImage_gray_threshold_horizontal, horizontalStructure,
                                                            (-1, -1))
        # cv2.imshow("win3",pdfPageImage_gray_threshold_horizontal)
        # cv2.waitKey()

        ##Preparing vertical structure

        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(verticalSize)))
        pdfPageImage_gray_threshold_vertical = cv2.erode(pdfPageImage_gray_threshold_vertical, verticalStructure,
                                                         (-1, -1))
        pdfPageImage_gray_threshold_vertical = cv2.dilate(pdfPageImage_gray_threshold_vertical, verticalStructure,
                                                          (-1, -1))
        # cv2.imshow("win4", pdfPageImage_gray_threshold_vertical)
        # cv2.waitKey()

        pdfPageImage_Mask = pdfPageImage_gray_threshold_horizontal + pdfPageImage_gray_threshold_vertical
        # cv2.imshow("win5",pdfPageImage_Mask)
        # cv2.waitKey()

        self.imageContoursProcessing(pdfFilePath, pdfPageImage, pdfPageImage_Mask, PDF_Page_Tables, intermediate_Image_Path,
                                output_dir, total_number_of_Figures, filename)

    # Function to convert PDFs to images
    def convertPDFToImage(self,pdfFilePath, PDF_pages, PDF_Page_Tables, intermediate_Directory, output_dir,
                          total_number_of_Figures, filename):
        print("printing PDF pages", PDF_pages)
        print("Printing tables on PDF pages", PDF_Page_Tables)
        pageImages = convert_from_path(pdfFilePath, 100)
        intermediate_Image_Path = intermediate_Directory + str(PDF_pages) + '.jpg'
        pageImages[PDF_pages - 1].save(intermediate_Image_Path, 'JPEG')
        self.imagePreProcess(pdfFilePath, intermediate_Image_Path, PDF_Page_Tables, output_dir, total_number_of_Figures,
                        filename)

    # Function to convert PDF to text to determine if there is any word like 'Table'
    def convertPDFToText(self,pdfFile, intermediate_Directory, output_dir, filename):
        pdfReader = PyPDF2.PdfFileReader(pdfFile)
        number_Of_Pages = pdfReader.numPages
        print(number_Of_Pages)
        for page in range(1, number_Of_Pages + 1):
            pageText = pdfReader.getPage(page - 1).extractText()
            # print(type(pageText))
            # print(pageText)
            tableList1 = re.findall(r'\s+Table \d+', pageText)
            tableList2 = (re.findall(r'following table', pageText))
            tableList3 = (re.findall(r'\d+Table \d+', pageText))
            tableList4 = (re.findall(r'Table [A-Z]', pageText))
            tableList1_set = set(tableList1)
            tableList2_set = set(tableList2)
            tableList3_set = set(tableList3)
            tableList4_set = set(tableList4)

            print("Tableist", tableList4)
            # print("Position of word table ",pageText.find("Impacts"))
            # print(type(table))
            # print(table)
            print(tableList4)
            number_Of_Tables1 = len(list(tableList1_set))
            number_Of_Tables2 = len(list(tableList2_set))
            number_Of_Tables3 = len(list(tableList3_set))
            number_Of_Tables4 = len(list(tableList4_set))
            total_number_of_tables = number_Of_Tables1 + number_Of_Tables2 + number_Of_Tables3 + number_Of_Tables4
            # if(total_number_of_tables != 0):
            print("Page number ", page, " has ", total_number_of_tables, " tables")

            figList1 = re.findall(r'\s+Figure \d+', pageText)
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
            # if(total_number_of_tables != 0):
            print("Page number ", page, " has ", total_number_of_Figures, " figures")

            self.convertPDFToImage(pdfFile, page, total_number_of_tables, intermediate_Directory, output_dir,
                              total_number_of_Figures, filename)
            # else:

            #   print("Page number ",page," has no tables")

    def splitPDF(self,input_directory_path, filename, splitPDF_dir):
        pdf = PdfFileReader(open((input_directory_path + filename), "rb"))
        for i in range(pdf.numPages):
            output = PdfFileWriter()
            output.addPage(pdf.getPage(i))
            with open(splitPDF_dir + "_%s_" % (i + 1) + filename, "wb") as outputStream:
                output.write(outputStream)

    # Function to determine the document type
    def documentType(self,input_directory_path, output_dir):
        self.logger.info("PDF to Csv process started")
        for filename in os.listdir(input_directory_path):
            intermediate_Directory = input_directory_path + "IntermediateResults/" + filename + "/"
            output_dir = input_directory_path + "Outputs/"
            splitPDF_dir = input_directory_path + "splitPDFS/"
            if not os.path.exists(intermediate_Directory):
                os.makedirs(intermediate_Directory)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            if not os.path.exists(splitPDF_dir):
                os.makedirs(splitPDF_dir)
            if filename.endswith('.pdf'):
                self.splitPDF(input_directory_path, filename, splitPDF_dir)
                # convertPDFToText((input_directory_path+filename),intermediate_Directory,output_dir)
        try:
            for splitfilename in os.listdir((splitPDF_dir)):
                # convertPDFToText((splitPDF_dir + splitfilename), intermediate_Directory, (output_dir+"/"+splitfilename),filename)
                self.convertPDFToText((splitPDF_dir + splitfilename), intermediate_Directory, (output_dir + "/"),
                                 splitfilename)
                self.deleteExtraCSVs(output_dir)
                print("Filename", splitfilename)
                finalFile = re.sub('_.*?_', '', splitfilename)
                self.mergeFiles(output_dir, finalFile)



        except:
            print("oops!", sys.exc_info()[0], "occured.")
            # elif filename.endswith('.jpg'):
            # file = input_directory_path+filename
            # imagePreProcess(file,1)
        print("Output DIR", output_dir)
        return output_dir

    def deleteExtraCSVs(self,output_directory):
        for file in os.listdir(output_directory):
            pd_csv = pd.read_csv(output_directory + file)
            # print(pd_csv)
            num_of_cols = len(pd_csv.columns)
            cnt_of_text_lines = 1
            first_col = pd_csv.iloc[:, 0]
            print("first column", first_col)
            print(type(first_col))
            lines_count = 0
            delete_flag = ' '
            for rows in first_col:
                str1 = str(rows)
                print("String type", type(str1), str1)
                wordCount = len(str1.split())
                if wordCount > 12:
                    lines_count = lines_count + 1
                if lines_count > 10:
                    delete_flag = 'Y'
                    break
            if num_of_cols == 1 or delete_flag == 'Y':
                os.remove(output_directory + file)



