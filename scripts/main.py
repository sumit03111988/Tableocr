from scripts.image_to_pdf import FileConversion
from scripts.pdf_to_output import PdfOutput
import pandas as pd
import configparser
import scripts.csvgenerator as pdfprocessor
if __name__ == '__main__':

    config = configparser.ConfigParser()
    config.read("../conf/common.conf")
    #input_directory = config.get('input', 'input_path') + '/tables'
    input_directory = config.get('input', 'input_path') 
    bash_script = config.get('bash_script', 'bash_path')
    output_directory=config.get('output', 'output_path')

    try:
        objconvert = FileConversion(input_directory,bash_script)
        objconvert.FileConversion()
        #pass

    except:
        print("Error occured in converting pdf to image")
        raise

    try:

        #objpdf = PdfOutput(input_directory, output_directory)
        objpdf = pdfprocessor.mainPDF(input_directory,output_directory)
        #input_directory = sys.argv[1]
            # print(input_directory)
        #output_directory = ''
        #mappingDF_Yaxis = pd.read_excel(input_directory +"Plots_Y.xls")
        #mappingDF_Xaxis = pd.read_excel(input_directory + "Plots_X.xls")
        #print(mappingDF_Yaxis)

        #objpdf.objoutput.documentType(input_directory, output_directory)


    except:
        print("Error occured in converting pdf to csv")
        raise

