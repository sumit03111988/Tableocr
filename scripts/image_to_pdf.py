import img2pdf
from PIL import Image
import os
#import ocrmypdf
import subprocess
import shlex
import logging
import logging.config
from datetime import datetime
import os
import time


class FileConversion:

    def __init__(self,input_directory,bash_script):
        self.input_path=input_directory
        self.bash_script=bash_script
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

    def FileConversion(self):

        for filename in os.listdir(self.input_path):
            input_path = self.input_path
            bash_script = self.bash_script

            name, ext = os.path.splitext(filename)
            if (ext == ".jpg" or ext ==".png"):
                self.logger.info("conversion of image %s to pdf starting" % (name+ext))

                try:
                    image = Image.open(input_path + '/' + filename)
                    pdf_bytes = img2pdf.convert(image.filename)
                    file = open(input_path + '/' + name + ".pdf", "wb")
                    file.write(pdf_bytes)
                    image.close()
                    file.close()
                    subprocess.call(shlex.split('%s/image_pdf.sh %s/%s.pdf  %s/%s.pdf' % (
                        bash_script, input_path, name, input_path, name)))

                    self.logger.info("%s to Pdf successfully completed" % (name + ext))

                except:
                    print("Error in conversion of file %s" % (name + ext))



            elif(ext==".doc" or ext ==".docx" ):

                self.logger.info("conversion of word %s to pdf starting" % (name + ext))
                self.logger.info("conversion of word %s to pdf starting" % (input_path+name + ext))

                subprocess.call(shlex.split('%s/word_pdf.sh %s %s' % (
                    bash_script, input_path,name+ext)))


                time.sleep(1)

                subprocess.call(shlex.split('%s/image_pdf.sh %s/%s.pdf  %s/%s.pdf' % (
                    bash_script, input_path, name, input_path, name)))




            elif (ext != ".pdf"):
                print("Currently we are handling only jpg,png,doc,docx and pdf files")
                discard_path='../discardfiles/'
                #subprocess.call(shlex.split('%s/move_unwanted_file.sh %s/%s %s' % (
                #    bash_script, input_path, name+ext, discard_path)))



