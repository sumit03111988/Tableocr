#!/bin/sh
#Takes parameter from Pycharm
#1st parameter -input pdf 
#2nd parameter output pdf file
#This convert scanned pdf to searchable pdf 
ocrmypdf -l eng --rotate-pages --deskew $1 $2 
