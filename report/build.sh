#!/bin/sh
echo "Hi, I'm a customized script to compile your bachelor thesis."
pdflatex Image_Input_Calculator
pdflatex Image_Input_Calculator
biber Image_Input_Calculator
pdflatex Image_Input_Calculator
pdflatex Image_Input_Calculator
echo "I did it. Now, let's clean up."
