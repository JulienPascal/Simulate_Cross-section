#!/bin/bash
echo "CONVERTING TO MARKDOW"
jupyter nbconvert --to markdown $PWD/Young_2010.ipynb
echo "DONE"
