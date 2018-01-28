from PIL import Image
import pandas as pd
import glob
import csv

data = pd.DataFrame()
size = 0, 0
scores = {} # filename : score (num 1-4)

for filename in glob.glob('./test_ds/train/*.jpeg'):

	im = Image.open(filename, 'r')
	
	width, height = im.size
	
	if (im.size > size):
		size = width, height

with open('test_ds/labels.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter=',')
	for row in readCSV:
   		scores[row[0]] = row[1]
   		#print("filename: " + row[0] + ", score: " + str(row[1]))

for filename in glob.glob('./test_ds/train/*.jpeg'):
	im = Image.open(filename, 'r')

	new_width, new_height = size
	
	im = im.resize((new_width, new_height), Image.ANTIALIAS)
		
	px_vals = list(im.getdata())

	data = pd.DataFrame([x[0]*256**2 + 256* x[1] + x[2] for x in px_vals]).transpose()
	filename = filename.split('/')[3]

	data["dr"] = scores[filename.split('.')[0]]
	print("added {} for img {}".format(scores[filename.split('.')[0]], filename.split('.')[0] ))

