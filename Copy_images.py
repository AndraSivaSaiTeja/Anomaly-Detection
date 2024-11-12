import shutil
import os
import sys

source = "/home/turing/Ranjeet/pytorch-cutpaste-master/Data/"
dest = "/home/turing/Ranjeet/Code_Anomaly_Detection/RD_Unified/Data/train/good/"

fols = os.listdir(source)
j = 0
for fol in fols:
	fil = source + fol
	fil_p = fil + "/" + "train/good/"
	files = os.listdir(fil_p)
	for f_p in files:
		path_f = fil_p + f_p
		path_d = dest + str(j) + ".png"
		shutil.copy(path_f, path_d)
		j = j + 1
