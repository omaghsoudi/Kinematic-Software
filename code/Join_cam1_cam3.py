import cv2, os, sys, math, argparse
import numpy as np
import pandas as pd
from glob import glob
from copy import deepcopy
from termcolor import colored

parser = argparse.ArgumentParser(description = "Help on how to set the parameters and use short keys while calling the code:")
parser.add_argument("-animal", "--Animal_Name", type=str, help = "The name of animal", default = "rat")
args = parser.parse_args(); ARGS = vars(args)

Paw_Name = 'f'
Animal_Name = ARGS['Animal_Name']
Input_Name = Paw_Name+"."+Animal_Name



CurrentPath = "."
Coordinate_Path = sorted(glob(os.path.join(CurrentPath, "cam1."+ Input_Name +"*.csv")))

for DirPath in Coordinate_Path:
    cam1_Path_Front = DirPath
    cam3_Path = deepcopy(DirPath)
    cam3_Path = list(cam3_Path)
    cam3_Path[5] = '3'
    cam3_Path_Front = ''.join(cam3_Path)

    cam1_Path_Hind = list(cam1_Path_Front)
    cam1_Path_Hind[7] = 'h'
    cam1_Path_Hind = ''.join(cam1_Path_Hind)

    cam3_Path_Hind = list(cam3_Path_Front)
    cam3_Path_Hind[7] = 'h'
    cam3_Path_Hind = ''.join(cam3_Path_Hind)


    print("Combining cam1 and cam3 (front and hind paws); Processing dir %s AND %s  \r" %(cam1_Path_Front, cam3_Path_Front))

    Hedaer = pd.read_csv(cam1_Path_Front, sep=',', nrows=4)

    data_temp_cam1_f = pd.read_csv(cam1_Path_Front, sep=',', index_col=0, skiprows=[0,1,2,3,4])
    data_temp_cam1_h = pd.read_csv(cam1_Path_Hind, sep=',', index_col=0, skiprows=[0,1,2,3,4])

    columns_old = data_temp_cam1_f.columns[2:]
    data_temp_cam1_f = data_temp_cam1_f[columns_old]
    columns_new = []
    for L0 in columns_old: 
        L0 = list(L0)
        L0[13] = chr(ord(L0[13])+5)
        columns_new.append(''.join(L0))

    data_temp_cam1_f.columns = columns_new
    data_temp_cam1 = pd.concat([data_temp_cam1_h, data_temp_cam1_f], axis=1, sort=False)



    data_temp_cam3_f = pd.read_csv(cam3_Path_Front, sep=',', index_col=0, skiprows=[0,1,2,3,4])
    data_temp_cam3_h = pd.read_csv(cam3_Path_Hind, sep=',', index_col=0, skiprows=[0,1,2,3,4])

    columns_old = data_temp_cam3_f.columns[2:]
    data_temp_cam3_f = data_temp_cam3_f[columns_old]
    columns_new = []
    for L0 in columns_old: 
        L0 = list(L0)
        L0[13] = chr(ord(L0[13])-8)
        columns_new.append(''.join(L0))
    data_temp_cam3_f.columns = columns_new

    columns_old = data_temp_cam3_h.columns[2:]
    columns_new = [data_temp_cam3_h.columns[0], data_temp_cam3_h.columns[1]]
    for L0 in columns_old: 
        L0 = list(L0)
        L0[13] = chr(ord(L0[13])-13)
        columns_new.append(''.join(L0))
    data_temp_cam3_h.columns = columns_new

    data_temp_cam3 = pd.concat([data_temp_cam3_h, data_temp_cam3_f], axis=1, sort=False)


    columns_old = data_temp_cam3.columns[2:]
    first_col = 'Third' + data_temp_cam3.columns[0][5:]
    second_col = 'Fourth' + data_temp_cam3.columns[1][6:]
    columns_new = [first_col, second_col]
    for L0 in columns_old: 
        L0 = list(L0)
        L0[13] = chr(ord(L0[13])+5)
        columns_new.append(''.join(L0))
    data_temp_cam3.columns = columns_new

    COLUMNS = data_temp_cam1.columns.tolist()
    for L0 in data_temp_cam3.columns.tolist():
        COLUMNS.append(L0)
    data_temp = pd.concat([data_temp_cam1, data_temp_cam3], axis=1, sort=False)
    data_temp.columns = COLUMNS

    File_Name = './' + DirPath[ DirPath.find('.'+Animal_Name)+1 : ]

    data_temp.to_csv(File_Name)





print(colored("Please wait; Processing new function.",'yellow'))


