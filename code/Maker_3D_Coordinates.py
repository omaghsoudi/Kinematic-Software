import cv2, os, sys, math, argparse, pdb
import numpy as np
import pandas as pd
from shutil import copyfile
from glob import glob
from copy import deepcopy
from termcolor import colored
from scipy.signal import butter, filtfilt
from sympy import *



parser = argparse.ArgumentParser(description = "Help on how to set the parameters and use short keys while calling the code:")
parser.add_argument("-animal", "--Animal_Name", type=str, help = "The name of animal", default = "rat")
args = parser.parse_args(); ARGS = vars(args)

Paw_Name = 'f'
Animal_Name = ARGS['Animal_Name']
Input_Name = Paw_Name+"."+Animal_Name



def DLT(Position, Coef, Cam_number):
    if Cam_number == 1:
        Cam1_Coe = Coef[...,int(0)]; Cam1_Coe = np.append(Cam1_Coe,-1); 
        Cam2_Coe = Coef[...,int(1)]; Cam2_Coe = np.append(Cam2_Coe,-1);
    else:
        Cam1_Coe = Coef[...,int(2)]; Cam1_Coe = np.append(Cam1_Coe,-1); 
        Cam2_Coe = Coef[...,int(3)]; Cam2_Coe = np.append(Cam2_Coe,-1);
    Coe = []
    Coe = [Cam1_Coe.reshape((3,4)),Cam2_Coe.reshape((3,4))]

    U = np.zeros([1,2]);V = np.zeros([1,2])
    V[0,0] = 700 - Position[1]; V[0,1] = 700 - Position[3]

    if Cam_number == 3:
        U[0,0] = 2048 - Position[0]; U[0,1] = 2048 - Position[2]
    else:
        U[0,0] = Position[0]; U[0,1] = Position[2]

    m1 = np.zeros([4,3]); m2 = np.zeros([4,1])
    m1[0::2,0] = [U[0,0]*Cam1_Coe[8]-Cam1_Coe[0],  U[0,1]*Cam2_Coe[8]-Cam2_Coe[0]]
    m1[0::2,1] = [U[0,0]*Cam1_Coe[9]-Cam1_Coe[1],  U[0,1]*Cam2_Coe[9]-Cam2_Coe[1]]
    m1[0::2,2] = [U[0,0]*Cam1_Coe[10]-Cam1_Coe[2], U[0,1]*Cam2_Coe[10]-Cam2_Coe[2]]

    m1[1::2,0] = [V[0,0]*Cam1_Coe[8]-Cam1_Coe[4],  V[0,1]*Cam2_Coe[8]-Cam2_Coe[4]]
    m1[1::2,1] = [V[0,0]*Cam1_Coe[9]-Cam1_Coe[5],  V[0,1]*Cam2_Coe[9]-Cam2_Coe[5]]
    m1[1::2,2] = [V[0,0]*Cam1_Coe[10]-Cam1_Coe[6], V[0,1]*Cam2_Coe[10]-Cam2_Coe[6]]

    m2[0::2,0] = [Cam1_Coe[3]-U[0,0], Cam2_Coe[3]-U[0,1]]
    m2[1::2,0] = [Cam1_Coe[7]-V[0,0], Cam2_Coe[7]-V[0,1]]

    xyz, res, rank, sdv = np.linalg.lstsq(m1,m2, rcond=None)
    return(xyz)


def Filter(Loaded_Cordinates):
    Num_Markers = int(round(Loaded_Cordinates1.shape[1]/4))

    for counter in range(Loaded_Cordinates.shape[1]):
        Current_Marker_Num = counter/4
        if Current_Marker_Num < 2 or (Current_Marker_Num>=5 and Current_Marker_Num<6):
            Loaded_Cordinates[:,counter] = filtfilt(B2, A2, Loaded_Cordinates[:,counter])
        else:
            Loaded_Cordinates[:,counter] = filtfilt(B, A, Loaded_Cordinates[:,counter])

    return(Loaded_Cordinates)



def DLT_Transform(Loaded_Cordinates, Num_Markers, Markers_Label, data_temp, B, A, Cam_number):
    Num_Rows = Loaded_Cordinates.shape[0]
    xyz = np.zeros([Num_Rows, Num_Markers*3])
    Position = np.zeros(4,)
    Temp = np.zeros(4,)

    Labels = ['Frame Number']
    C = 0
    for NM, ML in zip(range(Num_Markers), Markers_Label):
        Frame_Numbers = []
        for NR in range(Num_Rows):
            Trial_Name = data_temp['First Camera Frames Name'].iloc[NR]
            Frame_Numbers.append(int(Trial_Name[-9:-4]))


            Temp[0:] = Loaded_Cordinates[NR, NM*4:NM*4+4]
            Position[0] = Temp[1]; Position[1] = Temp[0]; Position[2] = Temp[3]; Position[3] = Temp[2]
            xyz[NR,C*3: C*3+3] = DLT(Position, DLT_Coef, Cam_number).transpose()

        for CL in Coordinate_Label:
            Labels.append(ML+'_'+CL)

        C += 1


    for counter in range(xyz.shape[1]): xyz[:,counter] = filtfilt(B, A, xyz[:,counter])

    df = pd.DataFrame(data = xyz, columns = Labels[1:])
    df[Labels[0]] = pd.Series(data = Frame_Numbers, index = df.index)
    df = df[Labels]

    return(df)



    

Num_Markers = 8
CurrentPath = "."
Coordinate_Path = sorted(glob(os.path.join(CurrentPath, Animal_Name+"*.csv")))
DLT_Path = deepcopy(Coordinate_Path)
C3D_Output_Path = deepcopy(Coordinate_Path)
count = 0
for L in Coordinate_Path:
    DLT_Path[count] = L[0:2] + 'DLT.' + L[2:]
    C3D_Output_Path[count] = L[0:2] + 'Coord_3D.' + L[2:]
    count += 1


filt_order = 3; sample_rate = 250.0; filt_freq = 20;
filt_param = filt_freq/sample_rate; filt_type = 'low';
[B,A] = butter(filt_order,filt_param,filt_type);


filt_order = 3; sample_rate = 250.0; filt_freq = 10;
filt_param = filt_freq/sample_rate; filt_type = 'low';
[B2,A2] = butter(filt_order,filt_param,filt_type);


Coordinate_Label = ['x', 'y', 'z']
Markers_Label = ['Asis', 'Hip', 'Knee', 'Ankle', 'Toe', 'Shoulder', 'Elbow', 'Hand']
Markers_Label1 = []
for L0 in Markers_Label: Markers_Label1.append(L0+'1')
Markers_Label2 = []
for L0 in Markers_Label: Markers_Label2.append(L0+'2')

for (DirPath, DLTPaths, C3DPaths) in zip(Coordinate_Path, DLT_Path, C3D_Output_Path):
    print("Making 3D coordinates; PLEASE WAIT, Processing dir %s   \r" %(DirPath))

    Error_Flag = 0
    if not(os.path.exists(DirPath)):
        print(colored("Error: The Tracked Coordinates do not exist in %s \r" %(DirPath),'red'))
        Error_Flag = 1
    if not(os.path.exists(DirPath)):
        print(colored("Error: The DLT Coefficients do not exist in %s \r" %(DirPath),'red'))
        Error_Flag = 1


    if Error_Flag == 0:
        data_temp = pd.read_csv(DirPath, sep=',', index_col=0)
        index = pd.isnull(data_temp).any(1).nonzero()[0]
        data_temp = data_temp.drop(index)
        
        Loaded_Cordinates1 = data_temp.iloc[:, 2: int(data_temp.shape[1]/2)].values
        Loaded_Cordinates2 = data_temp.iloc[:, int(data_temp.shape[1]/2)+2:].values


        Loaded_Cordinates1 = Filter(Loaded_Cordinates1)
        Loaded_Cordinates2 = Filter(Loaded_Cordinates2)
            

        DLT_Coef = np.genfromtxt(DLTPaths, delimiter=',')

        # if DirPath == "./rat2.week0.2018-06-08_13_29_13.csv":
        #     pdb.set_trace()

        


        Cam_number = 1
        xyz1 = DLT_Transform(Loaded_Cordinates1, Num_Markers, Markers_Label1, data_temp, B, A, Cam_number)

        Cam_number = 3
        xyz2 = DLT_Transform(Loaded_Cordinates2, Num_Markers, Markers_Label2, data_temp, B, A, Cam_number)

        df = pd.concat([xyz1, xyz2[xyz2.columns[1:]]], axis=1, sort=False)
        df.to_csv(C3DPaths)



print(colored("Process has been completely DONE.",'green'))