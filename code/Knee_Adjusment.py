import cv2, os, sys, math, pdb, argparse, re, subprocess, argparse
import numpy as np
from glob import glob
from copy import deepcopy
from termcolor import colored
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.gridspec as gridspec


parser = argparse.ArgumentParser(description = "Help on how to set the parameters and use short keys while calling the code:")
parser.add_argument("-animal", "--Animal_Name", type=str, help = "The name of animal", default = "rat")
parser.add_argument("-ip", "--Image_Path_On_PC", type=str, help = "Image paths", default = '/Volumes/SCI_Storage')
parser.add_argument("-rl", "--Rat_List", nargs='+', type=str, help = "Number of animals", default = str( 1 ) )
parser.add_argument("-ak1", "--Ankle_Knee_R_L1", type=str, help = "", default = str( np.array([[30, 30, 30, 30]] )) )
parser.add_argument("-hk1", "--Hip_Knee_R_L1", type=str, help = "", default = str( np.array([[34, 34, 34, 34]]) ) )
parser.add_argument("-se1", "--Shoulder_Elbow_R_L1", type=str, help = "", default = str( np.array([[29, 29, 29, 29]]) ) )
parser.add_argument("-eh1", "--Elbow_Hand_R_L1", type=str, help = "", default = str( np.array([[31, 31, 31, 31]]) ) )
parser.add_argument("-ak2", "--Ankle_Knee_R_L2", type=str, help = "", default = str( np.array([[30, 30, 30, 30]]) ) )
parser.add_argument("-hk2", "--Hip_Knee_R_L2", type=str, help = "", default = str( np.array([[34, 34, 34, 34]]) ) )
parser.add_argument("-se2", "--Shoulder_Elbow_R_L2", type=str, help = "", default = str( np.array([[29, 29, 29, 29]]) ) )
parser.add_argument("-eh2", "--Elbow_Hand_R_L2", type=str, help = "", default = str( np.array([[31, 31, 31, 31]]) ) )
parser.add_argument("-pf", "--plot_flag", type=str, help = "Flag for demonstartion and saving", default = '0')
args = parser.parse_args(); ARGS = vars(args)


def Str_to_array(INPUT):
    temp = []
    while INPUT.find(",") > -1: 
        temp.append( int(INPUT[:INPUT.find(",")]) )
        INPUT = INPUT[INPUT.find(",")+1:]
        if INPUT.find(",") == -1:
            temp.append( int(INPUT) )
            break
    temp = np.array(temp)
    return(temp)


Animal_Name = ARGS['Animal_Name']
Rat_List        = Str_to_array(ARGS['Rat_List'][0])
Ankle_Knee_R_L1 = Str_to_array(ARGS['Ankle_Knee_R_L1'])
Hip_Knee_R_L1   = Str_to_array(ARGS['Hip_Knee_R_L1'])
Shoulder_Elbow_R_L1 = Str_to_array(ARGS['Shoulder_Elbow_R_L1'])
Elbow_Hand_R_L1   = Str_to_array(ARGS['Elbow_Hand_R_L1'])
Ankle_Knee_R_L2 = Str_to_array(ARGS['Ankle_Knee_R_L2'])
Hip_Knee_R_L2   = Str_to_array(ARGS['Hip_Knee_R_L2'])
Shoulder_Elbow_R_L2 = Str_to_array(ARGS['Shoulder_Elbow_R_L2'])
Elbow_Hand_R_L2   = Str_to_array(ARGS['Elbow_Hand_R_L2'])
Image_Path_On_PC = ARGS['Image_Path_On_PC']
plot_flag = int(ARGS['plot_flag'])




def Find_Translation_Matrix(Matrix):
    Translation_Matrix = np.array([ [1, 0, 0, -Matrix[0]], [0, 1, 0, -Matrix[1]], [0, 0, 1, -Matrix[2]], [0, 0, 0, 1] ])
    return(Translation_Matrix)



def Find_Rotation_Angles(Matrix):
    Coordinates1 = np.zeros([4,1])
    Coordinates2 = Matrix
    Diff = Coordinates1-Coordinates2
    R = math.sqrt(sum((Diff)**2))
    A = math.acos(Diff[2]/R)
    P = math.atan2(Diff[1], Diff[0])
    return(A, P)



def Creat_Old_New_Points_FromX(Coordinates):
    New_UVW = np.ones([4,1])
    New_UVW[1, :] = 0 # zeros
    New_UVW[2, :] = 0 # zeros
    New_UVW[0, 0] = np.linalg.norm(Coordinates)

    New_XYZ = Coordinates

    Old_XYZ = np.ones([4,1])
    Old_XYZ[1, :] = 0 # zeros
    Old_XYZ[2, :] = 0 # zeros
    Old_XYZ[0, 0] = np.linalg.norm(Coordinates)

    return(Old_XYZ, New_XYZ, New_UVW)



def Creat_Old_New_Points_FromX_Location(Norm_New_Z, Azimuth, Polar):
    New_UVW = np.ones([4,1])
    New_UVW[0, :] = 0 # zeros
    New_UVW[1, :] = 0 # zeros
    New_UVW[2, 0] = 1/(np.sin(Azimuth)*np.cos(Polar))

    New_XYZ = np.ones([4,1])
    New_XYZ[:-1,0] = Norm_New_Z*1

    Old_XYZ = np.ones([4,1])
    Old_XYZ[0, :] = 0 
    Old_XYZ[1, :] = 0 
    Old_XYZ[2, 0] = 1

    return(Old_XYZ, New_XYZ, New_UVW)



def Find_Rotation_Matrix(Old, New):
    Rotation = np.linalg.solve(New, Old)
    Rotation = np.reshape(Rotation, (1,-1))
    Rotation = np.append(Rotation, 0)
    return(Rotation)



def Sphere_Drawing_To_Circle(Rotated_Translated_Hip, Ankle_Knee_R, Hip_Knee_R):
    D = abs(Rotated_Translated_Hip[0])
    if D < Ankle_Knee_R + Hip_Knee_R:
        Radius_Circle = np.sqrt( 4 * Ankle_Knee_R**2 * D**2 - (Ankle_Knee_R**2 + D**2 - Hip_Knee_R**2)**2 ) / (2*D)
        Alpha = np.arcsin(Radius_Circle/Ankle_Knee_R)
        Center_Location = Ankle_Knee_R * np.cos(Alpha)

    else:
        Radius_Circle = 0
        Difference = D - (Ankle_Knee_R + Hip_Knee_R)
        Center_Location = Ankle_Knee_R + Difference/2

    return(Center_Location, Radius_Circle)



def Get_Knee_Point(Center_Location, Radius_Circle, Rotated_Translated_Hip):
    Angles = np.arange(0, 2*np.pi, np.pi/8)
    W = Radius_Circle*np.sin(Angles)
    V = Radius_Circle*np.cos(Angles)
    U = np.tile(Center_Location, len(Angles))
    Extra = np.ones([1,len(Angles)])
    Points = np.concatenate((U.reshape(1,len(Angles)), V.reshape(1,len(Angles)), W.reshape(1,len(Angles)), Extra.reshape(1,len(Angles))), axis = 0)
    return(Points)



def Sphere_Drawing_To_Circle2(Hip_Loc, Ankle_Loc, Ankle_Knee_R, Hip_Knee_R):
    Line_E = Hip_Loc[:-1,:] - Ankle_Loc[:-1,:]
    D = np.linalg.norm(Line_E)
    Line_E = Line_E/D

    if D < Ankle_Knee_R + Hip_Knee_R:
        Radius_Circle = np.sqrt( 4 * Ankle_Knee_R**2 * D**2 - (Ankle_Knee_R**2 + D**2 - Hip_Knee_R**2)**2 ) / (2*D)
        Alpha = np.arcsin(Radius_Circle/Ankle_Knee_R)
        Center_Location = Ankle_Knee_R * np.cos(Alpha)

    else:
        Radius_Circle = 0
        Difference = D - (Ankle_Knee_R + Hip_Knee_R)
        Center_Location = Ankle_Knee_R + Difference/2

    Center_Location *= Line_E
    return(Center_Location, Radius_Circle)



def Get_Knee_Point2(Center_Location, Radius_Circle, Hip_Loc, Ankle_Loc):
    Line_E = Hip_Loc[:-1,:] - Ankle_Loc[:-1,:]
    Line_E = Line_E/np.linalg.norm(Line_E)

    P1 = Center_Location[0] - (Line_E[1]*(1-Center_Location[1]) + Line_E[2]*(1-Center_Location[2]) )/Line_E[0]
    Line1 = np.array([[P1[0], 1, 1]]).T - Center_Location
    Line1 = Line1/np.linalg.norm(Line1)

    Line2 = np.cross(Line_E.reshape(3,), Line1.reshape(3,))
    Line2 = Line2/np.linalg.norm(Line2)

    Angles = np.arange(0, 2*np.pi, np.pi/8)
    Points = np.ones([4,len(Angles)])
    Points[0,:] = Center_Location[0]+ Line1[0] * np.sin(Angles) * Radius_Circle + Line2[0] * np.cos(Angles) * Radius_Circle
    Points[1,:] = Center_Location[1]+ Line1[1] * np.sin(Angles) * Radius_Circle + Line2[1] * np.cos(Angles) * Radius_Circle
    Points[2,:] = Center_Location[2]+ Line1[2] * np.sin(Angles) * Radius_Circle + Line2[2] * np.cos(Angles) * Radius_Circle
    Extra = np.ones([1,len(Angles)])
    return(Points)



def Knee_3D_Circle_Points(Ankle, Knee, Hip, Ankle_Knee_R, Hip_Knee_R): # markers are from top to bottom
    Ankle = np.append(Ankle,1)
    Ankle = np.reshape(Ankle, (4,-1))
    Knee = np.append(Knee,1)
    Knee = np.reshape(Knee, (4,-1))
    Hip = np.append(Hip,1)
    Hip = np.reshape(Hip, (4,-1))


    T = Find_Translation_Matrix(Ankle)
    T_Inv = np.linalg.inv(T)


    Translated_Ankle = np.dot(T, Ankle)
    Translated_Knee = np.dot(T, Knee)
    Translated_Hip = np.dot(T, Hip)


    (Azimuth, Polar) = Find_Rotation_Angles(Translated_Hip)


    Old_XYZ1, New_XYZ1, New_UVW1 = Creat_Old_New_Points_FromX(Translated_Hip)

    New_XYZ4 = deepcopy(New_XYZ1)
    New_XYZ4[2] = New_XYZ4[2]/10

    New_XYZ5 = deepcopy(New_XYZ1)
    New_XYZ5[2] = New_XYZ5[2]/2

    Norm_New_Y = np.cross(New_XYZ1[:-1,0], New_XYZ5[:-1,0] - New_XYZ4[:-1,0])
    Norm_New_Y = Norm_New_Y/np.linalg.norm(Norm_New_Y)
    Old_XYZ2, New_XYZ2, New_UVW2 = Creat_Old_New_Points_FromX_Location(Norm_New_Y, Azimuth, Polar)

    Norm_New_Z = np.cross(New_XYZ1[:-1,0],New_XYZ2[:-1,0])
    Old_XYZ3, New_XYZ3, New_UVW3 = Creat_Old_New_Points_FromX_Location(Norm_New_Z, Azimuth, Polar)

    Norm_New_XYZ1 = np.zeros([4,1])
    Norm_New_XYZ2 = np.zeros([4,1])
    Norm_New_XYZ3 = np.zeros([4,1])
    Norm_New_XYZ1[:-1, 0] = New_XYZ1[:-1, 0]/np.linalg.norm(New_XYZ1[:-1, 0])
    Norm_New_XYZ2[:-1, 0] = New_XYZ2[:-1, 0]/np.linalg.norm(New_XYZ2[:-1, 0])
    Norm_New_XYZ3[:-1, 0] = New_XYZ3[:-1, 0]/np.linalg.norm(New_XYZ3[:-1, 0])

    (Center_Location, Radius_Circle) = Sphere_Drawing_To_Circle2(Translated_Hip, Translated_Ankle, Ankle_Knee_R, Hip_Knee_R)
    if Radius_Circle == 0:
        XYZ_Points_Translated = Knee[:-1,:]
    else:
        Points = Get_Knee_Point2(Center_Location, Radius_Circle, Translated_Hip, Translated_Ankle)
        XYZ_Points_Translated = np.dot(T_Inv, Points)
        XYZ_Points_Translated = XYZ_Points_Translated[:-1,:]

    return (XYZ_Points_Translated)



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



def DLT_Inverse(xyz, Coef, Cam_number): # This function finds the 2D projection on each camera using 3D coordinates and DLT coeffs
    Position = np.zeros([4])
    
    if Cam_number == 1:
        Cam1_Coe = Coef[..., int(0)]; Cam1_Coe = np.append(Cam1_Coe, 1); 
        Cam2_Coe = Coef[..., int(1)]; Cam2_Coe = np.append(Cam2_Coe, 1);
    else:
        Cam1_Coe = Coef[..., int(2)]; Cam1_Coe = np.append(Cam1_Coe, 1); 
        Cam2_Coe = Coef[..., int(3)]; Cam2_Coe = np.append(Cam2_Coe, 1);

    Coe = []
    Coe = [Cam1_Coe.reshape((3,4)),Cam2_Coe.reshape((3,4))]

    temp = Coe[0].dot(np.append(xyz, 1))
    if Cam_number == 3:
        Position[1] = 2048 - float(temp[0]/temp[2])
    else:
        Position[1] = float(temp[0]/temp[2])
    Position[0] = 700 - float(temp[1]/temp[2])

    temp = Coe[1].dot(np.append(xyz, 1))
    if Cam_number == 3:
        Position[3] = 2048 - float(temp[0]/temp[2])
    else:
        Position[3] = float(temp[0]/temp[2])
    Position[2] = 700 - float(temp[1]/temp[2])

    return(Position)



def Save_CSV(Final_Results, Header, Path):
    Temp = [Header[0],
            Header[1],
            Header[2],
            Header[3],
            Header[4],
            '',
            ]
    header = '\n'.join([line for line in Temp])

    try:
        os.remove(Path)
    except OSError:
        pass

    File_Name = Path[2:]
    with open(File_Name, 'wt') as CSV:
        for line in header:
            CSV.write(line)
        Final_Results.to_csv(CSV)


def Find_Cameras_Path(data_C2D, Image_Path_On_PC):
    Image_Path = data_C2D['First Camera Frames Name'][0]
    Lower_Path = Image_Path.lower()

    Image_Path = Image_Path[:len(Image_Path)-Image_Path[-1::-1].find('/')]

    Study_Loc_PC = len(Image_Path_On_PC) - Image_Path_On_PC[-1::-1].find('/') -1
    Study_Name_found = Image_Path_On_PC[Study_Loc_PC:].lower()
    Study_Loc = Lower_Path.find(Study_Name_found)
    Image_Path1 = Image_Path_On_PC[:Study_Loc_PC] + Image_Path[Study_Loc:]
    
    Image_Path2 = deepcopy(Image_Path1)
    Cam_Loc = re.search('Cam', Image_Path2, re.IGNORECASE).start() 
    Image_Path2 = list(Image_Path2)
    Image_Path2[Cam_Loc+3] = '2'
    Image_Path2 = ''.join(Image_Path2)

    Image_Path3 = list(Image_Path1)
    Image_Path3[Cam_Loc+3] = '3'
    Image_Path3 = ''.join(Image_Path3)

    Image_Path4 = list(Image_Path1)
    Image_Path4[Cam_Loc+3] = '4'
    Image_Path4 = ''.join(Image_Path4)

    return(Image_Path1 ,Image_Path2, Image_Path3, Image_Path4)


def Load_Image(Image_Bank):
    Image = cv2.imread(Image_Bank)
    b, g, r = cv2.split(Image)
    Image = cv2.merge((r, g, b))
    return(Image)



def Image_Add_Markers(Image, Coord, Color, Cam_num):
    for D0, L0 in enumerate(Coord):
        Circle_Size = 12
        if Cam_num == 1:
            Image = cv2.circle(Image, (int(L0[1]), int(L0[0])), Circle_Size, Color, -1)
        else:
            Image = cv2.circle(Image, (2048-int(L0[1]), int(L0[0])), Circle_Size, Color, -1)
    return(Image)


def Labeling(Data_temp, Cam_index):
    shift = 0
    if Cam_index == '2':
        shift = int(Data_temp.shape[0]/2)

    List = ['Hip', 'Knee', 'Ankle', 'Shoulder', 'Elbow', 'Hand']
    Coor_List = ['_x', '_y', '_z']
    Selected_List = []
    for L in List:
        for L1 in Coor_List:
            Selected_List.append(L+ Cam_index +L1)

    Hip_List_2D = Data_temp[0+shift:4+shift]
    Hip_list_3D = []
    for L0 in Coor_List: Hip_list_3D.append(List[0]+ Cam_index +L0)

    Knee_List_2D = Data_temp[4+shift:8+shift]
    Knee_list_3D = []
    for L0 in Coor_List: Knee_list_3D.append(List[1]+ Cam_index +L0)

    Ankle_List_2D = Data_temp[8+shift:12+shift]
    Ankle_list_3D = []
    for L0 in Coor_List: Ankle_list_3D.append(List[2]+ Cam_index +L0)

    Shoulder_List_2D = Data_temp[12+shift:16+shift]
    Shoulder_list_3D = []
    for L0 in Coor_List: Shoulder_list_3D.append(List[3]+ Cam_index +L0)

    Elbow_List_2D = Data_temp[16+shift:20+shift]
    Elbow_list_3D = []
    for L0 in Coor_List: Elbow_list_3D.append(List[4]+ Cam_index +L0)

    Hand_List_2D = Data_temp[20+shift:24+shift]
    Hand_list_3D = []
    for L0 in Coor_List: Hand_list_3D.append(List[5]+ Cam_index +L0)

    return(Hand_list_3D, Hand_List_2D, Elbow_list_3D, Elbow_List_2D, Shoulder_list_3D, Shoulder_List_2D,
        Ankle_list_3D, Ankle_List_2D, Knee_list_3D, Knee_List_2D, Hip_list_3D, Hip_List_2D, Selected_List)

    

def Main_Processing(row, index, Selected_List, Start, Main_list_3D, Main_list_2D, Bottom_Main_R, Top_Main_R, DLT_Coef, Cam_num, data_temp, Data, K_E):
    if K_E == 0:
        XYZ_Points_On_Circle = Knee_3D_Circle_Points(row[Selected_List[Start+0:Start+3]].values, row[Selected_List[Start+3:Start+6]].values, row[Selected_List[Start+6:Start+9]].values, Top_Main_R, Bottom_Main_R)
    else:
        XYZ_Points_On_Circle = Knee_3D_Circle_Points(row[Selected_List[Start+0:Start+3]].values, row[Selected_List[Start+3:Start+6]].values, row[Selected_List[Start+6:Start+9]].values, Top_Main_R, Bottom_Main_R)

    # if not(np.isnan(XYZ_Points_On_Circle).all()):
    temp = deepcopy(XYZ_Points_On_Circle[0])
    TEMP = deepcopy(XYZ_Points_On_Circle)
    if K_E == 0:
        if np.sign(DLT_Coef[2,0]) > 0:
            temp[abs(temp)<np.max(abs(temp))-(np.max(abs(temp))-np.min(abs(temp)))/8] = 10000
        else:
            temp[abs(temp)>np.min(abs(temp))+(np.max(abs(temp))-np.min(abs(temp)))/8] = 10000
    else:
        if np.sign(DLT_Coef[2,0]) > 0:
            temp[abs(temp)>np.min(abs(temp))+(np.max(abs(temp))-np.min(abs(temp)))/8] = 10000
        else:
            temp[abs(temp)<np.max(abs(temp))-(np.max(abs(temp))-np.min(abs(temp)))/8] = 10000

    TEMP[0] = temp

    Difference_3D = TEMP.T - row[Selected_List[Start+3:Start+6]].values
    Difference_3D = np.sum(Difference_3D**2, axis =1)
    Index_Best_Match = np.argmin(Difference_3D)

    Best_3D_Point = XYZ_Points_On_Circle[:,Index_Best_Match]
    Best_2D_Coordinates = DLT_Inverse(Best_3D_Point, DLT_Coef, Cam_num)

    Data.at[index, Main_list_3D[0]] = Best_3D_Point[0]
    Data.at[index, Main_list_3D[1]] = Best_3D_Point[1]
    Data.at[index, Main_list_3D[2]] = Best_3D_Point[2]

    data_temp.at[index, Main_list_2D] = Best_2D_Coordinates

    return(data_temp, XYZ_Points_On_Circle, Best_3D_Point, Data, Best_2D_Coordinates)


def DLT_Inverse_For_Circle(Loaded_Cordinates_2D, XYZ_Points_On_Circle, DLT_Coef, Cam_Num):
    Coordinates_Points_On_Circle = []
    for xyz in XYZ_Points_On_Circle.T:
        Coordinates_Points_On_Circle = np.append(DLT_Inverse(xyz, DLT_Coef, Cam_Num), Coordinates_Points_On_Circle)
    Coordinates_Points_On_Circle = Coordinates_Points_On_Circle.reshape(-1,4)

    return(Coordinates_Points_On_Circle)


def Plot_Sphere(Bottom_Main_R, Top_Main_R, row, Selected_List, ax, Start, Best_3D_Point, XYZ_Points_On_Circle):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x1 = Bottom_Main_R * np.outer(np.cos(u), np.sin(v)) + row[Selected_List[6+Start]]
    y1 = Bottom_Main_R * np.outer(np.sin(u), np.sin(v)) + row[Selected_List[7+Start]]
    z1 = Bottom_Main_R * np.outer(np.ones(np.size(u)), np.cos(v)) + row[Selected_List[8+Start]]
    ax.plot_surface(x1, y1, z1,  color='b', linewidth=0, alpha=0.2)

    x2 = Top_Main_R * np.outer(np.cos(u), np.sin(v)) + row[Selected_List[0+Start]]
    y2 = Top_Main_R * np.outer(np.sin(u), np.sin(v)) + row[Selected_List[1+Start]]
    z2 = Top_Main_R * np.outer(np.ones(np.size(u)), np.cos(v)) + row[Selected_List[2+Start]]
    ax.plot_surface(x2, y2, z2,  color='g', linewidth=0, alpha=0.2)

    Bottom = Selected_List[0+Start][:Selected_List[0+Start].find('_')]
    Main = Selected_List[3+Start][:Selected_List[3+Start].find('_')]
    Top = Selected_List[6+Start][:Selected_List[6+Start].find('_')]

    ax.scatter(row[Selected_List[6+Start]], row[Selected_List[7+Start]], row[Selected_List[8+Start]], color='b', label=Top, alpha=1, s=60) # Tracked Ankle
    ax.scatter(row[Selected_List[0+Start]], row[Selected_List[1+Start]], row[Selected_List[2+Start]], color='g', label=Bottom, alpha=1, s=60) # Tracked Hip
    ax.scatter(row[Selected_List[3+Start]], row[Selected_List[4+Start]], row[Selected_List[5+Start]], color='r', label=Main, alpha=1, s=60) # Tracked Knee
    ax.scatter(Best_3D_Point[0], Best_3D_Point[1], Best_3D_Point[2], color='k', label= Main+' Selected', alpha=1, s=60) # Tracked Knee
    ax.scatter(XYZ_Points_On_Circle.T[:,0], XYZ_Points_On_Circle.T[:,1], XYZ_Points_On_Circle.T[:,2], color='c', label= Main+' Predictions', alpha=1, s=20, depthshade = False)
    ax.set_xlabel('X Label'); ax.set_ylabel('Y Label'); ax.set_zlabel('Z Label'); 



def Marked_Image(Image, ax, data_temp, Temp_data, Coordinates_Points_On_Circle, Best_2D_Coordinates, index, Image_num, Bottom_List_2D, Top_List_2D, Main_List_2D, Cam_Num):
    Image = Image_Add_Markers(Image, data_temp[Bottom_List_2D[0+Image_num:2+Image_num]].iloc[index].values.reshape(-1,2), (0,0,255), Cam_Num)
    Image = Image_Add_Markers(Image, data_temp[Top_List_2D[0+Image_num:2+Image_num]].iloc[index].values.reshape(-1,2), (0,255,0), Cam_Num)
    Image = Image_Add_Markers(Image, Temp_data[Main_List_2D[0+Image_num:2+Image_num]].iloc[index].values.reshape(-1,2), (255,0,0), Cam_Num)
    Image = Image_Add_Markers(Image, Coordinates_Points_On_Circle[:, 0+Image_num:2+Image_num], (0,255,255), Cam_Num)
    Image = Image_Add_Markers(Image, Best_2D_Coordinates.reshape(-1,4)[:, 0+Image_num:2+Image_num], (0,0,0), Cam_Num)
    ax.imshow(Image)









filt_order = 3; sample_rate = 250.0; filt_freq = 20;
filt_param = filt_freq/sample_rate; filt_type = 'low';
[B,A] = butter(filt_order,filt_param,filt_type);


CurrentPath = "."
Deleting_Path = sorted(glob(os.path.join(CurrentPath, "Adjusted."+ Animal_Name + "*.csv")))

for L in Deleting_Path: 
    os.remove(L)
Deleting_Path = sorted(glob(os.path.join(CurrentPath, "Image*.png")))
for L in Deleting_Path: 
    os.remove(L)

Coordinate_Path = sorted(glob(os.path.join(CurrentPath, "Coord_3D." + Animal_Name + "*.csv")))

Adjusted_Path = deepcopy(Coordinate_Path)
DLT_Path = deepcopy(Coordinate_Path)
Coordinates_2D_Path = deepcopy(Coordinate_Path)


count = 0
for L in Coordinate_Path: 
    Adjusted_Path[count] = L[0:2] + 'Adjusted.' + L[11:]
    DLT_Path[count] = L[0:2] + 'DLT.' + L[11:]
    Coordinates_2D_Path[count] = L[11:]
    count += 1


for (DirPath, Adjustedpath, DLTpath, CoorPath) in zip(Coordinate_Path, Adjusted_Path, DLT_Path, Coordinates_2D_Path):
    print("Processing dir %s   \r" %(DirPath))
    Data = []


    Rat_Num_Pos = DirPath.find('.'+ Animal_Name)
    Length_Rat_Name = DirPath[Rat_Num_Pos+1:].find('.')
    if Length_Rat_Name == 4:
        Rat_Num = DirPath[Rat_Num_Pos+4]
    elif Length_Rat_Name == 5:
        Rat_Num = DirPath[Rat_Num_Pos+4:Rat_Num_Pos+6]
    Rat_Num_Pos = np.where(Rat_List == int(Rat_Num))[0]

    DLT_Coef = np.genfromtxt(DLTpath, delimiter=',')


    Ankle_Knee_R1 = int(Ankle_Knee_R_L1[Rat_Num_Pos])
    Hip_Knee_R1 = int(Hip_Knee_R_L1[Rat_Num_Pos])
    Shoulder_Elbow_R1 = int(Shoulder_Elbow_R_L1[Rat_Num_Pos])
    Elbow_Hand_R1 = int(Elbow_Hand_R_L1[Rat_Num_Pos])

    Ankle_Knee_R2 = int(Ankle_Knee_R_L2[Rat_Num_Pos])
    Hip_Knee_R2 = int(Hip_Knee_R_L2[Rat_Num_Pos])
    Shoulder_Elbow_R2 = int(Shoulder_Elbow_R_L2[Rat_Num_Pos])
    Elbow_Hand_R2 = int(Elbow_Hand_R_L2[Rat_Num_Pos])

    Data = pd.read_csv(DirPath, sep=',', index_col=0)
    xyz_coor = Data[ Data.columns[ 1: ] ].values
    Frame_Number = Data[ Data.columns[0] ].values
    Num_Markers = int(round(xyz_coor.shape[1]/3))
    All_Columns_3D = Data.columns

    data_temp = pd.read_csv(CoorPath, sep=',', index_col=0)
    All_Columns_2D = data_temp.columns
    Just_Numbers = All_Columns_2D[2:int((data_temp.shape[1])/2)].append(All_Columns_2D[2+int((data_temp.shape[1])/2):])
    Loaded_Cordinates_2D = data_temp[ Just_Numbers[ np.r_[4:16, 20:32, 36:48, 52:64] ] ].values
    Loaded_Labels_2D = data_temp[ Just_Numbers[ np.r_[4:16, 20:32, 36:48, 52:64] ] ].columns


    if plot_flag == 1:
        Image_Path1, Image_Path2, Image_Path3, Image_Path4 = Find_Cameras_Path(data_temp, Image_Path_On_PC)
        Image_Bank1 = sorted(glob(os.path.join(Image_Path1, "*.png")))
        Image_Bank2 = sorted(glob(os.path.join(Image_Path2, "*.png")))
        Image_Bank3 = sorted(glob(os.path.join(Image_Path3, "*.png")))
        Image_Bank4 = sorted(glob(os.path.join(Image_Path4, "*.png")))
    

    Cam_index = '1'
    (Hand1_list_3D, Hand1_List_2D, Elbow1_list_3D, Elbow1_List_2D, Shoulder1_list_3D, Shoulder1_List_2D,
        Ankle1_list_3D, Ankle1_List_2D, Knee1_list_3D, Knee1_List_2D, Hip1_list_3D, Hip1_List_2D, Selected1_List) = Labeling(Loaded_Labels_2D, Cam_index)


    Cam_index = '2'
    (Hand2_list_3D, Hand2_List_2D, Elbow2_list_3D, Elbow2_List_2D, Shoulder2_list_3D, Shoulder2_List_2D,
        Ankle2_list_3D, Ankle2_List_2D, Knee2_list_3D, Knee2_List_2D, Hip2_list_3D, Hip2_List_2D, Selected2_List) = Labeling(Loaded_Labels_2D, Cam_index)


    Selected_List = deepcopy(Selected1_List)
    for L0 in Selected2_List: Selected_List.append(L0)
    Selected_Data = Data[Selected_List]

    Temp_data = deepcopy(data_temp)

    for index, row in Selected_Data.iterrows():

        Start = 0; Cam_num = 1; K_E = 0
        (data_temp, XYZ_Points_On_Circle_K1, Best_3D_Point_K1, Data, Best_2D_Point_K1) = Main_Processing(row, index, Selected_List, Start, Knee1_list_3D, Knee1_List_2D, Ankle_Knee_R1, Hip_Knee_R1, DLT_Coef, Cam_num, data_temp, Data, K_E)

        Start = 9; Cam_num = 1; K_E = 1
        (data_temp, XYZ_Points_On_Circle_E1, Best_3D_Point_E1, Data, Best_2D_Point_E1) = Main_Processing(row, index, Selected_List, Start, Elbow1_list_3D, Elbow1_List_2D, Elbow_Hand_R1, Shoulder_Elbow_R1, DLT_Coef, Cam_num, data_temp, Data, K_E)

        Start = 18; Cam_num = 3; K_E = 0
        (data_temp, XYZ_Points_On_Circle_K2, Best_3D_Point_K2, Data, Best_2D_Point_K2) = Main_Processing(row, index, Selected_List, Start, Knee2_list_3D, Knee2_List_2D, Ankle_Knee_R2, Hip_Knee_R2, DLT_Coef, Cam_num, data_temp, Data, K_E)

        Start = 27; Cam_num = 3; K_E = 1
        (data_temp, XYZ_Points_On_Circle_E2, Best_3D_Point_E2, Data, Best_2D_Point_E2) = Main_Processing(row, index, Selected_List, Start, Elbow2_list_3D, Elbow2_List_2D, Elbow_Hand_R2, Shoulder_Elbow_R2, DLT_Coef, Cam_num, data_temp, Data, K_E)


        if plot_flag == 1:
            print("Joinr Adjustment; Processing frame number %s   \r" %(str(index)))

            Image1 = Load_Image(Image_Bank1[index])
            Image2 = Load_Image(Image_Bank2[index])
            Image3 = Load_Image(Image_Bank3[index])
            Image4 = Load_Image(Image_Bank4[index])


            Cam_Num = 1
            Coordinates_Points_On_Circle_K1 = DLT_Inverse_For_Circle(Loaded_Cordinates_2D, XYZ_Points_On_Circle_K1, DLT_Coef, Cam_Num)
            Coordinates_Points_On_Circle_E1 = DLT_Inverse_For_Circle(Loaded_Cordinates_2D, XYZ_Points_On_Circle_E1, DLT_Coef, Cam_Num)

            Cam_Num = 3
            Coordinates_Points_On_Circle_K2 = DLT_Inverse_For_Circle(Loaded_Cordinates_2D, XYZ_Points_On_Circle_K2, DLT_Coef, Cam_Num)
            Coordinates_Points_On_Circle_E2 = DLT_Inverse_For_Circle(Loaded_Cordinates_2D, XYZ_Points_On_Circle_E2, DLT_Coef, Cam_Num)

            
            fig = plt.figure()
            fig.set_size_inches(20, 15)
            G = gridspec.GridSpec(2, 1)

            G0 = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=G[0])

            ax = fig.add_subplot(G0[:2,0], projection='3d')
            # ax.set_aspect('equal')
            Start = 0
            Plot_Sphere(Ankle_Knee_R1, Hip_Knee_R1, row, Selected_List, ax, Start, Best_3D_Point_K1, XYZ_Points_On_Circle_K1)
            ax.legend(bbox_to_anchor=(1.2, 1), loc=2, ncol=1, borderaxespad=0., fontsize=8)

            ax = fig.add_subplot(G0[:2,1], projection='3d')
            # ax.set_aspect('equal')
            Start = 9
            Plot_Sphere(Elbow_Hand_R1, Shoulder_Elbow_R1, row, Selected_List, ax, Start, Best_3D_Point_E1, XYZ_Points_On_Circle_E1)
            ax.legend(bbox_to_anchor=(1.2, 1), loc=2, ncol=1, borderaxespad=0., fontsize=8)

            
            ax = fig.add_subplot(G0[2,0]); 
            Image_Num = 0; Cam_Num = 1
            Marked_Image(Image1, ax, data_temp, Temp_data, Coordinates_Points_On_Circle_K1, Best_2D_Point_K1, index, Image_Num, Ankle1_List_2D, Hip1_List_2D, Knee1_List_2D, Cam_Num)
            Marked_Image(Image1, ax, data_temp, Temp_data, Coordinates_Points_On_Circle_E1, Best_2D_Point_E1, index, Image_Num, Hand1_List_2D, Shoulder1_List_2D, Elbow1_List_2D, Cam_Num)
            
            ax = fig.add_subplot(G0[2,1]); 
            Image_Num = 2; Cam_Num = 1
            Marked_Image(Image2, ax, data_temp, Temp_data, Coordinates_Points_On_Circle_K1, Best_2D_Point_K1, index, Image_Num, Ankle1_List_2D, Hip1_List_2D, Knee1_List_2D, Cam_Num)
            Marked_Image(Image2, ax, data_temp, Temp_data, Coordinates_Points_On_Circle_E1, Best_2D_Point_E1, index, Image_Num, Hand1_List_2D, Shoulder1_List_2D, Elbow1_List_2D, Cam_Num)


            
            G0 = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=G[1])

            ax = fig.add_subplot(G0[:2,0], projection='3d')
            # ax.set_aspect('equal')
            Start = 18
            Plot_Sphere(Ankle_Knee_R2, Hip_Knee_R2, row, Selected_List, ax, Start, Best_3D_Point_K2, XYZ_Points_On_Circle_K2)
            ax.legend(bbox_to_anchor=(1.2, 1), loc=2, ncol=1, borderaxespad=0., fontsize=8)

            ax = fig.add_subplot(G0[:2,1], projection='3d')
            # ax.set_aspect('equal')
            Start = 27
            Plot_Sphere(Elbow_Hand_R2, Shoulder_Elbow_R2, row, Selected_List, ax, Start, Best_3D_Point_E2, XYZ_Points_On_Circle_E2)
            ax.legend(bbox_to_anchor=(1.2, 1), loc=2, ncol=1, borderaxespad=0., fontsize=8)

            
            ax = fig.add_subplot(G0[2,0]); 
            Image_Num = 0; Cam_Num = 3
            Marked_Image(Image3, ax, data_temp, Temp_data, Coordinates_Points_On_Circle_K2, Best_2D_Point_K2, index, Image_Num, Ankle2_List_2D, Hip2_List_2D, Knee2_List_2D, Cam_Num)
            Marked_Image(Image3, ax, data_temp, Temp_data, Coordinates_Points_On_Circle_E2, Best_2D_Point_E2, index, Image_Num, Hand2_List_2D, Shoulder2_List_2D, Elbow2_List_2D, Cam_Num)
            
            ax = fig.add_subplot(G0[2,1]); 
            Image_Num = 2; Cam_Num = 3
            Marked_Image(Image4, ax, data_temp, Temp_data, Coordinates_Points_On_Circle_K2, Best_2D_Point_K2, index, Image_Num, Ankle2_List_2D, Hip2_List_2D, Knee2_List_2D, Cam_Num)
            Marked_Image(Image4, ax, data_temp, Temp_data, Coordinates_Points_On_Circle_E2, Best_2D_Point_E2, index, Image_Num, Hand2_List_2D, Shoulder2_List_2D, Elbow2_List_2D, Cam_Num)


            plt.savefig("Image%05d.png" %(index), dpi=300)
            plt.close()


    Columns = Data.columns[1:]
    if np.sign(DLT_Coef[2,0]) < 0:
        Data[Columns[0::3]] *= -1
        Data[Columns[1::3]] *= -1 
    # Data[Columns[1::3]] *= 0

    for L0 in Elbow1_list_3D: Data[L0] = filtfilt(B, A, Data[L0])
    for L0 in Elbow1_List_2D: data_temp[L0] = filtfilt(B, A, data_temp[L0])
    for L0 in Elbow2_list_3D: Data[L0] = filtfilt(B, A, Data[L0])
    for L0 in Elbow2_List_2D: data_temp[L0] = filtfilt(B, A, data_temp[L0])

    for L0 in Knee1_list_3D: Data[L0] = filtfilt(B, A, Data[L0])
    for L0 in Knee1_List_2D: data_temp[L0] = filtfilt(B, A, data_temp[L0])
    for L0 in Knee2_list_3D: Data[L0] = filtfilt(B, A, Data[L0])
    for L0 in Knee2_List_2D: data_temp[L0] = filtfilt(B, A, data_temp[L0])

    data_temp.to_csv(CoorPath)
    Data.to_csv(DirPath)


print(colored("Process has been completely DONE.",'green'))