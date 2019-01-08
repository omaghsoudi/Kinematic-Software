import cv2, os, sys, math, pdb, re, subprocess, argparse
import numpy as np
from glob import glob
from copy import deepcopy
import pandas as pd
from termcolor import colored
import matplotlib
if sys.platform == 'linux' or sys.platform == 'linux2':
    matplotlib.use("TkAgg")
else:
    matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import butter, filtfilt
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
from termcolor import colored


parser = argparse.ArgumentParser(description = "Help on how to set the parameters and use short keys while calling the code:")
parser.add_argument("-animal", "--Animal_Name", type=str, help = "The name of animal", default = "rat")
parser.add_argument("-bins", "--Number_Bins", type=str, help = "Number of bins for plotting", default = str(200))
parser.add_argument("-ip", "--Image_Path_On_PC", type=str, help = "Image paths", default = '/Volumes/SCI_Storage/Aging')
parser.add_argument("-separate", "--Separate_Cam1_Cam3", type=str, help = "If separate is 1 then each camera will be considered as a condition", default = str(0))
parser.add_argument("-nf", "--Number_Frames", type=str, help = "Number of frames", default = str(1000))
parser.add_argument("-hsr", "--Higher_Speed_Range", type=str, help = "Higher Speed Range", default = str(0.6))
parser.add_argument("-lsr", "--Lower_Speed_Range", type=str, help = "Lower Speed Range", default = str(-0.4))
parser.add_argument("-ta", "--Time_All", type=str, help = "Time for all frames in seconds", default = str(4))
parser.add_argument("-rx", "--Range_X", type=str, help = "Time for all frames in seconds", default = str(200))
parser.add_argument("-ry", "--Range_Y", type=str, help = "Time for all frames in seconds", default = str(50))
parser.add_argument("-rz", "--Range_Z", type=str, help = "Time for all frames in seconds", default = str(50))
args = parser.parse_args(); ARGS = vars(args)

global Number_Frames, Higher_Speed_Range, Lower_Speed_Range, Time_All

Animal_Name = ARGS['Animal_Name']
Num_Bins = int(ARGS['Number_Bins'])
Image_Path_On_PC = ARGS['Image_Path_On_PC']
Separate_Cam1_Cam3 = int(ARGS['Separate_Cam1_Cam3'])
Number_Frames = int(ARGS['Number_Frames'])
Higher_Speed_Range = float(ARGS['Higher_Speed_Range'])
Lower_Speed_Range = float(ARGS['Lower_Speed_Range'])
Time_All = float(ARGS['Time_All'])
Range_X = float(ARGS['Range_X'])
Range_Y = float(ARGS['Range_Y'])
Range_Z = float(ARGS['Range_Z'])

AZIMUTH = 0

def glob_remove(pattern):
    for file in glob(pattern):
        os.remove(file)



def Checking_Files_Existing(C2DPaths, DLTPaths, C3DPaths, StrPaths, SpePaths):
    Error_Flag = 0
    if not(os.path.exists(C2DPaths)):
        print(colored("Error: The 2D Tracked Coordinates do not exist in %s \r" %(C2DPaths),'red'))
        Error_Flag = 1

    if not(os.path.exists(DLTPaths)):
        print(colored("Error: The DLT Coefficients do not exist in %s \r" %(DLTPaths),'red'))
        Error_Flag = 1

    if not(os.path.exists(C3DPaths)):
        print(colored("Error: The 3D Tracked Coordinates do not exist in %s \r" %(C3DPaths),'red'))
        Error_Flag = 1

    for L0 in StrPaths:
        if not(os.path.exists(L0)):
            print(colored("Error: The Extarcted Information for Strides do not exist in %s \r" %(L0),'red'))
            Error_Flag = 1

    if not(os.path.exists(SpePaths)):
        print(colored("Error: The Extarcted Information for Speed do not exist in %s \r" %(SpePaths),'red'))
        Error_Flag = 1

    return(Error_Flag)



def Loading_Files(C2DPaths, DLTPaths, C3DPaths, StrPaths, SpePaths):
    data_C2D = pd.read_csv(C2DPaths, sep=',', index_col=0)
    data_DLT = pd.read_csv(DLTPaths, sep=',', header = None)
    data_C3D = pd.read_csv(C3DPaths, sep=',', index_col=0)
    data_Spe = pd.read_csv(SpePaths, sep=',')
    data_Str = []
    for L0 in StrPaths:
        data_Str.append(pd.read_csv(L0, sep=',', index_col=0))
    

    return(data_C2D, data_DLT, data_C3D, data_Str, data_Spe)


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



def Filetring(Dataframe, B,A):
    for column in Dataframe:
        if not(isinstance(Dataframe[column][0], str)):
            Dataframe[column] = filtfilt(B, A, Dataframe[column])

    return(Dataframe)


def Time_Series_Maker(data_Str, Number_Frames):
    X_Axis = np.array([])
    Y_Axis = np.array([])
    Z_Axis = np.array([])
    W_Axis = np.array([])
    Q_Axis = np.array([])
    Start = np.array([])
    End = np.array([])
    Stance_End = np.array([])
    Stride_Number = 0

    List_features = []
    for L0 in data_Str.columns:
        if L0.find("Angle")>-1:
            List_features.append(L0)

    for ii in range(0,Number_Frames):
        Index_Str = np.where(data_Str['Frame_Num']==ii)[0]
        if Index_Str.size == 1:
            X_Axis = np.append(X_Axis, ii)

            Frame_Str = data_Str.iloc[Index_Str]

            Y_Axis = np.append(Y_Axis, Frame_Str[List_features[0]])
            Z_Axis = np.append(Z_Axis, Frame_Str[List_features[1]])
            if len(List_features) == 3:
                W_Axis = np.append(W_Axis, Frame_Str[List_features[2]])
                Q_Axis = np.append(Q_Axis, np.nan)
            elif len(List_features) == 4:
                W_Axis = np.append(W_Axis, Frame_Str[List_features[2]])
                Q_Axis = np.append(Q_Axis, Frame_Str[List_features[3]])
            else: 
                W_Axis = np.append(W_Axis, np.nan)
                Q_Axis = np.append(Q_Axis, np.nan)

            if not(Stride_Number == Frame_Str['Number of Frames in Stride'].values):
                Start = np.append(Start, ii)
                End = np.append(End, Start[-1]+Frame_Str['Number of Frames in Stride'].values)
                Stance_End = np.append(Stance_End, Start[-1]+Frame_Str['Number of Frames in Phase'].values)
            Stride_Number = Frame_Str['Number of Frames in Stride'].values[0]
        else:
            Y_Axis = np.append(Y_Axis, np.nan)
            Z_Axis = np.append(Z_Axis, np.nan)
            W_Axis = np.append(W_Axis, np.nan)
            Q_Axis = np.append(Q_Axis, np.nan)
    return(X_Axis, Y_Axis, Z_Axis, W_Axis, Q_Axis, Start, End, Stance_End)



def Load_Image(Image_Bank):
    Image = cv2.imread(Image_Bank)
    b, g, r = cv2.split(Image)
    Image = cv2.merge((r, g, b))
    return(Image)




def Image_Add_Markers(Image, C2D, Color, Camera_Factor, Cam_Flag):
    temp1 = deepcopy(C2D)
    temp2 = deepcopy(C2D)
    for N, (L0, L1, L2, L3) in enumerate( zip(temp1[0:20:4], temp1[1:20:4], temp1[2:20:4], temp1[3:20:4]) ):
        temp2[(4-N)*4 + 0] = L0
        temp2[(4-N)*4 + 1] = L1
        temp2[(4-N)*4 + 2] = L2
        temp2[(4-N)*4 + 3] = L3


    Circle_Size = 12
    for D0, (L0 , L1, C0) in enumerate(zip(temp2[Camera_Factor*2::4], temp2[Camera_Factor*2+1::4], Color)):
        if Cam_Flag == 0:
            Image = cv2.circle(Image, (int(L1), int(L0)), Circle_Size, C0, -1)
        else:
            Image = cv2.circle(Image, (2048-int(L1), int(L0)), Circle_Size, C0, -1)


    for L0 , L1, L2, L3, C0 in zip(temp2[Camera_Factor*2::4], temp2[Camera_Factor*2+1::4], temp2[Camera_Factor*2+4::4], temp2[Camera_Factor*2+5::4], Color[4:]):
        if Cam_Flag == 0:
            Image = cv2.line(Image, (int(L1), int(L0)), (int(L3), int(L2)), (0,0,0), 5)
        else:
            Image = cv2.line(Image, (2048-int(L1), int(L0)), (2048-int(L3), int(L2)), (0,0,0), 5)
    return(Image)



def Animal_Location_Plotter(Frame_Spe_V, ax, fontsize):
    ax.set_xlim([-.15, .15])
    ax.set_ylim([-.05, .05])
    ax.set_xlabel('X Coordinate (m)', fontsize= fontsize, fontweight='bold')
    ax.set_ylabel('Y Coordinate (m)', fontsize= fontsize, fontweight='bold')
    ax.set_facecolor((.6, .6, .6))
    if Frame_Spe_V.shape[0]> 1:
        ax.plot(Frame_Spe_V[:-1,4], Frame_Spe_V[:-1,5], 'r*')
        ax.plot(Frame_Spe_V[-1,4], Frame_Spe_V[-1,5], 'bo')
    else:
        ax.plot(Frame_Spe_V[-1,4], Frame_Spe_V[-1,5], 'bo')

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize-1)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize-1)
        tick.label1.set_fontweight('bold')

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))



def Animal_Speed_Plotter(Frame_Spe_V, ax, fontsize, Frame_Number):
    ax.set_ylim([Lower_Speed_Range, Higher_Speed_Range])
    ax.set_xlim([0, Number_Frames])
    ax.set_xlabel('Frame Number', fontsize= fontsize, fontweight='bold')
    ax.set_ylabel('Speed (m/sec)', fontsize= fontsize, fontweight='bold')
    ax.set_facecolor((.6, .6, .6))
    if Frame_Spe_V.shape[0]> 1:
        ax.plot(range(len(Frame_Spe_V[:-1,2])), Frame_Spe_V[:-1,2], 'r-')
        ax.plot(Frame_Number, Frame_Spe_V[-1,2], 'bo')
    else:
        ax.plot(Frame_Number, Frame_Spe_V[-1,2], 'bo')

    ax.plot(range(len(Frame_Spe_V[:,3])), Frame_Spe_V[:,3], 'y-')

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize-1)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize-1)
        tick.label1.set_fontweight('bold')

    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))



def Plot_3D(Coordiantes_3D, Num_Markers, ax, FRONTPAW1, FRONTPAW2, HINDPAW1, HINDPAW2, fontsize, Color, Z_Min, AZIMUTH):
    ax.set_facecolor((.6, .6, .6))
    ax.grid(color='w', linestyle='-', linewidth=0.5)
    ax.set_xlabel('X Coordinate (mm)', fontsize= fontsize, fontweight='bold')
    ax.set_ylabel('Y Coordinate (mm)', fontsize= fontsize, fontweight='bold')
    ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel('Z Coordinate (mm)', fontsize= fontsize, fontweight='bold', rotation=90)

    temp1 = deepcopy(Coordiantes_3D[:int(Coordiantes_3D.shape[0]/2)])
    temp2 = deepcopy(Coordiantes_3D[:int(Coordiantes_3D.shape[0]/2)])
    for N, (L0, L1, L2) in enumerate( zip(temp1[0:15:3], temp1[1:15:3], temp1[2:15:3]) ):
        temp2[(4-N)*3 + 0] = L0
        temp2[(4-N)*3 + 1] = L1
        temp2[(4-N)*3 + 2] = L2


    MEANx = np.nanmean(Coordiantes_3D[0::3])
    MEANy = np.nanmean(Coordiantes_3D[1::3])
    MEANz = np.nanmean(Coordiantes_3D[2::3])

    for x1, y1, z1, x2, y2, z2, C in zip(temp2[0::3], temp2[1::3], temp2[2::3], temp2[3::3], temp2[4::3], temp2[5::3], Color): 
        ax.plot(np.array([x1,x2]), np.array([y1,y2]), np.array([z1,z2]), color = 'k')

    for x0, y0, z0, C0 in zip(temp2[0::3], temp2[1::3], temp2[2::3], Color): 
        ax.scatter(x0, y0, z0, s=20, color=C0)


    temp1 = deepcopy(Coordiantes_3D[int(Coordiantes_3D.shape[0]/2):])
    temp2 = deepcopy(Coordiantes_3D[int(Coordiantes_3D.shape[0]/2):])
    for N, (L0, L1, L2) in enumerate( zip(temp1[0:15:3], temp1[1:15:3], temp1[2:15:3]) ):
        temp2[(4-N)*3 + 0] = L0
        temp2[(4-N)*3 + 1] = L1
        temp2[(4-N)*3 + 2] = L2

    for x1, y1, z1, x2, y2, z2, C in zip(temp2[0::3], temp2[1::3], temp2[2::3], temp2[3::3], temp2[4::3], temp2[5::3], Color): 
        ax.plot(np.array([x1,x2]), np.array([y1,y2]), np.array([z1,z2]), color = 'k')

    for x0, y0, z0, C0 in zip(temp2[0::3], temp2[1::3], temp2[2::3], Color): 
        ax.scatter(x0, y0, z0, s=20, color=C0)

    ax.set_xlim([MEANx-Range_X, MEANx+Range_X])
    ax.set_ylim([MEANy-Range_Y, MEANy+Range_Y])
    ax.set_zlim([0, MEANz+Range_Z])

    AZIMUTH += 360/Number_Frames
    ax.view_init(elev=30., azim=AZIMUTH)
    # ax.axis('equal')

    return(AZIMUTH)



def Plot_Time_Series(X_Axis, Angle1, Angle2, Angle3, Angle4, Frame_Number, Start, End, ax, fontsize, Color, Color2, Stance_End, Second_time_Flag):
    X_Axis = np.linspace(0, Time_All, Number_Frames)
    ax.set_facecolor((.6, .6, .6))
    ax.set_xlabel('Time (Sec)', fontsize= fontsize, fontweight='bold')
    if Second_time_Flag == 0:
        ax.set_ylabel('Angle (Deg)', fontsize= fontsize, fontweight='bold')
    plt.plot(X_Axis, Angle1, Color2[0])
    plt.plot(X_Axis, Angle2, Color2[1])
    plt.plot(X_Axis, Angle3, Color2[2])
    plt.plot(X_Axis, Angle4, Color2[3])


    if np.isnan(np.nanmin(Angle3)) and np.isnan(np.nanmax(Angle4)):
        MINIMUM = min(np.nanmin(Angle2), np.nanmin(Angle1))
    elif np.isnan(np.nanmax(Angle3)):
        MINIMUM = min(np.nanmin(Angle4), np.nanmin(Angle2), np.nanmin(Angle1))
    else:
        MINIMUM = min(np.nanmin(Angle3), np.nanmin(Angle2), np.nanmin(Angle1))

    if np.isnan(np.nanmax(Angle3)) and np.isnan(np.nanmax(Angle4)):
        MAXIMUM = min(np.nanmax(Angle2), np.nanmax(Angle1))
    elif np.isnan(np.nanmax(Angle3)):
        MAXIMUM = min(np.nanmax(Angle4), np.nanmax(Angle2), np.nanmax(Angle1))
    else:
        MAXIMUM = min(np.nanmax(Angle3), np.nanmax(Angle2), np.nanmax(Angle1))

    Vertical_Line = 0
    for St, En, C0, St_En in zip(Start, End, Color, Stance_End):
        plt.plot(np.arange(St, En)/Number_Frames*Time_All, np.ones(int(En-St)) * 10, c=C0, linestyle='-', linewidth=2)
        plt.plot(np.arange(St, St_En)/Number_Frames*Time_All, np.ones(int(St_En-St)) * 15, c='w', linestyle='-', linewidth=2)


        if St<=Frame_Number and En>Frame_Number:
            plt.axvline(x=Frame_Number/Number_Frames*Time_All, c=C0, linewidth=1)
            Vertical_Line = 1

        if St<=Frame_Number and St_En>Frame_Number:
            plt.plot(np.ones(20) * Frame_Number/Number_Frames*Time_All, np.arange(160, 180), c='w', linewidth=1)


    if Vertical_Line==0:
        plt.axvline(x=Frame_Number/Number_Frames*Time_All, c='w', linewidth=1)

    ax.set_xlim([min(X_Axis), max(X_Axis)])
    ax.set_ylim([0, 180])
    if Second_time_Flag == 0:
        ax.set_ylim([0, 180])
    else:
        ax.get_yaxis().set_ticklabels([])
    X = np.linspace(0, Time_All, 17)
    ax.set_xticks(X, minor=True)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

    
    
    ax.grid(b=True, which='major', linestyle='-', linewidth='0.3', color='w')
    ax.grid(b=True, which='minor', linestyle='--', linewidth='0.2', color='w')

    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize-1)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize-1)
        tick.label1.set_fontweight('bold')



def Add_Title(Start, End, Color, Frame_Number, PAW, fontsize, ax, Name):
    Flag_t = 0
    Index_Str = np.where(PAW==Frame_Number)[0]
    if Index_Str.size == 1:
        Flag_t = 1

    if Flag_t == 1:
        for St, En, C0 in zip(Start, End, Color):
            if St<=Frame_Number and En>Frame_Number:
                ax.set_title(Name, fontsize= fontsize+1, fontweight='bold', color=C0)
    else:
        ax.set_title(Name, fontsize= fontsize+1, fontweight='bold', color='w')


def Add_Title_Image(Frame_Number, fontsize, ax, Name):
    ax.set_title(Name, fontsize= fontsize+1, fontweight='bold', color='w')



def Plot_Image_3D_Frame(Image1, Image2, Image3, Image4, Frame_Number, Num_Markers, Frame_C2D_V, Frame_C3D_V, Frame_Spe_V, FRONTPAW1, FRONTPAW2, HINDPAW1, HINDPAW2, Z_Min, AZIMUTH):
    fontsize = 8

    C2D_1 = Frame_C2D_V[2:int(Frame_C2D_V.shape[0]/2)]
    C2D_3 = Frame_C2D_V[2+int(Frame_C2D_V.shape[0]/2):]

    C3D = Frame_C3D_V[1:]

    Color1 = [(0,0,0), (0,0,255), (0,255,0), (255,255,0), (255,0,255), (0,255,255), (255,0,0), (255,255,255), (0,0,0), (0,0,0), (0,0,255), (0,255,0), (255,255,0), (255,0,255), (0,255,255), (255,0,0), (255,255,255), (0,0,0)]
    Color2 = ['yellow', 'lime', 'b', 'fuchsia', 'r','c', 'k', 'w', 'yellow', 'lime', 'b', 'fuchsia', 'r','c', 'k', 'w']
    Color3 = ['k', 'b', 'lime', 'yellow', 'fuchsia','c', 'r', 'w', 'k', 'b', 'lime', 'yellow', 'fuchsia', 'c', 'r', 'w']

    Image1 = Image_Add_Markers(Image1, C2D_1, Color1, 0, 0)
    Image2 = Image_Add_Markers(Image2, C2D_1, Color1, 1, 0)

    Image3 = Image_Add_Markers(Image3, C2D_3, Color1, 0, 1)
    Image4 = Image_Add_Markers(Image4, C2D_3, Color1, 1, 1)


    Flag = 0
    Index_Str = np.where(FRONTPAW1[0]==Frame_Number)[0]
    if Index_Str.size == 1:
        Flag = 1

    fig = plt.figure()
    plt.rcParams['savefig.facecolor'] = "0.4"
    plt.rcParams["font.family"] = "Times New Roman"
    
    GRID = gridspec.GridSpec(1, 4, left=0.01, bottom=0.83, right=0.99, top=0.95, wspace=0.02, hspace=0.01)
    ax = fig.add_subplot(GRID[0, 0])
    Add_Title_Image(Frame_Number, fontsize, ax, "Image from Camera 1")
    plt.imshow(Image1)
    plt.axis('off')

    ax = fig.add_subplot(GRID[0, 1])
    Add_Title_Image(Frame_Number, fontsize, ax, "Image from Camera 2")
    plt.imshow(Image2)
    plt.axis('off')

    ax = fig.add_subplot(GRID[0, 2])
    Add_Title_Image(Frame_Number, fontsize, ax, "Image from Camera 3")
    plt.imshow(Image3)
    plt.axis('off')

    ax = fig.add_subplot(GRID[0, 3])
    Add_Title_Image(Frame_Number, fontsize, ax, "Image from Camera 4")
    plt.imshow(Image4)
    plt.axis('off')



    GRID = gridspec.GridSpec(1, 1, left=0.74, right=0.99, bottom=0.1, top=0.25, wspace=0.01, hspace=0.01)
    ax = fig.add_subplot(GRID[0, 0])
    Add_Title_Image(Frame_Number, fontsize, ax, "Animal Location on Treadmill")
    Animal_Location_Plotter(Frame_Spe_V, ax, fontsize)

    GRID = gridspec.GridSpec(1, 1, left=0.74, right=0.99, bottom=0.38, top=0.51, wspace=0.01, hspace=0.01)
    ax = fig.add_subplot(GRID[0, 0])
    Add_Title_Image(Frame_Number, fontsize, ax, "Animal Speed on Treadmill")
    Animal_Speed_Plotter(Frame_Spe_V, ax, fontsize, Frame_Number)


    GRID = gridspec.GridSpec(1, 4, left=0.07, bottom=0.65, right=0.99, top=0.79, wspace=0.1, hspace=0.01)
    ax = fig.add_subplot(GRID[0, 0])
    Add_Title(HINDPAW1[5], HINDPAW1[6], Color2, Frame_Number, HINDPAW1[0], fontsize, ax, "Hind Paw Cam 1-2")
    Plot_Time_Series(HINDPAW1[0], HINDPAW1[1], HINDPAW1[2], HINDPAW1[3], HINDPAW1[4], Frame_Number, HINDPAW1[5], HINDPAW1[6], ax, fontsize, Color2, Color2[0:4], HINDPAW1[7], 0)

    ax = fig.add_subplot(GRID[0, 1])
    Add_Title(FRONTPAW1[5], FRONTPAW1[6], Color2, Frame_Number, FRONTPAW1[0], fontsize, ax, "Front Paw Cam 1-2")
    Plot_Time_Series(FRONTPAW1[0], FRONTPAW1[1], FRONTPAW1[2], FRONTPAW1[3], FRONTPAW1[4], Frame_Number, FRONTPAW1[5], FRONTPAW1[6], ax, fontsize, Color2, Color2[4:], FRONTPAW1[7], 1)

    
    ax = fig.add_subplot(GRID[0, 2])
    Add_Title(HINDPAW2[5], HINDPAW2[6], Color2, Frame_Number, HINDPAW2[0], fontsize, ax, "Hind Paw Cam 3-4")
    Plot_Time_Series(HINDPAW2[0], HINDPAW2[1], HINDPAW2[2], HINDPAW2[3], HINDPAW2[4], Frame_Number, HINDPAW2[5], HINDPAW2[6], ax, fontsize, Color2, Color2[0:4], HINDPAW2[7], 1)

    ax = fig.add_subplot(GRID[0, 3])
    Add_Title(FRONTPAW2[5], FRONTPAW2[6], Color2, Frame_Number, FRONTPAW2[0], fontsize, ax, "Front Paw Cam 3-4")
    Plot_Time_Series(FRONTPAW2[0], FRONTPAW2[1], FRONTPAW2[2], FRONTPAW2[3], FRONTPAW2[4], Frame_Number, FRONTPAW2[5], FRONTPAW2[6], ax, fontsize, Color2, Color2[4:], FRONTPAW2[7], 1)


    GRID = gridspec.GridSpec(1, 1, left=0.06, bottom=0.04, right=0.65, top=0.55, wspace=0.2, hspace=0.1)
    ax = fig.add_subplot(GRID[0 , 0], projection='3d')
    Add_Title_Image(Frame_Number, fontsize, ax, "3D Reconstruction")
    AZIMUTH = Plot_3D(C3D, Num_Markers, ax, FRONTPAW1, FRONTPAW2, HINDPAW1, HINDPAW2, fontsize, Color3, Z_Min, AZIMUTH)

    # plt.show()
    plt.savefig("Image%05d.png" %(Frame_Number), dpi=300)
    plt.close()
    return(AZIMUTH)



def decode_names(Path, List_PAW_ADDs, Animal_Name):
    Path = Path[:-4]
    Tri_Loc = len(Path) - Path[-1::-1].find(".")
    Trial_Name = Path[Tri_Loc:]

    Path = Path[:Tri_Loc-1]
    Con_Loc = len(Path) - Path[-1::-1].find(".")
    Condition_Name = Path[Con_Loc:]

    Path = Path[:Con_Loc-1]
    Animal_Number = Path.find(Animal_Name)
    Animal_Number = Path[Animal_Number+len(Animal_Name):]

    NEW = []
    for L0 in List_PAW_ADDs:
        NEW.append(Path[0:2] + "Stride_Features." + L0 + "." + Animal_Name + Animal_Number + "." + Condition_Name + "." + Trial_Name + ".csv")
    return(NEW)









CurrentPath = "."
Coordinate_Path = sorted(glob(os.path.join(CurrentPath, Animal_Name+"*.csv")))


count = 0
DLT_Path = deepcopy(Coordinate_Path)
C3D_Path = deepcopy(Coordinate_Path)
Speed_Path = deepcopy(Coordinate_Path)
Strides_Path_F1 = deepcopy(Coordinate_Path)
Strides_Path_F2 = deepcopy(Coordinate_Path)
Strides_Path_H1 = deepcopy(Coordinate_Path)
Strides_Path_H2 = deepcopy(Coordinate_Path)


filt_order = 3; sample_rate = 250.0; filt_freq = 20;
filt_param = filt_freq/sample_rate; filt_type = 'low';
[B,A] = butter(filt_order,filt_param,filt_type);


List_PAW_ADDs = ["F1", "F3", "H1", "H3"]


for L in Coordinate_Path: 
    DLT_Path[count] = L[0:2] + 'DLT.' + L[2:]
    C3D_Path[count] = L[0:2] + 'Coord_3D.' + L[2:]
    Speed_Path[count] = L[0:2] + 'Speed_Time.' + L[2:]
    [Strides_Path_F1[count], Strides_Path_F2[count], Strides_Path_H1[count], Strides_Path_H2[count]] = decode_names(L, List_PAW_ADDs, Animal_Name)
    
    count += 1



for (C2DPaths, DLTPaths, C3DPaths, StrPaths_F1, StrPaths_F2, StrPaths_H1, StrPaths_H2, SpePaths) in zip(Coordinate_Path, DLT_Path, C3D_Path, Strides_Path_F1, Strides_Path_F2, Strides_Path_H1, Strides_Path_H2, Speed_Path):
    print("Processing Video for dir %s   \r" %(C2DPaths))

    StrPaths = [StrPaths_F1, StrPaths_F2, StrPaths_H1, StrPaths_H2]
    Error_Flag = Checking_Files_Existing(C2DPaths, DLTPaths, C3DPaths, StrPaths, SpePaths)

    if Error_Flag == 0:
        (data_C2D, data_DLT, data_C3D, data_Str, data_Spe) = Loading_Files(C2DPaths, DLTPaths, C3DPaths, StrPaths, SpePaths)
        glob_remove(os.path.join(CurrentPath,"Image*.png"))

        Image_Path1, Image_Path2, Image_Path3, Image_Path4 = Find_Cameras_Path(data_C2D, Image_Path_On_PC)
        Image_Bank1 = sorted(glob(os.path.join(Image_Path1, "*.png")))
        Image_Bank2 = sorted(glob(os.path.join(Image_Path2, "*.png")))
        Image_Bank3 = sorted(glob(os.path.join(Image_Path3, "*.png")))
        Image_Bank4 = sorted(glob(os.path.join(Image_Path4, "*.png")))

        data_C2D = Filetring(data_C2D, B,A)

        FRONTPAW1 = Time_Series_Maker(data_Str[0], Number_Frames)
        FRONTPAW2 = Time_Series_Maker(data_Str[1], Number_Frames)
        HINDPAW1 = Time_Series_Maker(data_Str[2], Number_Frames)
        HINDPAW2 = Time_Series_Maker(data_Str[3], Number_Frames)

        Z_Min = [np.min(data_C3D.values[:, 3::3]), np.max(data_C3D.values[:, 3::3])]

        for index, Frame_C2D in data_C2D.iterrows():
            print("Please Wait; Processing Frame %s   \r" %(index))

            Skipping_Columns = 0
            for ii in Frame_C2D.values:
                if isinstance(ii, str):
                    Skipping_Columns += 1

            Num_Markers = 8

            Frame_C3D = data_C3D.iloc[index]

            Frame_Number = int(Frame_C3D['Frame Number'])

            Image1 = Load_Image(Image_Bank1[index])
            Image2 = Load_Image(Image_Bank2[index])
            Image3 = Load_Image(Image_Bank3[index])
            Image4 = Load_Image(Image_Bank4[index])

            Speed = data_Spe.iloc[:Frame_Number+1]

            AZIMUTH = Plot_Image_3D_Frame(Image1, Image2, Image3, Image4, Frame_Number, Num_Markers, Frame_C2D.values, Frame_C3D.values, Speed.values, FRONTPAW1, FRONTPAW2, HINDPAW1, HINDPAW2, Z_Min, AZIMUTH)


        subprocess.call('ffmpeg -i Image%*.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p Video.'+C2DPaths[2:-4]+'.avi', shell= True)
        glob_remove(os.path.join(CurrentPath,"Image*.png"))

        print(colored("Please wait; Processing the next trial.",'yellow'))


print(colored("Process has been completely DONE.",'green'))






