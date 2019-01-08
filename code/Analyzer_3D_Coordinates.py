import cv2, os, sys, math, pdb, argparse
import numpy as np
from glob import glob
from copy import deepcopy
from termcolor import colored
from peakdetect import peakdetect
import pandas as pd
from scipy.signal import butter, filtfilt



parser = argparse.ArgumentParser(description = "Help on how to set the parameters and use short keys while calling the code:")

parser.add_argument("-animal", "--Animal_Name", type=str, help = "The name of animal", default = "rat")
parser.add_argument("-ts", "--Thre_Speed", type=str, help = "Threshold for Speed (accepted variance of speed), lower means the speeds should be closer", default = str(0.15))
parser.add_argument("-mcf", "--Minimum_Consecutive_Frames", type=str, help = "Minimum number of consecutive frames which can be accepted as a stride", default = str(70))
parser.add_argument("-np", "--Number_Points", type=int, help = "Number of frames being used for demonstartion", default = 100)
parser.add_argument("-separate", "--Separate_By_Speed", type=str, help = "If separate is 1 then each speed will be considered as a condition", default = str(0))
parser.add_argument("-step", "--Steps_Speed", type=str, help = "Increament in speed", default = str(0.04))

args = parser.parse_args(); ARGS = vars(args)

Animal_Name = ARGS['Animal_Name']
Thre_Speed = float(ARGS['Thre_Speed'])
Minimum_Consecutive_Frames = int(ARGS['Minimum_Consecutive_Frames'])
Separate_By_Speed = float(ARGS['Separate_By_Speed'])
Steps_Speed = float(ARGS['Steps_Speed'])
Steps_Speed *= 100
Number_Points = ARGS['Number_Points']



def Feature_Extractor_points(XYZ, MIN, Num_Markers, Features_size): # markers are from top to bottom
    R_P = np.zeros( [1,Num_Markers-1] )*np.nan # distance from points
    R_P_J = np.zeros([1,1])*np.nan # distance top to bottom
    D_M = np.zeros( [1,Num_Markers] )*np.nan # distance from minimum Z
    J_A = np.zeros( [1,Num_Markers-2] )*np.nan # Joint Angle

    counter1 = np.arange(1, Num_Markers)
    counter2 = np.arange(0, Num_Markers-1)
    count = 0
    for C1,C2 in zip(counter1, counter2):
        Coordinates1 = XYZ[C1*3:C1*3+3]
        Coordinates2 = XYZ[C2*3:C2*3+3]
        Diff = Coordinates1-Coordinates2
        R_P[0, count] = math.sqrt(sum((Diff)**2))
        count += 1

    C1 = 0; C2 = Num_Markers-1
    Coordinates1 = XYZ[C1*3:C1*3+3]
    Coordinates2 = XYZ[C2*3:C2*3+3]
    Diff = Coordinates1-Coordinates2
    R_P_J = math.sqrt(sum((Diff)**2))

    counter = np.arange(0, Num_Markers)
    count = 0
    for C in counter:
        Coordinates = XYZ[C*3+2]
        D_M[0, count] = math.fabs(Coordinates - MIN)
        count += 1

    counter = np.arange(2, Num_Markers)
    count = 0
    for C in counter:
        Coordinates1 = XYZ[C*3 : C*3+3]
        Coordinates2 = XYZ[(C-1)*3 : (C-1)*3+3]
        Coordinates3 = XYZ[(C-2)*3 : (C-2)*3+3]
        ba = Coordinates1-Coordinates2
        bc = Coordinates3-Coordinates2
        ba = ba/np.linalg.norm(ba)
        bc = bc/np.linalg.norm(bc)

        Cos_Part = np.dot(ba, bc) 
        Sin_Part = np.linalg.norm(np.cross(ba, bc))
        J_A[0, count] = np.arctan2(Sin_Part, Cos_Part)
        count += 1

    Feature = np.zeros([1, Features_size])*np.nan

    Count = 0
    Feature[0, Count:Count+Num_Markers-1] = R_P
    Count += Num_Markers-1
    Feature[0, Count] = R_P_J
    Count += 1
    Feature[0, Count:Count+Num_Markers] = D_M
    Count += Num_Markers
    Feature[0, Count:Count+Num_Markers-2] = J_A/np.pi*180

    return (Feature)



def Feature_Extractor_points_both(XYZ1, XYZ2):
    Featrue = np.zeros([1,2])*np.nan

    Coordinates1 = XYZ2[0 : 3]
    Coordinates2 = XYZ1[0 : 3]
    Coordinates3 = XYZ1[3 : 6]
    ba = Coordinates1-Coordinates2
    bc = Coordinates3-Coordinates2
    ba = ba/np.linalg.norm(ba)
    bc = bc/np.linalg.norm(bc)

    Cos_Part = np.dot(ba, bc) 
    Sin_Part = np.linalg.norm(np.cross(ba, bc))
    Featrue[0, 0] = np.arctan2(Sin_Part, Cos_Part)

    Coordinates1 = XYZ2[0 : 3]
    Coordinates2 = XYZ1[0 : 3]
    Diff = Coordinates1-Coordinates2
    Featrue[0, 1] = math.sqrt(sum((Diff)**2))
    return(Featrue)



def consecutive(data, stepsize=1):
    Series = np.split(data, np.where(np.diff(data) != stepsize)[0]+1)

    return Series 



def Constant_Speed(X, Belt, Thre_Speed, Minimum_Consecutive_Frames):
    filt_order = 3; sample_rate = 250.0; filt_freq = 5;
    filt_param = filt_freq/sample_rate; filt_type = 'low'
    [B,A] = butter(filt_order,filt_param,filt_type)
    x = filtfilt(B, A, X)

    Loc_Thresholded = np.where(abs(x-Belt)<Thre_Speed)[0]
    Loc = []
    All_Loc = []
    if len(Loc_Thresholded) > 0:
        Series = consecutive(Loc_Thresholded)
        for S1 in Series:
            if len(S1) > Minimum_Consecutive_Frames:
                Loc.append(S1)

        for L1 in Loc:
            All_Loc = All_Loc+L1.tolist()

    return(Loc, All_Loc, X[All_Loc])



def List_Maker(List_Base_Features, List_Base_Main, List_Base_Other, List_Adding_Info):
    List_Features = []

    L0 = List_Base_Features[0]
    for L1, L2 in zip(List_Base_Main[:-1], List_Base_Main[1:]):
        List_Features.append(L0+' '+L1+' '+L2)
    List_Features.append('Dist Bottom Top')
    L1 = 'Min'
    for L2 in List_Base_Main:
        List_Features.append(L0+' '+L1+' '+L2)

    L0 = List_Base_Features[1]
    for L1, L2, L3 in zip(List_Base_Main[:-2], List_Base_Main[1:], List_Base_Main[2:]):
        List_Features.append(L0+' '+L1+' '+L2+' '+L3)

    L0 = List_Base_Features[0]; L1 = List_Base_Main[0]; L2 = List_Base_Other[0]; 
    List_Features.append(L0+' '+L1+' '+L2)

    L0 = List_Base_Features[1]; L1 = List_Base_Main[1]; L2 = List_Base_Main[0]; L3 = List_Base_Other[0];
    List_Features.append(L0+' '+L1+' '+L2+' '+L3)

    List_Features1 = []
    List_Features2 = []

    List_Sides = ['']
    for L1 in List_Features:
        List_Features1.append(L1+List_Sides[0])
        List_Features2.append(L1+List_Sides[0])

    for L0 in List_Adding_Info:
        List_Features1.append(L0)
        List_Features2.append(L0)

    return(List_Features1, List_Features2)



def Main_process(FEATURES, Data_temp, XYZ_Coor, XYZ_Coor_Other, Speed_time_temp, Part, Part_Min, List_Features, Num_Markers, Stride_Counter, Temp_Stride_Counter):
    peaks = peakdetect(Data_temp[Part],lookahead=30) 

    p_peaks = np.zeros([len(peaks[0]), ])
    n_peaks = np.zeros([len(peaks[1]), ])

    C = 0
    for L in peaks[0]: p_peaks[C] = L[0]; C += 1
    C = 0
    for L in peaks[1]: n_peaks[C] = L[0]; C += 1

    Min_peaks = int(np.amin(p_peaks))
    Max_peaks = int(np.amax(p_peaks))

    Min_peaks_Phase = int(np.amin([np.amin(p_peaks), np.amin(n_peaks)]))
    Max_peaks_Phase = int(np.amax([np.amax(p_peaks), np.amax(n_peaks)]))


    Num_Rows = Max_peaks - Min_peaks + 1
    Features = np.zeros([Num_Rows-1, len(List_Features)])*np.nan
    

    List_Peaks = deepcopy(p_peaks)
    List_Peaks.sort()

    List_Peaks_Phase = np.append(p_peaks, n_peaks)
    List_Peaks_Phase.sort()


    Counter = 0
    Temp_Peak_strat = Min_peaks
    change_flag = False

    MIN = Data_temp[Part_Min].min()

    # stance=1, swing=0
    # they are set opposite as switching in the first iter of loop
    if n_peaks[0] == Min_peaks_Phase:
        Swing_Flag = False
    else:
        Swing_Flag = True
    Temp_Peak_strat_Phase = Min_peaks_Phase


    for N, NR in enumerate(range(Min_peaks, Max_peaks)):
        Peak_Start = int(List_Peaks[List_Peaks<=NR][-1])
        Peak_end = int(List_Peaks[List_Peaks>NR][0])

        Percentage = (NR-Peak_Start)/(Peak_end-Peak_Start)


        Peak_Start_Phase = int(List_Peaks_Phase[List_Peaks_Phase<=NR][-1])
        Peak_end_Phase = int(List_Peaks_Phase[List_Peaks_Phase>NR][0])


        if Temp_Peak_strat_Phase != Peak_Start_Phase:
            Swing_Flag = not(Swing_Flag)



        if Temp_Peak_strat != Peak_Start or N == 0:
            Stride_Counter += 1

       
        Features[N, :len(List_Features)-8] = Feature_Extractor_points(XYZ_Coor[NR,:], MIN, Num_Markers, len(List_Features)-8) # counting markers oposite to make sure the first marker is bottom one
        Features[N, len(List_Features)-8:len(List_Features)-6] = Feature_Extractor_points_both(XYZ_Coor[NR,:], XYZ_Coor_Other[NR,:]) # counting markers oposite to make sure the first marker is bottom one
        Features[N, len(List_Features)-6] = Percentage
        Features[N, len(List_Features)-5] = Stride_Counter
        Features[N, len(List_Features)-4] = int(Peak_end-Peak_Start)
        Features[N, len(List_Features)-3] = Swing_Flag
        Features[N, len(List_Features)-2] = Peak_end_Phase - Peak_Start_Phase

        if Temp_Stride_Counter != Stride_Counter:
            Features[N, len(List_Features)-1] = 1
        else:
            Features[N, len(List_Features)-1] = 0

        Temp_Peak_strat = deepcopy(Peak_Start)
        Temp_Peak_strat_Phase = deepcopy(Peak_Start_Phase)
        Temp_Stride_Counter = deepcopy(Stride_Counter)

    Features_df = pd.DataFrame(data = Features, columns = List_Features)
    Speed_df = Speed_time_temp.iloc[Min_peaks:Max_peaks].reset_index(drop=True)

    Data_df = Data_temp.iloc[Min_peaks:Max_peaks].reset_index(drop=True)


    df = [Speed_df, Data_df, Features_df]
    DF = pd.concat(df, axis=1)


    if len(FEATURES) == 0:
        FEATURES = deepcopy(DF)
    else:
        FEATURES = pd.concat([FEATURES,DF], axis=0)
        FEATURES = FEATURES.reset_index(drop=True)

    return(FEATURES, Stride_Counter, Temp_Stride_Counter)








CurrentPath = "."

# find the files starting with rat and ending by csv
Deleting_Path = sorted(glob(os.path.join(CurrentPath, "Stride_Features*.csv")))
# Making the associated path for DLT and 3D output folder
for L in Deleting_Path: 
    os.remove(L)


Coordinate_Path = sorted(glob(os.path.join(CurrentPath, "Coord_3D." + Animal_Name + "*.csv")))


# Setting output paths
Features_Output_Path_F1 = deepcopy(Coordinate_Path)
Features_Output_Path_F2 = deepcopy(Coordinate_Path)
Features_Output_Path_H1 = deepcopy(Coordinate_Path)
Features_Output_Path_H2 = deepcopy(Coordinate_Path)
count = 0
for L in Coordinate_Path: 
    Features_Output_Path_F1[count] = L[0:2] + 'Stride_Features.F1.' + L[11:]
    Features_Output_Path_F2[count] = L[0:2] + 'Stride_Features.F3.' + L[11:]
    Features_Output_Path_H1[count] = L[0:2] + 'Stride_Features.H1.' + L[11:]
    Features_Output_Path_H2[count] = L[0:2] + 'Stride_Features.H3.' + L[11:]
    count += 1

List_Base_Features = ['Dist', 'Angle']
List_Base_Paw = ['Asis', 'Hip', 'Knee', 'Ankle', 'Toe']
List_Base_Hand = ['Shoulder', 'Elbow', 'Hand']
List_Adding_Info = ['Stride Percentage', 'Stride Number', 'Number of Frames in Stride', 'Phase Flag', 'Number of Frames in Phase', 'First Time Phase Changing']

(List_Features_H1, List_Features_H2) = List_Maker(List_Base_Features, List_Base_Paw, List_Base_Hand, List_Adding_Info)
(List_Features_F1, List_Features_F2) = List_Maker(List_Base_Features, List_Base_Hand, List_Base_Paw, List_Adding_Info)


for (DirPath, FeaturePathsF1, FeaturePathsF2, FeaturePathsH1, FeaturePathsH2) in zip(Coordinate_Path, Features_Output_Path_F1, Features_Output_Path_F2, Features_Output_Path_H1, Features_Output_Path_H2):
    print("Analysing 3D Information; Processing dir %s   \r" %(DirPath))
    # loading the speed_time data
    Speed_time_Path = './Speed_Time' + '.' + DirPath[11:]
    Speed_time = pd.read_csv(Speed_time_Path, sep=',')


    # loading the 3D coordinates
    Data_Main = pd.read_csv(DirPath, sep=',', index_col=0)
    Frame_Number = Data_Main[Data_Main.columns[0]].values # frame number
    


    # keep the speed time data for frame numbers availabe in tracking
    Speed_time = Speed_time.iloc[Frame_Number]
    # reindexing the speed to have the same indexes as Data
    Speed_time = Speed_time.reset_index(drop=True)
    # consider the most repeated speed as belt speed
    Temp = (Speed_time['Belt_Speed']*10).astype(int)
    Belt_Speed = np.bincount(Temp).argmax()/10.0


    # find indexs that meeting the threshold for speed and they have a specific minimum length
    (Index, All_Index, Speed) = Constant_Speed(Speed_time['Real_Speed'], Belt_Speed, Thre_Speed, Minimum_Consecutive_Frames)

    FEATURES_H1 = np.array([]); Stride_Counter_H1 = 0; Temp_Stride_Counter_H1 = 0
    FEATURES_F1 = np.array([]); Stride_Counter_F1 = 0; Temp_Stride_Counter_F1 = 0
    FEATURES_H2 = np.array([]); Stride_Counter_H2 = 0; Temp_Stride_Counter_H2 = 0
    FEATURES_F2 = np.array([]); Stride_Counter_F2 = 0; Temp_Stride_Counter_F2 = 0

    for Inx in Index:
        Speed_time_temp = Speed_time.iloc[Inx]
        Data = Data_Main[Data_Main.columns[1:]].iloc[Inx]


        # find peaks (Now fidning peaks for toe x, associating with stance. maybe z)
        List_Parts_Index = [12, 21, 36, 45]
        Part_Movement_Stride = Data.columns[List_Parts_Index]
        List_Parts_Index_Min = [14, 23, 38, 47]
        Part_Movement_Min = Data.columns[List_Parts_Index_Min]

        
        XYZ_Coor = Data[Data.columns[range(List_Parts_Index_Min[0]+1)]].values
        XYZ_Coor_Other = Data[Data.columns[List_Parts_Index_Min[0]+1 : List_Parts_Index_Min[1]+1]].values
        Num_Markers = int( XYZ_Coor.shape[1] /3 ) # number of markers
        Data_temp = Data[Data.columns[range(List_Parts_Index_Min[0]+1)]]
        peaks = peakdetect(Data_temp[ Part_Movement_Stride[0] ],lookahead=30) 
        if len(peaks[0])>0 and len(peaks[1])>0:
            (FEATURES_H1, Stride_Counter_H1, Temp_Stride_Counter_H1) = Main_process(FEATURES_H1, 
                Data_temp, XYZ_Coor, XYZ_Coor_Other, Speed_time_temp, Part_Movement_Stride[0], Part_Movement_Min[0], List_Features_H1, Num_Markers, Temp_Stride_Counter_H1, Stride_Counter_H1)


        XYZ_Coor = Data[Data.columns[ range(List_Parts_Index_Min[0]+1, List_Parts_Index_Min[1]+1) ]].values
        XYZ_Coor_Other = Data[Data.columns[ range(List_Parts_Index_Min[0]+1) ]].values
        Num_Markers = int( XYZ_Coor.shape[1] /3 ) # number of markers
        Data_temp = Data[Data.columns[ range(List_Parts_Index_Min[0]+1, List_Parts_Index_Min[1]+1) ]]
        peaks = peakdetect(Data_temp[ Part_Movement_Stride[1] ],lookahead=30) 
        if len(peaks[0])>0 and len(peaks[1])>0:
            (FEATURES_F1, Stride_Counter_F1, Temp_Stride_Counter_F1) = Main_process(FEATURES_F1, 
                Data_temp, XYZ_Coor, XYZ_Coor_Other, Speed_time_temp, Part_Movement_Stride[1], Part_Movement_Min[1], List_Features_F1, Num_Markers, Temp_Stride_Counter_F1, Stride_Counter_F1)


        XYZ_Coor = Data[Data.columns[ range(List_Parts_Index_Min[1]+1, List_Parts_Index_Min[2]+1) ]].values
        XYZ_Coor_Other = Data[Data.columns[ range(List_Parts_Index_Min[2]+1, List_Parts_Index_Min[3]+1) ]].values
        Num_Markers = int( XYZ_Coor.shape[1] /3 ) # number of markers
        Data_temp = Data[Data.columns[ range(List_Parts_Index_Min[1]+1, List_Parts_Index_Min[2]+1) ]]
        peaks = peakdetect(Data_temp[ Part_Movement_Stride[2] ],lookahead=30) 
        if len(peaks[0])>0 and len(peaks[1])>0:
            (FEATURES_H2, Stride_Counter_H2, Temp_Stride_Counter_H2) = Main_process(FEATURES_H2, 
                Data_temp, XYZ_Coor, XYZ_Coor_Other, Speed_time_temp, Part_Movement_Stride[2], Part_Movement_Min[2], List_Features_H2, Num_Markers, Temp_Stride_Counter_H2, Stride_Counter_H2)


        XYZ_Coor = Data[Data.columns[ range(List_Parts_Index_Min[2]+1, List_Parts_Index_Min[3]+1) ]].values
        XYZ_Coor_Other = Data[Data.columns[ range(List_Parts_Index_Min[1]+1, List_Parts_Index_Min[2]+1) ]].values
        Num_Markers = int( XYZ_Coor.shape[1] /3 ) # number of markers
        Data_temp = Data[Data.columns[ range(List_Parts_Index_Min[2]+1, List_Parts_Index_Min[3]+1) ]]
        peaks = peakdetect(Data_temp[ Part_Movement_Stride[3] ],lookahead=30) 
        if len(peaks[0])>0 and len(peaks[1])>0:
            (FEATURES_F2, Stride_Counter_F2, Temp_Stride_Counter_F2) = Main_process(FEATURES_F2, 
                Data_temp, XYZ_Coor, XYZ_Coor_Other, Speed_time_temp, Part_Movement_Stride[3], Part_Movement_Min[3], List_Features_F2, Num_Markers, Temp_Stride_Counter_F2, Stride_Counter_F2)

    
    if len(FEATURES_H1) != 0: 
        if Separate_By_Speed == 1:
            Rat_Loc = FeaturePathsH1.find('.'+Animal_Name)+1
            Temp_Path = deepcopy(FeaturePathsH1)
            Temp_Path = Temp_Path[Rat_Loc:]
            Start_Condition_loc = Temp_Path.find(".")+1
            Temp_Path = Temp_Path[Start_Condition_loc:]
            Start_Condition_loc += Rat_Loc

            End_Condition_loc = Temp_Path.find(".")
            Temp_Path = Temp_Path[:Temp_Path.find(".")]
            End_Condition_loc += Start_Condition_loc

            Speed_average = FEATURES_H1['Belt_Speed'].mean()
            Speed_average *= 100
            Speed_average /= Steps_Speed
            Speed_average = round(Speed_average)
            Speed_average *= Steps_Speed
            Speed_average = int(Speed_average)

            FeaturePathsH1 = FeaturePathsH1[:31] + "_speed" + str(Speed_average) + FeaturePathsH1[31:]
            FeaturePathsF1 = FeaturePathsF1[:31] + "_speed" + str(Speed_average) + FeaturePathsF1[31:]
            FeaturePathsH2 = FeaturePathsH2[:31] + "_speed" + str(Speed_average) + FeaturePathsH2[31:]
            FeaturePathsF2 = FeaturePathsF2[:31] + "_speed" + str(Speed_average) + FeaturePathsF2[31:]



    if len(FEATURES_H1) != 0: 
        FEATURES_H1.to_csv(FeaturePathsH1)
    if len(FEATURES_F1) != 0: 
        FEATURES_F1.to_csv(FeaturePathsF1)
    if len(FEATURES_H2) != 0: 
        FEATURES_H2.to_csv(FeaturePathsH2)
    if len(FEATURES_F2) != 0: 
        FEATURES_F2.to_csv(FeaturePathsF2)

print(colored("Process has been completely DONE.",'green'))
