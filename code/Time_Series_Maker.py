import cv2, os, sys, math, pdb, argparse
import numpy as np
from glob import glob
from copy import deepcopy
import pandas as pd
from termcolor import colored


parser = argparse.ArgumentParser(description = "Help on how to set the parameters and use short keys while calling the code:")
parser.add_argument("-animal", "--Animal_Name", type=str, help = "The name of animal", default = "rat")
parser.add_argument("-separate", "--Separate_Cam1_Cam3", type=str, help = "If separate is 1 then each will be considered as a condition", default = str(0))
parser.add_argument("-bins", "--Number_Bins", type=str, help = "Number of bins for plotting", default = str(200))
parser.add_argument("-mfn", "--Max_Frame_Number", type=str, help = "Number of bins for plotting", default = str(250))
args = parser.parse_args(); ARGS = vars(args)

Animal_Name = ARGS['Animal_Name']
Separate_Cam1_Cam3 = int(ARGS['Separate_Cam1_Cam3'])
Num_Bins = int(ARGS['Number_Bins'])
Number_Max = int(ARGS['Max_Frame_Number'])




def Interpolator(temp, Num_Bins, LIST, Features_List):
    Number_Stride = int(len(temp['Stride Number'].unique()))
    Temp_Percentage_1 = np.zeros([Num_Bins+1, Number_Stride])
    Temp_Percentage_2 = np.zeros([Num_Bins+1, Number_Stride])
    Temp_Percentage_3 = np.zeros([Num_Bins+1, Number_Stride])
    Temp_Percentage_4 = np.zeros([Num_Bins+1, Number_Stride])
    Zero_Hundred = np.arange(0, Num_Bins+1)
    
    Data_Percentage = temp[LIST]

    Uniq_Strides = temp['Stride Number'].unique()
    Uniq_Number_Frames = []
    for US in Uniq_Strides:
        TEMP = temp[temp['Stride Number']==US]
        Uniq_Number_Frames.append(TEMP['Number of Frames in Stride'].iloc[0])

    Uniq_Strides = np.arange(0,len(Uniq_Strides))

    Uniq_Number_Frames = np.asarray(Uniq_Number_Frames)

    Phase_info = np.insert(Uniq_Number_Frames, 0, 0)
    Phase_info = np.cumsum(Phase_info)

    Start = np.zeros([Uniq_Strides.shape[0], 2])

    Start[:,0] = Phase_info[(Uniq_Strides).astype(int)]
    Start[:,1] = Phase_info[(Uniq_Strides+1).astype(int)]

    Stride_List_A = ['Number of Strides',]
    Stride_List_K = []
    Stride_List_As = []
    Stride_List_Asa = []

    for S0, (S1, L1) in enumerate(zip(Start, range(Number_Stride))):

        S1 = S1.astype(int)
        df = Data_Percentage[S1[0]:S1[1]]
        Length_Features = len(Features_List)

        if Length_Features == 2:
            X = df[LIST[2]] * Num_Bins
            Y = df[LIST[0]]
            Z = df[LIST[1]]

            Temp_Percentage_1[:,L1] = np.interp(np.array(Zero_Hundred), X, Y)
            Temp_Percentage_2[:,L1] = np.interp(np.array(Zero_Hundred), X, Z)

            Stride_List_A.append(Features_List[0]+' Stride Number '+str(S0))
            Stride_List_K.append(Features_List[1]+' Stride Number '+str(S0))

        elif Length_Features == 3:  
            X = df[LIST[3]] * Num_Bins
            Y = df[LIST[0]]
            Z = df[LIST[1]]
            W = df[LIST[2]]

            Temp_Percentage_1[:,L1] = np.interp(np.array(Zero_Hundred), X, Y)
            Temp_Percentage_2[:,L1] = np.interp(np.array(Zero_Hundred), X, Z)
            Temp_Percentage_3[:,L1] = np.interp(np.array(Zero_Hundred), X, W)

            Stride_List_A.append(Features_List[0]+' Stride Number '+str(S0))
            Stride_List_K.append(Features_List[1]+' Stride Number '+str(S0))
            Stride_List_As.append(Features_List[2]+' Stride Number '+str(S0))

        else:
            X = df[LIST[4]] * Num_Bins
            Y = df[LIST[0]]
            Z = df[LIST[1]]
            W = df[LIST[2]]
            Q = df[LIST[3]]

            Temp_Percentage_1[:,L1] = np.interp(np.array(Zero_Hundred), X, Y)
            Temp_Percentage_2[:,L1] = np.interp(np.array(Zero_Hundred), X, Z)
            Temp_Percentage_3[:,L1] = np.interp(np.array(Zero_Hundred), X, W)
            Temp_Percentage_4[:,L1] = np.interp(np.array(Zero_Hundred), X, Q)

            Stride_List_A.append(Features_List[0]+' Stride Number '+str(S0))
            Stride_List_K.append(Features_List[1]+' Stride Number '+str(S0))
            Stride_List_As.append(Features_List[2]+' Stride Number '+str(S0))
            Stride_List_Asa.append(Features_List[3]+' Stride Number '+str(S0))


        
        

    if Length_Features == 2:
        Feature_Name = ['Average F1', 'STD F1', 'Average F2', 'STD F2']
        Features = np.zeros([Num_Bins+1, len(Feature_Name)])
        Features[:,0] = np.mean(Temp_Percentage_1, axis=1)
        Features[:,1] = np.std(Temp_Percentage_1, axis=1)
        Features[:,2] = np.mean(Temp_Percentage_2, axis=1)
        Features[:,3] = np.std(Temp_Percentage_2, axis=1)

        Stride_List = Stride_List_A + Stride_List_K
        Stride_List.append(Features_List[0]+' Mean')
        Stride_List.append(Features_List[0]+' Std')
        Stride_List.append(Features_List[1]+' Mean')
        Stride_List.append(Features_List[1]+' Std')

        Stride_Number = np.ones([Num_Bins+1, 1])*Number_Stride

        Temp_Percentage = np.append(Stride_Number, Temp_Percentage_1, axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Temp_Percentage_2, axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,0].reshape((Num_Bins+1,1)), axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,1].reshape((Num_Bins+1,1)), axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,2].reshape((Num_Bins+1,1)), axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,3].reshape((Num_Bins+1,1)), axis = 1)


    elif Length_Features == 3:  
        Feature_Name = ['Average F1', 'STD F1', 'Average F2', 'STD F2', 'Average F3', 'STD F3']
        Features = np.zeros([Num_Bins+1, len(Feature_Name)])
        Features[:,0] = np.mean(Temp_Percentage_1, axis=1)
        Features[:,1] = np.std(Temp_Percentage_1, axis=1)
        Features[:,2] = np.mean(Temp_Percentage_2, axis=1)
        Features[:,3] = np.std(Temp_Percentage_2, axis=1)
        Features[:,4] = np.mean(Temp_Percentage_3, axis=1)
        Features[:,5] = np.std(Temp_Percentage_3, axis=1)

        Stride_List = Stride_List_A + Stride_List_K + Stride_List_As
        Stride_List.append(Features_List[0]+' Mean')
        Stride_List.append(Features_List[0]+' Std')
        Stride_List.append(Features_List[1]+' Mean')
        Stride_List.append(Features_List[1]+' Std')
        Stride_List.append(Features_List[2]+' Mean')
        Stride_List.append(Features_List[2]+' Std')

        Stride_Number = np.ones([Num_Bins+1, 1])*Number_Stride

        Temp_Percentage = np.append(Stride_Number, Temp_Percentage_1, axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Temp_Percentage_2, axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Temp_Percentage_3, axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,0].reshape((Num_Bins+1,1)), axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,1].reshape((Num_Bins+1,1)), axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,2].reshape((Num_Bins+1,1)), axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,3].reshape((Num_Bins+1,1)), axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,4].reshape((Num_Bins+1,1)), axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,5].reshape((Num_Bins+1,1)), axis = 1)


    else:
        Feature_Name = ['Average F1', 'STD F1', 'Average F2', 'STD F2', 'Average F3', 'STD F3', 'Average F4', 'STD F4']
        Features = np.zeros([Num_Bins+1, len(Feature_Name)])
        Features[:,0] = np.mean(Temp_Percentage_1, axis=1)
        Features[:,1] = np.std(Temp_Percentage_1, axis=1)
        Features[:,2] = np.mean(Temp_Percentage_2, axis=1)
        Features[:,3] = np.std(Temp_Percentage_2, axis=1)
        Features[:,4] = np.mean(Temp_Percentage_3, axis=1)
        Features[:,5] = np.std(Temp_Percentage_3, axis=1)
        Features[:,6] = np.mean(Temp_Percentage_4, axis=1)
        Features[:,7] = np.std(Temp_Percentage_4, axis=1)

        Stride_List = Stride_List_A + Stride_List_K + Stride_List_As + Stride_List_Asa
        Stride_List.append(Features_List[0]+' Mean')
        Stride_List.append(Features_List[0]+' Std')
        Stride_List.append(Features_List[1]+' Mean')
        Stride_List.append(Features_List[1]+' Std')
        Stride_List.append(Features_List[2]+' Mean')
        Stride_List.append(Features_List[2]+' Std')
        Stride_List.append(Features_List[3]+' Mean')
        Stride_List.append(Features_List[3]+' Std')

        Stride_Number = np.ones([Num_Bins+1, 1])*Number_Stride

        Temp_Percentage = np.append(Stride_Number, Temp_Percentage_1, axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Temp_Percentage_2, axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Temp_Percentage_3, axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Temp_Percentage_4, axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,0].reshape((Num_Bins+1,1)), axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,1].reshape((Num_Bins+1,1)), axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,2].reshape((Num_Bins+1,1)), axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,3].reshape((Num_Bins+1,1)), axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,4].reshape((Num_Bins+1,1)), axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,5].reshape((Num_Bins+1,1)), axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,6].reshape((Num_Bins+1,1)), axis = 1)
        Temp_Percentage = np.append(Temp_Percentage, Features[:,7].reshape((Num_Bins+1,1)), axis = 1)


    All_Strides = pd.DataFrame(data = Temp_Percentage, columns = Stride_List)

    return(All_Strides)



def TS_Plot_Data_Orgnizer(List, temp, Features_List, Rat_Num, Experiment_Name, Data_All, Num_Bins, Counter = 0):
    LIST = deepcopy(List)
    LIST.append('Stride Percentage')

    List_Phase = ['Stride']

    All_Strides = Interpolator(temp, Num_Bins, LIST, Features_List)

    Features_List = []
    for L in List:
        for P in List_Phase:
            for F in ['Mean', 'SD']:
                Features_List.append(F + ' of ' + L + ' for ' + P)


    percentage = np.arange(0, Num_Bins+1)/2

    All_Strides['Percentage'] = pd.Series(data= percentage)


    Mean_Phase = temp[['Number of Frames in Phase', 'Phase Flag']].groupby('Phase Flag').mean()
    Percentage_Mean_Phase = Mean_Phase/Mean_Phase.sum() * 100
    Percentage_Mean_Phase = round(Percentage_Mean_Phase)
    Percentage_Mean_Phase = Percentage_Mean_Phase['Number of Frames in Phase'][1]
    Percentage_Mean_Phase = np.ones([Num_Bins+1, ]) * Percentage_Mean_Phase
    All_Strides['Stance Number of Bins'] = pd.Series(data= Percentage_Mean_Phase)


    Std_Phase = temp[['Number of Frames in Phase', 'Phase Flag']].groupby('Phase Flag').std().mean()
    Percentage_Std_Phase = Std_Phase/Mean_Phase.sum() * 100
    Percentage_Std_Phase = round(Percentage_Std_Phase)
    Percentage_Std_Phase = np.ones([Num_Bins+1, ]) * Percentage_Std_Phase[0]
    All_Strides['Stance STD Number of Bins'] = pd.Series(data= Percentage_Std_Phase)



    Rat_Num = [Rat_Num]
    Rat_Num = Rat_Num * (Num_Bins+1)
    All_Strides['Rat Number'] = pd.Series(data= Rat_Num)


    Experiment_Name = [Experiment_Name]
    Experiment_Name = Experiment_Name * (Num_Bins+1)
    All_Strides['Experiment'] = pd.Series(data= Experiment_Name)

    COLUMNS = All_Strides.columns

    if Counter == 0:
        Data_All = deepcopy(All_Strides)
    else:
        Temp = [Data_All, All_Strides]
        Data_All = pd.concat(Temp, sort=True)

    return(Data_All)



def Remove_Old(PAW_ADD):
    CurrentPath = "."
    Coordinate_Path = sorted(glob(os.path.join(CurrentPath, 'Stride_All_Final_Concatenated_Matrix_' + PAW_ADD[0] + "*.csv")))
    for DirPaths in Coordinate_Path:
        try:
            os.remove(DirPaths)
        except OSError:
            pass



def Main(Animal_Name, PAW_ADD, Num_Bins, Number_Max):
    CurrentPath = "."
    Coordinate_Path = sorted(glob(os.path.join(CurrentPath, "Stride_Features." +PAW_ADD+ "." +Animal_Name+ "*.0000-00-00_00_00_00.csv")))
    Sample = Coordinate_Path[0]
    Features_Name_temp = ['Rat Number', 'Experiment']
    temp = pd.read_csv(Coordinate_Path[0], sep=',', index_col=0)
    Features_Name = temp.columns
    while True:
        for N, L0 in enumerate(Features_Name):
            # pdb.set_trace()
            if L0[:4].find("Dist")> -1:
                break
        break
    Features_Name = Features_Name[N:]

    Data = pd.DataFrame(data = [], columns = Features_Name)
    Counter = 0
    Data_All_Strides = Data_All_Strides1 = Data_All_Strides2 = Data_All_Strides3 = Data_All_Strides4 = Data_All_Strides5 = []
    for DirPaths in Coordinate_Path:
        temp = []
        print("Making time series; Processing dir %s   \r" %(DirPaths))

        Rat_Num_Pos = DirPaths.find("."+Animal_Name)+1
        Length_Rat_Name = DirPaths[Rat_Num_Pos:].find('.')

        if Length_Rat_Name == 4:
            Rat_Num = DirPaths[Rat_Num_Pos:Rat_Num_Pos+4]
            Experiment_Name = DirPaths[Rat_Num_Pos+5:]
            Experiment_pos = Experiment_Name.find('.')
            Experiment_Name = Experiment_Name[:Experiment_pos]
            Experiment_Name = Experiment_Name.lower()
        else:
            Rat_Num = DirPaths[Rat_Num_Pos:Rat_Num_Pos+5]
            Experiment_Name = DirPaths[Rat_Num_Pos+6:]
            Experiment_pos = Experiment_Name.find('.')
            Experiment_Name = Experiment_Name[:Experiment_pos]
            Experiment_Name = Experiment_Name.lower()


        temp = pd.read_csv(DirPaths, sep=',', index_col=0)


        Rat_Num_T = [Rat_Num] 
        Rat_Num_T = Rat_Num_T * (temp.shape[0])
        temp['Rat Number'] = pd.Series(data= Rat_Num_T, index = temp.index)

        if len(PAW_ADD) == 2:
            Experiment_Name = Experiment_Name

        Experiment_Name_T = [Experiment_Name] 
        Experiment_Name_T = Experiment_Name_T * (temp.shape[0])
        temp['Experiment'] = pd.Series(data= Experiment_Name_T, index = temp.index)


        if Experiment_Name == 'normal_thota':
            Number_Max = 250

        if (temp['Number of Frames in Stride']>Number_Max).any(): 
            temp = temp.reset_index(drop=True)
            Condition = temp['Number of Frames in Stride']>Number_Max
            Index_delete = np.where(Condition)[0]
            List_delete = temp.index[[Index_delete]]
            temp = temp.drop(List_delete)



        if not(temp.empty):
            temp = temp.reset_index(drop=True)

            Temp = [Data, temp]
            Data = pd.concat(Temp, sort=True)
            Data = Data[Features_Name]


            Features_List = []
            List = []
            while True:
                for N, L0 in enumerate(Features_Name):
                    if L0[:5].find("Angle") > -1:
                        List.append(L0)
                        if L0 == 'Angle Shoulder Elbow Hand':
                            Features_List.append('Elbow')

                        elif L0 == 'Angle Elbow Shoulder Asis':
                            Features_List.append('Shoulder')

                        elif L0 == 'Angle Asis Hip Knee':
                            Features_List.append('Hip')

                        elif L0 == 'Angle Hip Knee Ankle':
                            Features_List.append('Knee')

                        elif L0 == 'Angle Knee Ankle Toe':
                            Features_List.append('Ankle')

                        elif L0 == 'Angle Hip Asis Shoulder':
                            Features_List.append('Asis')
                break

            Data_All_Strides = TS_Plot_Data_Orgnizer(List, temp, Features_List, Rat_Num, Experiment_Name, Data_All_Strides, Num_Bins, Counter)
            
            Counter += 1

        ##################################################################################
    Data = Data.reset_index(drop=True)
    Data = Data.reindex(Features_Name, axis=1)
    Data.to_csv('Stride_Final_Concatenated_Matrix.csv', sep=',')

    Saving_Path = ""
    for N, L0 in enumerate(Features_List):
        if N == 0:
            Saving_Path = Saving_Path+L0
        else:
            Saving_Path = Saving_Path+"_"+L0
    

    Saving_file = 'Stride_All_Final_Concatenated_Matrix_' +PAW_ADD[0]+ '_' +Saving_Path+ '.csv'
    COLUMNS = Data_All_Strides.columns.values.tolist()
    if not(os.path.exists(Saving_file)):
        Data_All_Features = Data_All_Strides.reset_index(drop=True)
        Data_All_Features.to_csv(Saving_file)

    else:
        temp = pd.read_csv(Saving_file, sep=',', index_col=0)
        Temp = [temp, Data_All_Strides]
        Data_All_Features = pd.concat(Temp, sort=True)
        Data_All_Features = Data_All_Features[COLUMNS]

        Data_All_Features = Data_All_Features.reset_index(drop=True)

        Data_All_Features.to_csv(Saving_file)



if Separate_Cam1_Cam3 == 0:
    PAW_ADD = "F"
    Remove_Old(PAW_ADD)
    Main(Animal_Name, PAW_ADD, Num_Bins, Number_Max)

    PAW_ADD = "H"
    Remove_Old(PAW_ADD)
    Main(Animal_Name, PAW_ADD, Num_Bins, Number_Max)

else:
    PAW_ADD = "F1"
    Remove_Old(PAW_ADD)
    Main(Animal_Name, PAW_ADD, Num_Bins, Number_Max)

    PAW_ADD = "F3"
    Main(Animal_Name, PAW_ADD, Num_Bins, Number_Max)
    
    
    PAW_ADD = "H1"
    Remove_Old(PAW_ADD)
    Main(Animal_Name, PAW_ADD, Num_Bins, Number_Max)

    PAW_ADD = "H3"
    Main(Animal_Name, PAW_ADD, Num_Bins, Number_Max)





print(colored("Process has been completely DONE.",'green'))
