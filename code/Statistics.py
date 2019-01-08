import cv2, os, sys, math, pdb, argparse
import numpy as np
from glob import glob
from copy import deepcopy
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
from matplotlib import cm
from peakdetect import peakdetect
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
from termcolor import colored


parser = argparse.ArgumentParser(description = "Help on how to set the parameters and use short keys while calling the code:")

parser.add_argument("-pa", "--Plot_All", type=int, help = "One image for all plots", default = str(1))
parser.add_argument("-pas", "--Plot_Average_Separately", type=int, help = "It saves the average graphs separately", default = str(1))

parser.add_argument("-animal", "--Animal_Name", type=str, help = "The name of animal", default = "rat")
parser.add_argument("-separate", "--Separate_Cam1_Cam3", type=str, help = "If separate is 1 then each will be considered as a condition", default = str(0))
parser.add_argument("-bins", "--Number_Bins", type=str, help = "Number of bins for plotting", default = str(200))

parser.add_argument("-ymin", "--YlimMin", type=str, help = "Number of bins for plotting", default = str(0))
parser.add_argument("-ymax", "--YlimMax", type=str, help = "Number of bins for plotting", default = str(180))
parser.add_argument("-fs", "--fontsize", type=str, help = "Number of bins for plotting", default = str(10))

parser.add_argument("-sn", "--Study_Name", type=str, help = "Name of study", default = "DREADDS_Control")
parser.add_argument("-xplf", "--Experiment_List_Front", type=str, help = "List of experiments", default = "week0,week1,week2")
parser.add_argument("-xplh", "--Experiment_List_Hind", type=str, help = "List of experiments", default = "week0,week1,week2")

global Num_Bins, COLOR

args = parser.parse_args(); ARGS = vars(args)

Plot_Average_Separately = int(ARGS['Plot_Average_Separately'])
Plot_All = int(ARGS['Plot_All'])

Animal_Name = ARGS['Animal_Name']
Separate_Cam1_Cam3 = int(ARGS['Separate_Cam1_Cam3'])
Num_Bins = int(ARGS['Number_Bins'])

YlimMin = int(ARGS['YlimMin'])
YlimMax = int(ARGS['YlimMax'])
fontsize = int(ARGS['fontsize'])

Name_Study = ARGS['Study_Name']
Experiment_List_Front = ARGS['Experiment_List_Front']
Experiment_List_Hind = ARGS['Experiment_List_Hind']


def decode_exp_list(list_paw):
    temp = []
    while True:
        temp.append(list_paw[:list_paw.find(",")])
        list_paw = list_paw[list_paw.find(",")+1:]
        if list_paw.find(",") == -1:
            temp.append(list_paw)
            break; break

    return(temp)

Experiment_List_Front = decode_exp_list(Experiment_List_Front)
Experiment_List_Hind = decode_exp_list(Experiment_List_Hind)

COLOR = ['lime', 'b', 'r', 'm', 'c', 'y']


def My_Ts_Plot2(X, Y, Var, Stance, col, ax1, fontsize, Rat, YlimMin, YlimMax, M, N, Exp, Number_Rows, Number_Cols):
    ax1.set_facecolor('xkcd:silver')

    if Stance[0] > -1:
        ax1.axvspan(Stance[0]+Stance[1], 100, facecolor='xkcd:grey', alpha=0.35)
        ax1.axvspan(Stance[0]-Stance[1], Stance[0]+Stance[1], facecolor='xkcd:medium grey', alpha=0.45)
        ax1.axvline(x=Stance[0], c='k', linewidth=1)


    ax1.plot(X, Y, color = col)
    ax1.fill_between(X, Y+Var, Y-Var, color = col, alpha=0.5)


    fontweight='bold'
    
    if M == 0:
        ax1.set_title(Rat, fontweight = fontweight, fontsize = fontsize+5)

    
    if N == 0:
        ax1.set_ylabel(Exp, fontweight = fontweight, fontsize = fontsize+5)


    if M == Number_Rows-1:
        ax1.set_xlabel('Stride Percentage', fontweight = fontweight, fontsize = fontsize+2)

    ax1.set_ylim([YlimMin, YlimMax])    

    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')




def Ploting_Ankle_Knee_Each_Rat(Color, Exp, Data, Rat, Feature_Name, YlimMin, YlimMax, ax1, fontsize, M, N, Number_Rows, Number_Cols):
    Rows_Index = ( Data.index[ (Data['Experiment']==Exp) & (Data['Rat Number']==Rat)] ).tolist()

    if len(Rows_Index)>0:
        Temp_Data = Data.iloc[Rows_Index]
        Temp_Data = Temp_Data.reset_index(drop=True)

        Stance_Mean = round(Temp_Data['Stance Number of Bins'].mean())
        Stance_Std = round(Temp_Data['Stance STD Number of Bins'].mean())

        for N0, (L0, CC) in enumerate(zip(Feature_Name, COLOR)):
            if N0 == 0:
                My_Ts_Plot2(Temp_Data['Percentage'], Temp_Data[L0+' Mean'], Temp_Data[L0+' Std'], [Stance_Mean, Stance_Std], CC, ax1, fontsize, '', YlimMin, YlimMax, M, N, Exp, Number_Rows, Number_Cols)
            else:
                My_Ts_Plot2(Temp_Data['Percentage'], Temp_Data[L0+' Mean'], Temp_Data[L0+' Std'], [-1, Stance_Std], CC, ax1, fontsize, Rat, YlimMin, YlimMax, M, N, Exp, Number_Rows, Number_Cols)

        Patches_list = []
        for L0, CC in zip(Feature_Name, COLOR):
             Patches_list.append( mpatches.Patch(color=CC, label=L0+ ', ' + str(int(Temp_Data['Number of Strides'].iloc[0])) + ' Strides') )

        ax1.legend(handles=Patches_list, fontsize=fontsize-5)

    else:
        fontweight='bold'
        if N == 0:
            ax1.set_ylabel(Exp, fontweight = fontweight, fontsize = fontsize+5)

        if M == 0:
            ax1.set_title(Rat, fontweight = fontweight, fontsize = fontsize+5)





def Plot_Average(Color, Exp, Data, Feature_Name, YlimMin, YlimMax, ax1, fontsize, M, N, Number_Rows, Number_Cols):
    Rows_Index = ( Data.index[ (Data['Experiment']==Exp)] ).tolist()
    if len(Rows_Index)>0:
        Temp_Data = Data.iloc[Rows_Index]
        Temp_Data = Temp_Data.reset_index(drop=True)
        Mean = Temp_Data.groupby('Percentage').mean()
        Std = Temp_Data.groupby('Percentage').std()

        Stance_Mean = round(Temp_Data['Stance Number of Bins'].mean())
        Stance_Std = round(Temp_Data['Stance STD Number of Bins'].mean())

        for N0, (L0, CC) in enumerate(zip(Feature_Name, COLOR)):
            if N0 == 0:
                My_Ts_Plot2(Temp_Data['Percentage'].iloc[0:Num_Bins+1], Mean[L0+' Mean'], Std[L0+' Std'], [Mean['Stance Number of Bins'].iloc[0], Std['Stance STD Number of Bins'].iloc[0]], CC, ax1, fontsize, '', YlimMin, YlimMax, M, N, Exp, Number_Rows, Number_Cols)
            else:
                My_Ts_Plot2(Temp_Data['Percentage'].iloc[0:Num_Bins+1], Mean[L0+' Mean'], Std[L0+' Std'], [-1, Std['Stance STD Number of Bins'].iloc[0]], CC, ax1, fontsize, 'Average For All Rats', YlimMin, YlimMax, M, N, Exp, Number_Rows, Number_Cols)
        
        Patches_list = []
        for L0, CC in zip(Feature_Name, COLOR):
             Patches_list.append( mpatches.Patch(color=CC, label=L0) )

        ax1.legend(handles=Patches_list, fontsize=fontsize-5)




def PLOTTING_Together(Color, Experiment_List, Data, Rat, fontsize, Feature_Name, YlimMin, YlimMax, ax1, M, N, Number_Rows, Number_Cols):
    
    Mean_temp = np.array([])
    for Exp in Experiment_List:
        Rows_Index = ( Data.index[ (Data['Experiment']==Exp) & (Data['Rat Number']==Rat)] ).tolist()
        Temp_Data = Data.iloc[Rows_Index]
        Mean_t = round(np.nanmean(Temp_Data['Stance Number of Bins']))
        Mean_temp = np.append(Mean_temp, Mean_t)
        
    Stance_Mean = round(np.nanmean(Mean_temp))
    Stance_Std = round(np.nanstd(Mean_temp))


    Counter = 0
    pacths = []
    for D, (Exp, col) in enumerate(zip(Experiment_List, Color)):
        Rows_Index = ( Data.index[ (Data['Experiment']==Exp) & (Data['Rat Number']==Rat)] ).tolist()

        if len(Rows_Index)>0:
            Temp_Data = Data.iloc[Rows_Index]
            Temp_Data = Temp_Data.reset_index(drop=True)

            if Counter == 0:
                My_Ts_Plot2(Temp_Data['Percentage'], Temp_Data[Feature_Name+' Mean'], Temp_Data[Feature_Name+' Std'], [Stance_Mean, Stance_Std], col, ax1, fontsize, '', YlimMin, YlimMax, M, N, 'All '+Feature_Name, Number_Rows, Number_Cols)
            else:
                My_Ts_Plot2(Temp_Data['Percentage'], Temp_Data[Feature_Name+' Mean'], Temp_Data[Feature_Name+' Std'], [-1, Stance_Std], col, ax1, fontsize, '', YlimMin, YlimMax, M, N, 'All '+Feature_Name, Number_Rows, Number_Cols)
            Counter += 1

        pacths.append(mpatches.Patch(color=col, label=Exp))
    ax1.legend(handles=pacths, fontsize=fontsize-5)



def PLOTTING_All_Average(Color, Experiment_List, Data, fontsize, Feature_Name, YlimMin, YlimMax, ax1, M, N, Number_Rows, Number_Cols):
    Mean_temp = np.array([])
    for Exp in Experiment_List:
        Rows_Index = ( Data.index[ (Data['Experiment']==Exp)] ).tolist()
        Temp_Data = Data.iloc[Rows_Index]
        Mean_t = round(np.nanmean(Temp_Data['Stance Number of Bins']))
        Mean_temp = np.append(Mean_temp, Mean_t)
        
    Stance_Mean = round(np.nanmean(Mean_temp))
    Stance_Std = round(np.nanstd(Mean_temp))


    Counter = 0
    pacths = []
    for D, (Exp, col) in enumerate(zip(Experiment_List, Color)):
        Rows_Index = ( Data.index[ (Data['Experiment']==Exp)] ).tolist()

        if len(Rows_Index)>0:
            Temp_Data = Data.iloc[Rows_Index]
            Temp_Data = Temp_Data.reset_index(drop=True)

            Mean = Temp_Data.groupby('Percentage').mean()
            Std = Temp_Data.groupby('Percentage').std()

            if Counter == 0:
                My_Ts_Plot2(Temp_Data['Percentage'].iloc[0:Num_Bins+1], Mean[Feature_Name+' Mean'], Std[Feature_Name+' Std'], [Stance_Mean, Stance_Std], col, ax1, fontsize, '', YlimMin, YlimMax, M, N, Exp, Number_Rows, Number_Cols)
            else:
                My_Ts_Plot2(Temp_Data['Percentage'].iloc[0:Num_Bins+1], Mean[Feature_Name+' Mean'], Std[Feature_Name+' Std'], [-1, Stance_Std], col, ax1, fontsize, '', YlimMin, YlimMax, M, N, Exp, Number_Rows, Number_Cols)    
            Counter += 1


            pacths.append(mpatches.Patch(color=col, label=Exp))
    ax1.legend(handles=pacths, fontsize=fontsize-5)




def Full_Plot(Name_Study, CurrentPath, Data, Experiment_List, Color, YlimMin, YlimMax, Features_list, fontsize):
    Number_rats = len(Data['Rat Number'].unique())
    Number_exp = len(Experiment_List)

    Number_Rows = Number_exp+len(Features_list)
    Number_Cols = Number_rats+1

    fig = plt.figure(Name_Study)
    fig.set_size_inches(20  ,  20)
    
    plt.rcParams['savefig.facecolor'] = "0.6"
    plt.rcParams["font.family"] = "Times New Roman"


    for M, Exp in enumerate(Experiment_List):
        for N, Rat in enumerate(Data['Rat Number'].unique()):
            ax = fig.add_subplot(Number_Rows, Number_Cols, M*(Number_rats+1)+N+1)
            Ploting_Ankle_Knee_Each_Rat(Color, Exp, Data, Rat, Features_list, YlimMin, YlimMax, ax, fontsize, M, N, Number_Rows, Number_Cols)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_axisbelow(True); ax.grid(b=None, which='major', axis='both')


    for M, Exp in enumerate(Experiment_List):
        N = deepcopy(Number_rats)
        ax = fig.add_subplot(Number_Rows, Number_Cols, M*(Number_rats+1)+N+1)
        Plot_Average(Color, Exp, Data, Features_list, YlimMin, YlimMax, ax, fontsize, M, N, Number_Rows, Number_Cols)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_axisbelow(True); ax.grid(b=None, which='major', axis='both')


    # Thota_removing = np.where(Data['Experiment']=='normal_thota')[0]
    # if len(Thota_removing) > 0 :
    #    Data.at[Thota_removing, 'Stance STD Number of Bins'] = np.nan
    #    Data.at[Thota_removing, 'Stance Number of Bins'] = np.nan

    # Thota_removing = np.where(Data['Experiment']=='week0_thota')[0]
    # if len(Thota_removing) > 0 :
    #    Data.at[Thota_removing, 'Stance STD Number of Bins'] = np.nan
    #    Data.at[Thota_removing, 'Stance Number of Bins'] = np.nan

    # Thota_removing = np.where(Data['Experiment']=='week4_thota')[0]
    # if len(Thota_removing) > 0 :
    #    Data.at[Thota_removing, 'Stance STD Number of Bins'] = np.nan
    #    Data.at[Thota_removing, 'Stance Number of Bins'] = np.nan

    # Thota_removing = np.where(Data['Experiment']=='normal')[0]
    # if len(Thota_removing) > 0 :
    #    Data.at[Thota_removing, 'Stance STD Number of Bins'] = np.nan
    #    Data.at[Thota_removing, 'Stance Number of Bins'] = np.nan

    for N0, L0 in enumerate(Features_list):
        for N, Rat in enumerate(Data['Rat Number'].unique()):
            M = deepcopy(Number_exp)+N0
            ax = fig.add_subplot(Number_Rows, Number_Cols, M*(Number_rats+1)+N+1)
            PLOTTING_Together(Color, Experiment_List, Data, Rat, fontsize, L0, YlimMin, YlimMax, ax, M, N, Number_Rows, Number_Cols)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.set_axisbelow(True); ax.grid(b=None, which='major', axis='both')


    for N0, L0 in enumerate(Features_list):
        M = Number_exp+N0; N = deepcopy(Number_rats)
        ax = fig.add_subplot(Number_Rows, Number_Cols, M*(Number_rats+1)+N+1)
        PLOTTING_All_Average(Color, Experiment_List, Data, fontsize, L0, YlimMin, YlimMax, ax, M, N, Number_Rows, Number_Cols)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_axisbelow(True); ax.grid(b=None, which='major', axis='both')

    FEATURES = ""
    for N0, L0 in enumerate(Features_list):
        if N0 == 0:
            FEATURES = FEATURES+L0
        else:
            FEATURES = FEATURES+"_"+L0
    plt.tight_layout()
    fig.savefig(os.path.join(os.path.join(CurrentPath, 'Figures'), Name_Study+'_All_'+FEATURES+'.png'), dpi=600)
    ax.set_axisbelow(True); 
    plt.close()



def Plot_Single_One_Fully(Name_Study, Color, Experiment_List, Data, fontsize, Name_Features_Plot, YlimMin, YlimMax, CurrentPath):
    fig = plt.figure(Name_Study)
    fig.set_size_inches(5, 4)
    plt.rcParams['savefig.facecolor'] = "0.6"
    plt.rcParams["font.family"] = "Times New Roman"
    ax = fig.add_subplot(1, 1, 1)
    PLOTTING_All_Average(Color, Experiment_List, Data, fontsize, Name_Features_Plot, YlimMin, YlimMax, ax, 0, 0, 1, 1)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_axisbelow(True); ax.grid(b=None, which='major', axis='both')
    plt.tight_layout()
    ax.set_ylabel(Name_Features_Plot + ' Angle (deg)', fontweight = 'bold', fontsize = fontsize+5)
    fig.savefig(os.path.join(os.path.join(CurrentPath, 'Figures'), Name_Study +'_' +Name_Features_Plot+'.png'), dpi=600)
    ax.set_axisbelow(True); 
    plt.close()



def Details(Experiment_List, PAW, Name_Study, YlimMin, YlimMax, fontsize):
    CurrentPath = "."
    if not os.path.exists('Figures'):
        os.makedirs('Figures')

    Color = [ cm.jet(x) for x in np.linspace(0, 1, len(Experiment_List))  ]

    Data_Path_All = sorted(glob(os.path.join(CurrentPath, "Stride_All_Final_Concatenated_Matrix_"+PAW+"*.csv")))
    Data = pd.read_csv(Data_Path_All[0], sep=',', index_col=0)

    List_Rats_all = Data['Rat Number'].unique()
    List_Exp_all = Data['Experiment'].unique()
    All_columns = Data.columns
    Features_list = []
    for L0 in All_columns:
        Place_Mean = L0.find(" Mean")
        if Place_Mean > -1:
            Features_list.append(L0[:Place_Mean])

    if Plot_All == 1:
        Full_Plot(Name_Study, CurrentPath, Data, Experiment_List, Color, YlimMin, YlimMax, Features_list, fontsize)


    if Plot_Average_Separately == 1:
        for L0 in Features_list:
            Plot_Single_One_Fully(Name_Study, Color, Experiment_List, Data, fontsize, L0, YlimMin, YlimMax, CurrentPath)

print(colored("Please wait; Processing Front Paw.",'yellow'))
Details(Experiment_List_Front, "F", Name_Study, YlimMin, YlimMax, fontsize)
print(colored("Please wait; Processing Hind Paw.",'yellow'))
Details(Experiment_List_Hind, "H", Name_Study, YlimMin, YlimMax, fontsize)
print(colored("Process has been completely DONE.",'green'))



