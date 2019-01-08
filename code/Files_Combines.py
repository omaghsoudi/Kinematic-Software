import cv2, argparse, os, sys, pdb
import numpy as np
from shutil import copyfile
from glob import glob
import pandas as pd
from termcolor import colored


parser = argparse.ArgumentParser(description = "Help on how to set the parameters and use short keys while calling the code:")
parser.add_argument("-animal", "--Animal_Name", type=str, help = "The name of animal", default = "rat")
parser.add_argument("-separate", "--Separate_Cam1_Cam3", type=str, help = "If separate is 1 then each camera will be considered as a condition", default = str(0))
args = parser.parse_args(); ARGS = vars(args)

Animal_Name = ARGS['Animal_Name']
Separate_Cam1_Cam3 = int(ARGS['Separate_Cam1_Cam3'])


def Comibne_Files(Animal_Name, STRING, PAW_ADD):
    CurrentPath = "."
    Looking_Name_Base = "Stride_Features."+ PAW_ADD
    Looking_Name = Looking_Name_Base+"*."+ Animal_Name+ "*"
    Coordinate_Path = sorted(glob(os.path.join(CurrentPath, Looking_Name+".csv")))

    Labels = ["Path", "Rat_Num", "Experiment"]
    Data = pd.DataFrame(data=[], columns=Labels)

    for Num, DirPaths in enumerate(Coordinate_Path):
        print("Combining the files; Processing dir %s   \r" %(DirPaths))

        Rat_Num_Pos = DirPaths.find("." + Animal_Name)+1
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



        Saving_file = "./"+Looking_Name_Base+"." + Rat_Num + "." + Experiment_Name + STRING + ".0000-00-00_00_00_00.csv"
        temp = pd.read_csv(DirPaths, sep=',', index_col=0)
        COLUMNS = temp.columns.values.tolist()
        if not(os.path.exists(Saving_file)):
            temp.to_csv(Saving_file)

        else:
            temp2 = pd.read_csv(Saving_file, sep=',', index_col=0)

            temp['Stride Number'] += temp2['Stride Number'].max()

            Temp = [temp, temp2]
            Data_All_Strides = pd.concat(Temp, sort=True)
            Data_All_Strides = Data_All_Strides[COLUMNS]

            Data_All_Strides = Data_All_Strides.reset_index(drop=True)

            Data_All_Strides.to_csv(Saving_file)


def Remove_Old(PAW_ADD, Animal_Name):
    CurrentPath = "."
    Looking_Name_Base_temp = "Stride_Features."+ PAW_ADD[0]
    Looking_Name = Looking_Name_Base_temp+"*."+ Animal_Name+ "*"
    Coordinate_Path = sorted(glob(os.path.join(CurrentPath, Looking_Name+".0000-00-00_00_00_00.csv")))
    for DirPaths in Coordinate_Path:
        try:
            os.remove(DirPaths)
        except OSError:
            pass




if Separate_Cam1_Cam3 == 0:
    STRING = ""
    PAW_ADD = "F"
    Remove_Old(PAW_ADD, Animal_Name)
    Comibne_Files(Animal_Name, STRING, PAW_ADD)

    PAW_ADD = "H"
    Remove_Old(PAW_ADD, Animal_Name)
    Comibne_Files(Animal_Name, STRING, PAW_ADD)

else:
    PAW_ADD = "F1"
    STRING = "_front_cam1"
    Remove_Old(PAW_ADD, Animal_Name)
    Comibne_Files(Animal_Name, STRING, PAW_ADD)

    PAW_ADD = "F3"
    STRING = "_front_cam3"
    Comibne_Files(Animal_Name, STRING, PAW_ADD)
    
    
    PAW_ADD = "H1"
    STRING = "_hind_cam1"
    Remove_Old(PAW_ADD, Animal_Name)
    Comibne_Files(Animal_Name, STRING, PAW_ADD)

    PAW_ADD = "H3"
    STRING = "_hind_cam3"
    Comibne_Files(Animal_Name, STRING, PAW_ADD)





print(colored("Please wait; Processing new function.",'yellow'))
