import subprocess, os, sys, tempfile, atexit, shutil, guidata
import numpy as np 
import pandas as pd
from glob import glob
from guidata.dataset.datatypes import DataSet, BeginGroup, EndGroup, DataSetGroup, BeginTabGroup, EndTabGroup, ValueProp
from guidata.dataset.dataitems import (FloatItem, IntItem, FileOpenItem, DirectoryItem, FloatArrayItem, BoolItem, MultipleChoiceItem, ButtonItem, StringItem, StringItem)
from guidata.qthelpers import create_action, add_actions, get_std_icon
from guidata.utils import update_dataset

class Numbre_of_Animals_GUI(DataSet):
    """
    DataSet test
    Please select number of animals in the study
    """
    Numbre_of_Animals = IntItem("Numbre of Animals", default=4, min=1, slider=False)



global Numbre_of_Animals
_app = guidata.qapplication()
GUI_Numbre_of_Animals = Numbre_of_Animals_GUI("Number of Animals")
GUI_Numbre_of_Animals.edit()
Numbre_of_Animals = GUI_Numbre_of_Animals.Numbre_of_Animals
prop1 = ValueProp(False)



def Code_array_to_string(Input):
    OUTPUT = ""
    for N, L0 in enumerate(np.nditer(Input)): 
        if N == 0:
            OUTPUT = OUTPUT +str(L0)
        else:
            OUTPUT = OUTPUT + "," +str(L0)
    return(OUTPUT)



def Coder_Paw_List(Paw_List):
    Temp = []
    while True:
        temp = ""
        Paw_List = Paw_List[Paw_List.find("'")+1:]
        for L0 in Paw_List:
            if L0 == "'":
                break
            temp = temp+L0
        Temp.append(temp)

        if Paw_List[Paw_List.find("'")+1:] == "]":
            break; break
        else:
            Paw_List = Paw_List[Paw_List.find(",")+2:]



    Paw = ""
    if len(Paw_List) > 1:
        for N, L0 in enumerate(Temp):
            if N == 0:
                Paw = Paw + L0
            else:
                Paw = Paw + ","+ L0
    else:
        Paw = Temp[0]
    return(Paw)


class Run_Step_By_Step(DataSet):
    def Make_3D_Coordinates(self, item, value, parent):
        Code_To_Be_Processed = "python3 " + os.path.join(self.Code_Path, "Join_cam1_cam3.py") + " -animal " + self.Animal_Name
        subprocess.call(Code_To_Be_Processed, shell=True)

        Code_To_Be_Processed = "python3 " + os.path.join(self.Code_Path, "Maker_3D_Coordinates.py") + " -animal " + self.Animal_Name
        subprocess.call(Code_To_Be_Processed, shell=True)



    def Joint_Edition(self, item, value, parent):
        if self.plot_flag_joint_adjustment == False:
            plot_flag = 0
        else:
            plot_flag = 1

        Code_To_Be_Processed = "python3 " + os.path.join(self.Code_Path, "Knee_Adjusment.py") + " -animal " + self.Animal_Name + " -ip " + self.Images_General_Path + \
            " -rl " + Code_array_to_string(self.Animal_list) + " -ak1 " + Code_array_to_string(self.Ankle_Knee_Length_cam1) + " -hk1 " + Code_array_to_string(self.Hip_Knee_Length_cam1) + " -se1 " + Code_array_to_string(self.Shoulder_Elbow_Length_cam1) + \
            " -eh1 " + Code_array_to_string(self.Elbow_Hand_Length_cam1) + " -ak2 " + Code_array_to_string(self.Ankle_Knee_Length_cam3) + " -hk2 " + Code_array_to_string(self.Hip_Knee_Length_cam3) + " -se2 " + Code_array_to_string(self.Shoulder_Elbow_Length_cam3) + \
            " -eh2 " + Code_array_to_string(self.Elbow_Hand_Length_cam3) + " -pf " + str(plot_flag)
        subprocess.call(Code_To_Be_Processed, shell=True)



    def Featrues_Extraction(self, item, value, parent):
        if self.Separate_By_Speed == False:
            Separate_By_Speed_flag = 0
        else:
            Separate_By_Speed_flag = 1
        # self.Threshold_Speed /= 100

        Code_To_Be_Processed = "python3 " + os.path.join(self.Code_Path, "Analyzer_3D_Coordinates.py") + " -animal " + self.Animal_Name + " -ts " + str(self.Threshold_Speed) + " -mcf " + str(self.Minimum_Consecutive_Frames) + \
            " -separate " + str(Separate_By_Speed_flag) + " -step " + str(self.Steps_Speed)
        subprocess.call(Code_To_Be_Processed, shell=True)
        # self.Threshold_Speed *= 100



    def Run_Time_Series_Maker(self, item, value, parent):
        if self.Separate_cam1_cam3 == False:
            Separate_cam1_cam3 = 0
        else:
            Separate_cam1_cam3 = 1

        Code_To_Be_Processed = "python3 " + os.path.join(self.Code_Path, "Files_Combines.py") + " -animal " + self.Animal_Name + " -separate " + str(Separate_cam1_cam3)
        subprocess.call(Code_To_Be_Processed, shell=True)

        Code_To_Be_Processed = "python3 " + os.path.join(self.Code_Path, "Time_Series_Maker.py") + " -animal " + self.Animal_Name + " -separate " + str(Separate_cam1_cam3) + " -bins " + str(self.Number_of_Bins) + " -mfn " + str(self.Maximum_Acceptable_Number_of_Frames_In_One_Stride)
        subprocess.call(Code_To_Be_Processed, shell=True)


    def Update_Plot_Info_Fun(self, item, value, parent):
        CurrentPath = "."
        Data_Path_All = sorted(glob(os.path.join(CurrentPath, "Stride_All_Final_Concatenated_Matrix_F*.csv")))
        Data = pd.read_csv(Data_Path_All[0], sep=',', index_col=0)
        self.Front_List = Data['Experiment'].unique().tolist()

        Data_Path_All = sorted(glob(os.path.join(CurrentPath, "Stride_All_Final_Concatenated_Matrix_H*.csv")))
        Data = pd.read_csv(Data_Path_All[0], sep=',', index_col=0)
        self.Hind_List = Data['Experiment'].unique().tolist()


    def PLOT(self, item, value, parent):
        if self.Separate_cam1_cam3 == False:
            Separate_cam1_cam3 = 0
        else:
            Separate_cam1_cam3 = 1

        if self.Plot_All == False:
            Plot_All_Flag = 0
        else:
            Plot_All_Flag = 1

        if self.Plot_Sep == False:
            Plot_Sep_Flag = 0
        else:
            Plot_Sep_Flag = 1

        Front = Coder_Paw_List(self.Front_List)
        Hind = Coder_Paw_List(self.Hind_List)

        Code_To_Be_Processed = "python3 " + os.path.join(self.Code_Path, "Statistics.py") + " -animal " + self.Animal_Name + " -separate " + \
            str(Separate_cam1_cam3) + " -bins " + str(self.Number_of_Bins) + " -pa " + str(Plot_All_Flag) + " -pas " + str(Plot_Sep_Flag) + \
            " -sn " + self.Study_Name + " -ymin " + str(self.Y_Limit_Min) + " -ymax " + str(self.Y_Limit_Max) + " -fs " + str(self.Font_Size) + \
            " -xplf " + Front + " -xplh " + Hind

        subprocess.call(Code_To_Be_Processed, shell=True)


    def Video_Maker(self, item, value, parent):
        if self.Separate_cam1_cam3 == False:
            Separate_cam1_cam3 = 0
        else:
            Separate_cam1_cam3 = 1
        # self.Higher_Speed_Range /= 100
        # self.Lower_Speed_Range /= 100

        Code_To_Be_Processed = "python3 " + os.path.join(self.Code_Path, "Video_Maker.py") + " -animal " + self.Animal_Name + " -separate " + \
            str(Separate_cam1_cam3) + " -bins " + str(self.Number_of_Bins) + " -nf " + str(self.Number_Frames) + " -hsr " + str(self.Higher_Speed_Range) + \
            " -lsr " + str(self.Lower_Speed_Range) + " -ta " + str(self.Time_All) + " -rx " + str(self.X_Range) + " -ry " + str(self.Y_Range) + " -rz " + str(self.Z_Range) 

        subprocess.call(Code_To_Be_Processed, shell=True)

        # self.Higher_Speed_Range *= 100
        # self.Lower_Speed_Range *= 100

    """
    DataSet test
    This is a Graphical User Interface for getting the initial values and genral paths.
    """
    Current_Path = os.getcwd()
    Code_Path_temp = os.path.realpath(__file__)
    Code_Name_loc = Code_Path_temp[::-1].find('/')
    Code_Path_temp = Code_Path_temp[:-1*Code_Name_loc-1]

    _bg1 = BeginGroup("1) Parameters to combine the tracked coordinates for cam1-2 and cam3-4 and 3D reconstruction")
    Code_Path = DirectoryItem("Code Path", Code_Path_temp).set_pos(col=0)
    Animal_Name = StringItem("Animal Name", default='rat').set_pos(col=3)
    Image_Use = BoolItem("Needing any Image Processing", default=False).set_pos(col=0).set_prop("display", store=prop1)
    Images_General_Path = DirectoryItem("Path for Images", '/Volumes/SCI_Storage/Aging').set_pos(col=1).set_prop("display", active=prop1)
    Make_3D_Coordinates = ButtonItem("Make 3D Coordinates", callback=Make_3D_Coordinates).set_pos(col=6)
    _eg1 = EndGroup("Parameters to combine the tracked coordinates for cam1-2 and cam3-4 and 3D reconstruction")
	


    _bg2 = BeginGroup("2) Parameters for joint edition (OPTIONAL, Step (1) is needed)")
    Hip_Knee_Length_cam1 = FloatArrayItem("Hip Kne L Cam1-2", default=np.ones([1,Numbre_of_Animals], dtype=int)*35)
    Ankle_Knee_Length_cam1 = FloatArrayItem("Ank Kne L Cam1-2", default=np.ones([1,Numbre_of_Animals], dtype=int)*30).set_pos(col=1)
    Shoulder_Elbow_Length_cam1 = FloatArrayItem("Sho Elb L Cam1-2", default=np.ones([1,Numbre_of_Animals], dtype=int)*33).set_pos(col=2)
    Elbow_Hand_Length_cam1 = FloatArrayItem("Elb Han L Cam1-2", default=np.ones([1,Numbre_of_Animals], dtype=int)*28).set_pos(col=3)
    Animal_list = FloatArrayItem("Animal List", default=np.arange(1,Numbre_of_Animals+1)).set_pos(col=4)

    Hip_Knee_Length_cam3 = FloatArrayItem("Hip Kne L Cam3-4", default=np.ones([1,Numbre_of_Animals], dtype=int)*35).set_pos(col=0)
    Ankle_Knee_Length_cam3 = FloatArrayItem("Ank Kne L Cam3-4", default=np.ones([1,Numbre_of_Animals], dtype=int)*30).set_pos(col=1)
    Shoulder_Elbow_Length_cam3 = FloatArrayItem("Sho Elb L Cam3-4", default=np.ones([1,Numbre_of_Animals], dtype=int)*33).set_pos(col=2)
    Elbow_Hand_Length_cam3 = FloatArrayItem("Elb Han L Cam3-4", default=np.ones([1,Numbre_of_Animals], dtype=int)*28).set_pos(col=3)
    plot_flag_joint_adjustment = BoolItem("Make Frames", default=False).set_pos(col=4)
    # Images_General_Path = DirectoryItem("Images_General_Path", Current_Path).set_pos(col=1)
    
    Joint_Edition = ButtonItem("Joint Edition", callback=Joint_Edition).set_pos(col=5)
    # plot_flag_joint_adjustment = BoolItem("plot_flag_joint_adjustment", default=False).set_pos(col=6)
    _eg2 = EndGroup("Joint edition if needed")


    _bg3 = BeginGroup("3) Parameters for feature extarction from the 3D reconstructed data (Step 1 is needed)")
    Minimum_Consecutive_Frames = IntItem("Minimum Consecutive Frames", default=70, min=0, slider=False)
    Threshold_Speed = FloatItem("Threshold Speed", default=0.12, unit="cm/sec", slider=False).set_pos(col=1)
    Steps_Speed = FloatItem("Steps Speed", default=0.04, unit="cm/sec", slider=False).set_pos(col=2)
    Separate_By_Speed = BoolItem("Separate by Speed", default=False).set_pos(col=3)
    Featrues = ButtonItem("Featrues", callback=Featrues_Extraction).set_pos(col=4)
    _eg3 = EndGroup("Feature extarction from the 3D reconstructed data")


    _bg4 = BeginGroup("4) Parameters to combine cameras data and create time series (Steps 1 and 3 are needed)")
    Separate_cam1_cam3 = BoolItem("Separate Cam1-2 Cam3-4", default=False).set_pos(col=0)
    Number_of_Bins = IntItem("Number of Bins in Time Series", default=200, slider=False).set_pos(col=1)
    Maximum_Acceptable_Number_of_Frames_In_One_Stride = IntItem("Maximum Acceptable Number of Frames in a Stride", default=200, slider=False).set_pos(col=2)
    Time_series = ButtonItem("Time series", callback=Run_Time_Series_Maker).set_pos(col=3)
    _eg4 = EndGroup("Combine front and hind paws data and create time series data")


    _bg5 = BeginGroup("5) Parameters to plot time series data (Step 4 or a Stride_Feature CSV file formate is needed)")
    Study_Name = StringItem("Study Name", default="Choose_a_Name")
    Update_Plot_Info = ButtonItem("Update Experiment Lists", callback=Update_Plot_Info_Fun).set_pos(col=3)
    Front_List = StringItem("List Exp Front", default=['Use update and then select the needed ones then push Plot button'])
    Hind_List = StringItem("List Exp Hind", default=['Use update and then select the needed ones then push Plot button'])
    Y_Limit_Min = IntItem("Y Limit Min", default=0, slider=False)
    Y_Limit_Max = IntItem("Y Limit Max", default=180, slider=False).set_pos(col=1)
    Font_Size = IntItem("Font Size", default=10, slider=False).set_pos(col=2)
    Plot_All = BoolItem("Plot All Conditions", default=True).set_pos(col=3)
    Plot_Sep = BoolItem("Plot Separately", default=True).set_pos(col=4)
    PLOT = ButtonItem("Make Plots", callback=PLOT).set_pos(col=5)
    _eg5 = EndGroup("Plot time series data")


    _bg6 = BeginGroup("6) Parameters to make video form the results (Steps 1 and 3 are needed)")
    Number_Frames = IntItem("Total Number of Frames in a Trial", default=1000, slider=False).set_pos(col=0)
    Time_All = IntItem("Total Time for a Trial", default=4, unit="sec", slider=False).set_pos(col=1)
    Higher_Speed_Range = FloatItem("Speed Plot Highest Value", default=0.4, unit="m/sec", slider=False).set_pos(col=2)
    Lower_Speed_Range = FloatItem("Speed Plot Lowest Value", default=-0.1, unit="m/sec", slider=False).set_pos(col=3)
    X_Range = IntItem("Maximum Variation from Center in X", default=200, unit="mm", slider=False).set_pos(col=0)
    Y_Range = IntItem("Maximum Variation from Center in Y", default=50, unit="mm", slider=False).set_pos(col=1)
    Z_Range = IntItem("Maximum Variation from Center in Z", default=50, unit="mm", slider=False).set_pos(col=2)
    Video = ButtonItem("Make Videos", callback=Video_Maker).set_pos(col=3)
    _eg6 = EndGroup("Making video form the results")




if __name__ == "__main__":
    GUI_Data = Run_Step_By_Step("Run step by step")
    GUI_Data.edit(size=(10,5))
    print(GUI_Data)



