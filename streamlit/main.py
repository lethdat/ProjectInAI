import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import SimpleITK as sitk
from IPython.display import Image 
from skimage.transform import resize
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import utils_med
from utils_med import compute_suv, convert_to_hu
import glob
from visualize3D import showPETCT3D
import pydicom
import ipywidgets as ipyw 

import torch
import torchtuples as tt

from pycox.datasets import metabric
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv

# Kaplan-Meier curve
from string import ascii_lowercase# Visualisation
import seaborn as sns
#sns.set(style='dark', context='talk')# Kaplan-Meier curve
from lifelines import KaplanMeierFitter 
from sksurv.nonparametric import kaplan_meier_estimator

import missingno
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter, WeibullAFTFitter , LogNormalAFTFitter, LogLogisticAFTFitter , PiecewiseExponentialRegressionFitter
from lifelines.utils import k_fold_cross_validation
from IPython.display import HTML

from radiomics import featureextractor 
import nrrd
import traceback
import datetime
import warnings
warnings.filterwarnings("ignore")
import csv

import pandas as pd # for reading and writing tables
import skimage # for image processing and visualizations
import sklearn # for machine learning and statistical models
from pathlib import Path # help manage files
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]

def find(s, el):
    for i in s.index:
        if s[i] == el:
            return i
    return None

def dir_selector(folder_path='.'):
    dirnames = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    selected_folder = st.sidebar.selectbox('Select a folder', dirnames)
    if selected_folder is None:
        return None
    return os.path.join(folder_path, selected_folder)

def plot_slice(volCT, volPET, slice_ix):
    fig, ax = plt.subplots()
    plt.axis('off')
    selected_sliceCT = volCT[slice_ix, :, :]
    selected_slicePET= volPET[slice_ix,:, :]
    ax.imshow(selected_sliceiCT, origin='lower', cmap='gray')
    ax.imshow(selected_slicePET, origin="lower", cmap='hot', alpha=0.5)
    return fig

def resize_volume(img, new_shape):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = new_shape[-1]
    desired_width = new_shape[0]
    desired_height = new_shape[1]
    # Get current depth
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    # Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    # Rotate
    img = ndimage.rotate(img, 90, reshape=False)
    # Resize across z-axis
    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def show_dataset(ds, indent):
    for elem in ds:
        if elem.VR == "SQ":
            indent += 4 * " "
            for item in elem:
                show_dataset(item, indent)
            indent = indent[4:]
        print(indent + str(elem))

def read_CTwInfo(id_):
    ct_dir_ = "../data_n_400/AIDATA_CT_20201105(n=246)_20210202/CT_"+id_+"/"
    list_ct_ = []
    info_ct_ = []
    ct_path_ = sorted(glob.glob(ct_dir_+ "IM*"),key=natural_keys)
    for path_ in ct_path_:
        num_of_path_ = path_.replace(ct_dir_,"")
        pub_name_ = "CT-"+id_+"_"+num_of_path_
        dataset_ = pydicom.dcmread(path_)
        info_ct_.append(dataset_)
        list_ct_.append(convert_to_hu(dataset_))
    return np.asarray(list_ct_), info_ct_

def read_PETwInfo(id_):
    pet_dir_ = "../data_n_400/AIDATA_PET_20201105(n=246)_20210202/"+id_+"/"
    list_pet_=[]
    info_pet_=[]
    pet_path_ = sorted(glob.glob(pet_dir_+"IM*"),key=natural_keys)
    for path_ in pet_path_:
        num_of_path_ = path_.replace(pet_dir_,"")
        pub_name_ = "PET-"+id_+"_"+num_of_path_
        dataset_ = pydicom.dcmread(path_)
        info_pet_.append(dataset_)
        list_pet_.append(dataset_.pixel_array*dataset_.RescaleSlope)
    pet_image_file_list_ = utils_med.sort_by_instance_number(pet_path_)
    suv_, _,_,_,_,_ = compute_suv(pet_image_file_list_)
    return suv_, info_pet_

def getIndexes(dfObj, value):
    ''' Get index positions of value in dataframe i.e. dfObj.'''
    listOfPos = list()
    # Get bool dataframe with True at positions where the given value exists
    result = dfObj.isin([value])
    # Get list of columns that contains the value
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    # Iterate over list of columns and fetch the rows indexes where value exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append((row, col))
    # Return a list of tuples indicating the positions of value in the dataframe
    return listOfPos

def plot_slice_yz(volCT_, volPET_, slice_ix_, suv_b_, suv_t_):
    fg_x, ax = plt.subplots(figsize=(2,2))
    plt.axis('off')
    selected_slice_CT  = volCT_[slice_ix_, :, :]
    selected_slice_PET =volPET_[slice_ix_, :, :]
    ax.imshow(selected_slice_CT,  origin='lower', cmap='gray')
    ax.imshow(selected_slice_PET, origin='lower', cmap='hot', vmin=suv_b_, vmax=suv_t_,alpha=0.5)
    st.pyplot(fg_x)
    return

def plot_slice_xz(volCT_, volPET_, slice_iy_, suv_b_, suv_t_):
    fg_y, ax = plt.subplots(figsize=(2,2))
    plt.axis('off')
    selected_slice_CT  = volCT_[:, slice_iy_, :]
    selected_slice_PET =volPET_[:, slice_iy_, :]
    ax.imshow(selected_slice_CT,  origin='lower', cmap='gray')
    ax.imshow(selected_slice_PET, origin='lower', cmap='hot', vmin=suv_b_, vmax=suv_t_,alpha=0.5)
    st.pyplot(fg_y)
    return

def plot_slice_xy(volCT_, volPET_, slice_iz_, suv_b_, suv_t_):
    fg_z, ax = plt.subplots(figsize=(2,2))
    plt.axis('off')
    selected_slice_CT  = volCT_[:, :, slice_iz_]
    selected_slice_PET =volPET_[:, :, slice_iz_]
    ax.imshow(selected_slice_CT,  origin='lower', cmap='gray')
    ax.imshow(selected_slice_PET, origin='lower', cmap='hot', vmin=suv_b_, vmax=suv_t_,alpha=0.5)
    st.pyplot(fg_z)
    return

def plot_hazard_rate(surv_,time_):
    text_kwargs = dict(ha='left', va='top', fontsize=10, color='blue')
    surv_.iloc[:, :].plot()
    plt.ylabel('Probability of survival P(t | x)')
    plt.ylim(0,1)
    plt.axvline(x=time_, color='r', linestyle='-.')
    plt.text(time_+0.1, 0.6, 't = ' + str(time_), **text_kwargs)
    _ = plt.xlabel('Time')
    
    st.pyplot(plt)
    return

# START - SESSION STATE
SST_SELECTED_FOLDER = 'selected folder'
SST_CENTER = 'center'
SST_WD_SIZE = 'wd_size'
SST_AXIAL_SLIDER = 'axial_slider'
SST_SAGITTAL_SLIDER = 'sagittal_slider'
SST_CORONAL_SLIDER = 'coronal_slider'


def add_to_session_state_if_absent(key, value):
    if key not in st.session_state:
        add_to_session_state(key, value)


def add_to_session_state(key, value):
    st.session_state[key] = value


def get_from_session_state(key):
    return st.session_state[key]

add_to_session_state_if_absent(SST_AXIAL_SLIDER, 0)
add_to_session_state_if_absent(SST_SAGITTAL_SLIDER, 0)
add_to_session_state_if_absent(SST_CORONAL_SLIDER, 0)

# END - SESSION STATE

# START - CACHE
# END - CACHE

# START - UTIL

## fix folder to os(fix error when macos, window://, //)
def extract_root_path_and_patient_num(selected_folder):
    opts = selected_folder.rsplit(os.sep, 1)
    return opts[0] + os.sep, opts[1]

st.sidebar.title('Lung prognosis of NSCLC')
csv_file = "../data_n_400/AIDATA_NSCLC+SCLC_20201130_tissue(n246)_20210202.xlsx"

csv_feature = "../radiomics-246patients.csv"
df_clinical_ = pd.read_excel(open(csv_file,"rb"), sheet_name="Sheet1")

df_feature_  = pd.read_csv(csv_feature,sep='\t')

list_patient_= df_clinical_["PatientID"]
PatientID_choose = st.sidebar.selectbox('Select PatientID', list_patient_, index=0)
ready_button = st.sidebar.button("Select & read", key=False)
stop_button = st.sidebar.button("stop", key=False)

if (PatientID_choose is not None):
    try:
        ## Read Feature:
        patient_feature = df_feature_.loc[df_feature_['ID'] == PatientID_choose]
        patient_feature.drop('ID', inplace=True, axis=1)
        patient_feature.rename(columns = {'Survival.time':'duration', 'Deadstatus.event':'event'}, inplace = True)
        dur_t = patient_feature.iloc[0]['duration']
        
        patient_feature = patient_feature.iloc[: , 1:]
        ## Mapper feature
        all_cols = patient_feature.columns.values.tolist()
        cols_leave = ['gender', 'Smoking.status', 'event']
        cols_standardize = [col_ids for col_ids in all_cols if col_ids not in cols_leave]
        standardize = [([col], StandardScaler()) for col in cols_standardize]
        leave = [(col, None) for col in cols_leave]
        x_mapper = DataFrameMapper(standardize + leave)
        x_feature_t = x_mapper.fit_transform(patient_feature).astype('float32')
        x_feature = x_mapper.transform(patient_feature).astype('float32')

        get_target = lambda df: (df['duration'].values, df['event'].values)

        durations_test, events_test = get_target(patient_feature)
        
        ## Load model
        in_features = x_feature.shape[1]
        num_nodes = [8,8]
        out_features = 1
        batch_norm = True
        dropout = 0.01
        output_bias = False
        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,dropout,output_bias=output_bias)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = CoxPH(net, tt.optim.Adam)
        model.load_net('../pre-trainedDeepSurv-128.pt')
        
        ## Patient's Surv
        surv = model.predict_surv_df(x_feature)

        ## Read DICOM by PatientID:
        vol_CT, info_ct = read_CTwInfo(PatientID_choose)
        vol_PET, info_pet = read_PETwInfo(PatientID_choose)
        vol_CT  = np.rot90(resize(vol_CT , np.shape(vol_PET)), k=2, axes=(1,0))
        vol_PET = np.rot90(resize(vol_PET, np.shape(vol_PET)), k=2, axes=(1,0))
        pet_roi = vol_PET
        ct_roi  = vol_CT
        
        (shape_x, shape_y, shape_z) = np.shape(pet_roi)

        slice_ix = st.sidebar.slider('Axial'   , 0, shape_x, int(shape_x/2),key=False)
        slice_iy = st.sidebar.slider('Sagittal', 0, shape_y, int(shape_y/2),key=False)
        slice_iz = st.sidebar.slider('Coronal' , 0, shape_z, int(shape_z/2),key=False)
        (suv_min, suv_max) = st.sidebar.slider("SUV scale", 0.0, np.max(pet_roi), (2.5, 12.0), 0.5)
        plot_slice_yz(ct_roi, pet_roi, slice_ix, suv_min, suv_max)
        plot_slice_xz(ct_roi, pet_roi, slice_iy, suv_min, suv_max)
        plot_slice_xy(ct_roi, pet_roi, slice_iz, suv_min, suv_max)
        plot_hazard_rate(surv,dur_t)
        
        index_id = getIndexes(df_clinical_,PatientID_choose)[0][0]
        st.sidebar.text_area("Patient's Infomation",str(df_clinical_.iloc[index_id][:10]))
#        st.pyplot(fig)
    except RuntimeError:
        pass
