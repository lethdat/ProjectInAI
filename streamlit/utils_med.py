# Import libraries
import glob
import pydicom
import numpy as np
import time
import os
import re
from scipy import ndimage
from PIL import Image, ImageOps
from zipfile import ZipFile
import itertools
import scipy.ndimage
from scipy.ndimage import rotate
import pandas as pd
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
import SimpleITK as sitk
import traceback
import datetime
import warnings
warnings.filterwarnings("ignore")


def sort_by_instance_number(image_file_list):
    data = []
    for row in image_file_list:
        f=pydicom.dcmread(row)
        data.append({'f':row,'n':f.InstanceNumber})
    data=sorted(data,key=lambda x: x['n'])
    return [x['f'] for x in data]

def imread(fpath):
    if isinstance(fpath,list):
        image_file_list = fpath
        image_file_list = sort_by_instance_number(image_file_list)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(image_file_list)
    elif fpath.endswith('.list'):
        with open(fpath,'r') as f:
            dicom_names = [x for x in f.read().split('\n') if len(x) > 0]
        if not os.path.exists(dicom_names[0]):
            image_file_list = [os.path.join(os.path.dirname(fpath),x) for x in dicom_names]
            image_file_list = sort_by_instance_number(image_file_list)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(image_file_list)
    else:
        reader= sitk.ImageFileReader()
        reader.SetFileName(fpath)
    img = reader.Execute()
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    return arr,spacing,origin,direction

def compute_suv(image_file_list):

    estimated = False

    raw,spacing,origin,direction = imread(image_file_list)

    f=pydicom.dcmread(image_file_list[0])

    try:
        weight_grams = float(f.PatientWeight)*1000
    except:
        traceback.print_exc()
        weight_grams = 75000
        estimated = True

    try:
        # Get Scan time
        scantime = datetime.datetime.strptime(f.AcquisitionTime,'%H%M%S')
        # Start Time for the Radiopharmaceutical Injection
        injection_time = datetime.datetime.strptime(f.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime,'%H%M%S.%f')
        # Half Life for Radionuclide # seconds
        half_life = float(f.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
        # Total dose injected for Radionuclide
        injected_dose = float(f.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)

        # Calculate decay
        decay = np.exp(-np.log(2)*((scantime-injection_time).seconds)/half_life);
        # Calculate the dose decayed during procedure
        injected_dose_decay = injected_dose*decay; # in Bq
    except:
        traceback.print_exc()
        decay = np.exp(-np.log(2)*(1.75*3600)/6588); # 90 min waiting time, 15 min preparation
        injected_dose_decay = 420000000 * decay; # 420 MBq
        estimated = True

    # Calculate SUV # g/ml
    suv = raw*weight_grams/injected_dose_decay

    return suv, estimated, raw,spacing,origin,direction

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def convert_to_hu(dicom_file):
    bias = dicom_file.RescaleIntercept
    slope = dicom_file.RescaleSlope
    pixel_values = dicom_file.pixel_array
    new_pixel_values = (pixel_values * slope) + bias
    return new_pixel_values

def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max

    return window_image


