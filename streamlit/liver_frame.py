import datetime
import glob
import os
import re
import traceback
import warnings

import SimpleITK
import numpy as np
import pydicom
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from PIL import Image

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def map_ax(vol_arr, axx):
    map_xy = np.ndarray.max(vol_arr, axx)
    return map_xy


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


def window_low(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_min

    return window_image


def ext_organs_vol(vol_ct, center, w_size):  # Use Hounsfied to extract organs in vol_ct
    new_vol = np.zeros(np.shape(vol_ct))
    for slic in range(np.shape(vol_ct)[0]):
        new_vol[slic, :, :] = window_image(vol_ct[slic, :, :], center, w_size)
    return new_vol


def ext_organs_low(vol_ct, center, w_size):  # Use Hounsfied to extract organs in vol_ct
    new_vol = np.zeros(np.shape(vol_ct))
    for slic in range(np.shape(vol_ct)[0]):
        new_vol[slic, :, :] = window_low(vol_ct[slic, :, :], center, w_size)
    return new_vol


def im_read(f_path):
    if isinstance(f_path, list):
        image_file_list = f_path
        image_file_list = sort_by_instance_number(image_file_list)
        reader = SimpleITK.ImageSeriesReader()
        reader.SetFileNames(image_file_list)
    elif f_path.endswith('.list'):
        with open(f_path, 'r') as f:
            dicom_names = [x for x in f.read().split('\n') if len(x) > 0]
        if not os.path.exists(dicom_names[0]):
            image_file_list = [os.path.join(os.path.dirname(f_path), x) for x in dicom_names]
            image_file_list = sort_by_instance_number(image_file_list)
        reader = SimpleITK.ImageSeriesReader()
        reader.SetFileNames(image_file_list)
    else:
        reader = SimpleITK.ImageFileReader()
        reader.SetFileName(f_path)
    img = reader.Execute()
    arr = SimpleITK.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()
    return arr, spacing, origin, direction

#def read_img(root_path, patient_number, patient_no, file):
    #path = root_path + patient_number + kind + patient_no + file
    #im = Image.open("path")
    #return(im)

def compute_suv(image_file_list):
    estimated = False

    raw, spacing, origin, direction = im_read(image_file_list)

    f = pydicom.dcmread(image_file_list[0])

    try:
        weight_grams = float(f.PatientWeight) * 1000
    except:
        traceback.print_exc()
        weight_grams = 75000
        estimated = True

    try:
        # Get Scan time
        scantime = datetime.datetime.strptime(f.AcquisitionTime, '%H%M%S.%f')
        # Start Time for the Radiopharmaceutical Injection
        injection_time = datetime.datetime.strptime(
            f.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime, '%H%M%S.%f')
        # Half Life for Radionuclide # seconds
        half_life = float(f.RadiopharmaceuticalInformationSequence[0].RadionuclideHalfLife)
        # Total dose injected for Radionuclide
        injected_dose = float(f.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose)

        # Calculate decay
        decay = np.exp(-np.log(2) * ((scantime - injection_time).seconds) / half_life);
        # Calculate the dose decayed during procedure
        injected_dose_decay = injected_dose * decay;  # in Bq
    except:
        traceback.print_exc()
        decay = np.exp(-np.log(2) * (1.75 * 3600) / 6588);  # 90 min waiting time, 15 min preparation
        injected_dose_decay = 420000000 * decay;  # 420 MBq
        estimated = True

    # Calculate SUV # g/ml
    suv = raw * weight_grams / injected_dose_decay

    return suv, estimated, raw, spacing, origin, direction


def sort_by_instance_number(image_file_list):
    data = []
    for row in image_file_list:
        f = pydicom.dcmread(row)
        data.append({'f': row, 'n': f.InstanceNumber})
    data = sorted(data, key=lambda x: x['n'])
    return [x['f'] for x in data]

def ext_organs_low(vol_ct, center, w_size):
    new_vol = np.zeros(np.shape(vol_ct))
    for slic in range (np.shape(vol_ct)[0]):
        new_vol[slic,:,:] = window_low(vol_ct[slic,:,:], center, w_size)
    return new_vol        


def anisodiff3(stack, niter=1, kappa=50, gamma=0.1, step=(1., 1., 1.), option=1, ploton=False):
    """
    3D Anisotropic diffusion.

    Usage:
    stackout = anisodiff(stack, niter, kappa, gamma, option)

    Arguments:
            stack  - input stack
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (z,y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the middle z-plane will be plotted on every
                 iteration

    Returns:
            stackout   - diffused stack.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x,y and/or z axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones
    """

    # ...you could always diffuse each color channel independently if you
    # really want
    if stack.ndim == 4:
        warnings.warn("Only grayscale stacks allowed, converting to 3D matrix")
        stack = stack.mean(3)

    # initialize output array
    stack = stack.astype('float32')
    stackout = stack.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl

        showplane = stack.shape[0] // 2

        fig = pl.figure(figsize=(20, 5.5), num="Anisotropic diffusion")
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

        ax1.imshow(stack[showplane, ...].squeeze(), interpolation='nearest')
        ih = ax2.imshow(stackout[showplane, ...].squeeze(), interpolation='nearest', animated=True)
        ax1.set_title("Original stack (Z = %i)" % showplane)
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in range(niter):

        # calculate the diffs
        deltaD[:-1, :, :] = np.diff(stackout, axis=0)
        deltaS[:, :-1, :] = np.diff(stackout, axis=1)
        deltaE[:, :, :-1] = np.diff(stackout, axis=2)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gD = np.exp(-(deltaD / kappa) ** 2.) / step[0]
            gS = np.exp(-(deltaS / kappa) ** 2.) / step[1]
            gE = np.exp(-(deltaE / kappa) ** 2.) / step[2]
        elif option == 2:
            gD = 1. / (1. + (deltaD / kappa) ** 2.) / step[0]
            gS = 1. / (1. + (deltaS / kappa) ** 2.) / step[1]
            gE = 1. / (1. + (deltaE / kappa) ** 2.) / step[2]

        # update matrices
        D = gD * deltaD
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. don't as questions. just do it. trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:, :, :] -= D[:-1, :, :]
        NS[:, 1:, :] -= S[:, :-1, :]
        EW[:, :, 1:] -= E[:, :, :-1]

        # update the image
        stackout += gamma * (UD + NS + EW)

        if ploton:
            iterstring = "Iteration %i" % (ii + 1)
            ih.set_data(stackout[showplane, ...].squeeze())
            ax2.set_title(iterstring)
            fig.canvas.draw()
            # sleep(0.01)

    return stackout



def read_patient_information(root_path, patient_num):
    import pydicom
    medical_image = pydicom.read_file(root_path + patient_num + '//FDG//CT//IM10')
    return show_dcm_info(medical_image)


def show_dcm_info(dataset):
    import pandas as pd

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name

    data = [{
        'Key': 'Storage type',
        'Value': dataset.SOPClassUID
    }, {
        'Key': 'Patient\'s name',
        'Value': display_name
    }, {
        'Key': 'Patient id',
        'Value': dataset.PatientID
    }, {
        'Key': 'Patient\'s Age',
        'Value': dataset.PatientAge
    }, {
        'Key': 'Patient\'s Sex',
        'Value': dataset.PatientSex
    }, {
        'Key': 'Modality',
        'Value': dataset.Modality
    }]

    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        data.append({
            'Key': "Image size",
            'Value': "{rows:d} x {cols:d}, {size:d} bytes".format(
                rows=rows, cols=cols, size=len(dataset.PixelData))
        })
        if 'PixelSpacing' in dataset:
            data.append({
                'Key': "Pixel spacing",
                'Value': str(dataset.PixelSpacing)
            })

    df = pd.DataFrame(data=data, columns=['Key', 'Value'])
    return df


def draw_x_ray(root_path, patient_num):
    list_act = []
    info_act = []
    list_fdg = []
    info_fdg = []

    list_ct_act = []
    info_ct_act = []
    list_ct_fdg = []
    info_ct_fdg = []

    act_root = "ACT"
    fdg_root = "FDG"
    ct_act_root = "CT"
    ct_fdg_root = "CT"

    dir_C = "gdata-Acetate/"
    dir_F = "gdata-FDG/"
    dir_CT_act = "gdata-CT-act/"
    dir_CT_fdg = "gdata-CT-fdg/"

    act_dir = root_path + patient_num + "/" + act_root + "/PT/"
    act_paths = sorted(glob.glob(act_dir + "IM*"), key=natural_keys)

    fdg_dir = root_path + patient_num + "/" + fdg_root + "/PT/"
    fdg_paths = sorted(glob.glob(fdg_dir + "IM*"), key=natural_keys)

    ct_act_dir = root_path + patient_num + "/" + act_root + "/" + ct_act_root + "/"
    ct_act_paths = sorted(glob.glob(ct_act_dir + "IM*"), key=natural_keys)

    ct_fdg_dir = root_path + patient_num + "/" + fdg_root + "/" + ct_fdg_root + "/"
    ct_fdg_paths = sorted(glob.glob(ct_fdg_dir + "IM*"), key=natural_keys)

    for path in act_paths:
        num_of_path = path.replace(act_dir, "")
        pub_name = act_root + "-" + patient_num + "_" + num_of_path
        dataset = pydicom.dcmread(path)
        info_act.append(dataset)
        # write_jpg_w2b(dataset.pixel_array, pub_name, dir_C)
        list_act.append(dataset.pixel_array * dataset.RescaleSlope)

    for path in fdg_paths:
        num_of_path = path.replace(fdg_dir, "")
        pub_name = fdg_root + "-" + patient_num + "_" + num_of_path
        dataset = pydicom.dcmread(path)
        info_act.append(dataset)
        # write_jpg_w2b(daaset.pixel_array, pub_name, dir_F)
        list_fdg.append(dataset.pixel_array * dataset.RescaleSlope)

    for path in ct_act_paths:
        num_of_path = path.replace(ct_act_dir, "")
        pub_name = ct_act_root + "-" + patient_num + "_" + num_of_path
        dataset = pydicom.dcmread(path)
        info_ct_act.append(dataset)
        # write_jpg_b2w(dataset.pixel_array, pub_name, dir_CT_act)
        list_ct_act.append(convert_to_hu(dataset))

    for path in ct_fdg_paths:
        num_of_path = path.replace(ct_fdg_dir, "")
        pub_name = ct_fdg_root + "-" + patient_num + "_" + num_of_path
        dataset = pydicom.dcmread(path)
        info_ct_fdg.append(dataset)
        # write_jpg_b2w(dataset.pixel_array, pub_name, dir_CT_fdg)
        list_ct_fdg.append(convert_to_hu(dataset))

    num_of_file = np.shape(act_paths)[0] + 1
    # current layer index start with the first layer
    idx = 0

    ## load 3D patient:
    # act_3D = np.rot90(np.transpose(np.asarray(list_act), (2, 1, 0)),k=1, axes=(0, 2))[:,:,:]
    # fdg_3D = np.rot90(np.transpose(np.asarray(list_fdg), (2, 1, 0)),k=1, axes=(0, 2))[:,:,:]
    act_3D = compute_suv(act_paths)[0]
    fdg_3D = compute_suv(fdg_paths)[0]
    ct_act_3D = zoom(np.asarray(list_ct_act), (1, 0.375, 0.375))[:, :, :]
    ct_fdg_3D = zoom(np.asarray(list_ct_fdg), (1, 0.375, 0.375))[:, :, :]

    crop_vol_temp = ct_fdg_3D[80:200, 30:130, 30:130]

    pet_cropped = fdg_3D[80:200, 30:130, 30:130]

    crop_vol = anisodiff3(crop_vol_temp, niter=5, kappa=40, gamma=0.1, step=(1., 1., 1.), option=1, ploton=False)
    # crop_vol = anisodiff3(crop_vol_temp, niter=5, kappa=40, gamma=0.1, step=(1., 1., 1.), option=1, ploton=False)

    liver_vol = ext_organs_vol(crop_vol, 50, 90)
    liver_bg = ext_organs_vol(ct_fdg_3D,50,30)[:,80,:]
    #map_bone = map_ax(bone_vol, 1)
    img_sg = map_ax(fdg_3D[:,:,:],1)
    a = crop_vol_temp[45,:,:]

    # plt.imshow(map_bone, cmap="gray")
    # plt.colorbar()

    return liver_bg





def pct_images(selector_top, root_path, patient_num, center, wd_size, niter, kappa, wdwmi, wdwma, liver_vol_from=40,
               liver_vol_to=70, interations=2):
    list_act = []
    info_act = []
    list_fdg = []
    info_fdg = []

    list_ct_act = []
    info_ct_act = []
    list_ct_fdg = []
    info_ct_fdg = []

    act_root = "ACT"
    fdg_root = "FDG"
    ct_act_root = "CT"
    ct_fdg_root = "CT"

    dir_C = "gdata-Acetate/"
    dir_F = "gdata-FDG/"
    dir_CT_act = "gdata-CT-act/"
    dir_CT_fdg = "gdata-CT-fdg/"

    act_dir = root_path + patient_num + "/" + act_root + "/PT/"
    act_paths = sorted(glob.glob(act_dir + "IM*"), key=natural_keys)

    fdg_dir = root_path + patient_num + "/" + fdg_root + "/PT/"
    fdg_paths = sorted(glob.glob(fdg_dir + "IM*"), key=natural_keys)

    ct_act_dir = root_path + patient_num + "/" + act_root + "/" + ct_act_root + "/"
    ct_act_paths = sorted(glob.glob(ct_act_dir + "IM*"), key=natural_keys)

    ct_fdg_dir = root_path + patient_num + "/" + fdg_root + "/" + ct_fdg_root + "/"
    ct_fdg_paths = sorted(glob.glob(ct_fdg_dir + "IM*"), key=natural_keys)

    for path in act_paths:
        num_of_path = path.replace(act_dir, "")
        pub_name = act_root + "-" + patient_num + "_" + num_of_path
        dataset = pydicom.dcmread(path)
        info_act.append(dataset)
        # write_jpg_w2b(dataset.pixel_array, pub_name, dir_C)
        list_act.append(dataset.pixel_array * dataset.RescaleSlope)

    for path in fdg_paths:
        num_of_path = path.replace(fdg_dir, "")
        pub_name = fdg_root + "-" + patient_num + "_" + num_of_path
        dataset = pydicom.dcmread(path)
        info_act.append(dataset)
        # write_jpg_w2b(daaset.pixel_array, pub_name, dir_F)
        list_fdg.append(dataset.pixel_array * dataset.RescaleSlope)

    for path in ct_act_paths:
        num_of_path = path.replace(ct_act_dir, "")
        pub_name = ct_act_root + "-" + patient_num + "_" + num_of_path
        dataset = pydicom.dcmread(path)
        info_ct_act.append(dataset)
        # write_jpg_b2w(dataset.pixel_array, pub_name, dir_CT_act)
        list_ct_act.append(convert_to_hu(dataset))

    for path in ct_fdg_paths:
        num_of_path = path.replace(ct_fdg_dir, "")
        pub_name = ct_fdg_root + "-" + patient_num + "_" + num_of_path
        dataset = pydicom.dcmread(path)
        info_ct_fdg.append(dataset)
        # write_jpg_b2w(dataset.pixel_array, pub_name, dir_CT_fdg)
        list_ct_fdg.append(convert_to_hu(dataset))

    num_of_file = np.shape(act_paths)[0] + 1
    # current layer index start with the first layer
    idx = 0

    ## load 3D patient:
    # act_3D = np.rot90(np.transpose(np.asarray(list_act), (2, 1, 0)),k=1, axes=(0, 2))[:,:,:]
    # fdg_3D = np.rot90(np.transpose(np.asarray(list_fdg), (2, 1, 0)),k=1, axes=(0, 2))[:,:,:]
    act_3D = compute_suv(act_paths)[0]
    fdg_3D = compute_suv(fdg_paths)[0]
    ct_act_3D = zoom(np.asarray(list_ct_act), (1, 0.375, 0.375))[:, :, :]
    ct_fdg_3D = zoom(np.asarray(list_ct_fdg), (1, 0.375, 0.375))[:, :, :]

    crop_vol_temp = ct_fdg_3D[80:200, 30:130, 30:130]

    # use set_position
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')


    if selector_top == 'ACT':
        crop_vol_temp = ct_act_3D[80:200, 30:130, 30:130]
    else:
        crop_vol_temp = ct_fdg_3D[80:200, 30:130, 30:130]

    # crop_vol_temp = ct_fdg_3D[50:280, 0:135, 30:160]

    pet_cropped = act_3D[80:200, 0:135, 30:160]

    crop_vol = anisodiff3(crop_vol_temp, niter=niter, kappa=kappa, gamma=0.1, step=(1., 1., 1.), option=1, ploton=False)

    # https://radiopaedia.org/articles/windowing-ctact
    liver_vol = ext_organs_low(crop_vol, 50,90)
    liver_bg = ext_organs_vol(ct_fdg_3D, 50,30)[:,80,:]


    if selector_top == 'ACT':
        liver_bg = ext_organs_vol(ct_act_3D, wdwma, wdwmi)[:, 110, :]
        #bone_bg_show = ext_organs_vol(ct_act_3D, 1000, 600)[:, 110]
        img_sg = map_ax(act_3D[:, :, :], 1)
        
    else:
        liver_bg = ext_organs_vol(ct_fdg_3D, wdwma, wdwmi)[:, 110, :]
        #bone_bg_show = ext_organs_vol(ct_act_3D, 1000, 600)[:, 110]
        img_sg = map_ax(fdg_3D[:, :, :], 1)
       

    if selector_top == 'ACT':
        c = map_ax(act_3D[:, :, :], 1)
    else:
        c = map_ax(fdg_3D[:, :, :], 1)

    if selector_top == 'ACT':
        d = map_ax(act_3D[:, :, :], 1)
    else:
        d = map_ax(fdg_3D[:, :, :], 1)

    liver_bin = np.where((liver_vol >= liver_vol_from) & (liver_vol <= liver_vol_to), 1, 0)

    # ImageSliceViewer3D(bone_bin)

    from scipy import ndimage
    #med_bone_bin = ndimage.binary_closing(bone_bin, iterations=interations).astype(np.int8)
    #bone_pet1 = np.where(med_bone_bin == 1, pet_cropped, 0)
    isv3 = ImageSliceViewer3D(liver_vol, cmap="gray")

    # map_bin = map_ax(med_bone_bin, 1)
    #
    # comb_bin = bone_bin
    #
    # import region_growing as rgc
    # x0, y0, z0 = 0, 0, 0
    #
    # new_bone_bin = rgc.grow(med_bone_bin, x0, y0, z0, 1)
    #
    # ImageSliceViewer3D(new_bone_bin)
    # rev_bone_bin = np.where(new_bone_bin == 1, 0, 1)
    #
    # ImageSliceViewer3D(rev_bone_bin, cmap="gray")

    return crop_vol, liver_vol, c, d, liver_bg, img_sg, isv3.views()


class ImageSliceViewer3D:
    """
    ImageSliceViewer3D is for viewing volumetric image slices in jupyter or
    ipython notebooks.

    User can interactively change the slice plane selection for the image and
    the slice plane being viewed.

    Argumentss:
    Volume = 3D input image
    figsize = default(8,8), to set the size of the figure
    cmap = default('gray'), string for the matplotlib colormap. You can find
    more matplotlib colormaps on the following link:
    https://matplotlib.org/users/colormaps.html

    """

    def __init__(self, volume, figsize=(100, 100), cmap='gray'):
        self.volume = volume
        self.figsize = figsize
        self.cmap = cmap
        self.v = [np.min(volume), np.max(volume)]

    def views(self):
        self.vol1 = np.transpose(self.volume, [1, 2, 0])
        self.vol2 = np.rot90(np.transpose(self.volume, [2, 0, 1]), 3)  # rotate 270 degrees
        self.vol3 = np.transpose(self.volume, [0, 1, 2])
        maxZ1 = self.vol1.shape[2] - 1
        maxZ2 = self.vol2.shape[2] - 1
        maxZ3 = self.vol3.shape[2] - 1
        # ipyw.interact(self.plot_slice,
        #               z1=ipyw.IntSlider(min=0, max=maxZ1, step=1, continuous_update=False,
        #                                 description='Axial:'),
        #               z2=ipyw.IntSlider(min=0, max=maxZ2, step=1, continuous_update=False,
        #                                 description='Coronal:'),
        #               z3=ipyw.IntSlider(min=0, max=maxZ3, step=1, continuous_update=False,
        #                                 description='Sagittal:'))
        return (self.vol1, maxZ1, self.vol2, maxZ2, self.vol3, maxZ3, self.cmap, self.v)

    def plot_slice(self, z1, z2, z3):
        # Plot slice for the given plane and slice
        f, ax = plt.subplots(1, 3, figsize=self.figsize)
        # TODO: z1 - Axial, z2 - Coronal, z3 - Sagittal
        ax[0].imshow(self.vol1[:, :, z1], cmap=plt.get_cmap(self.cmap),
                     vmin=self.v[0], vmax=self.v[1])
        ax[1].imshow(self.vol2[:, :, z2], cmap=plt.get_cmap(self.cmap),
                     vmin=self.v[0], vmax=self.v[1])
        ax[2].imshow(self.vol3[:, :, z3], cmap=plt.get_cmap(self.cmap),
                     vmin=self.v[0], vmax=self.v[1])
        plt.show()

# def pct_images(selector_top, root_path, patient_num, center, wd_size, niter, kappa, wdwmi, wdwma):
# pct_images('ACT', 'C:/Users/cle/Downloads/', 'HJ19002 CT,PT', 600, 1000, 1, 40, 1000, 600)

# 2 Download button (2 lastest images)

