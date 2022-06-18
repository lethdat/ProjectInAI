import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
import ipywidgets as ipyw
import streamlit as st

class showPETCT3D:
    """ 
    Modified from ImageSlice3D to show both PET and CT in same volume
    
    """
    
    def __init__(self, volumeCT, volumePET, figsize=(100,100), cmapCT='gray', cmapPET="hot"):
        self.volumeCT  = volumeCT
        self.volumePET = volumePET
        self.figsize   = figsize
        self.cmapCT    = cmapCT
        self.cmapPET   = cmapPET
        self.vCT = [np.min(volumeCT), np.max(volumeCT)]
        self.vPET = [np.min(volumePET), np.max(volumePET)]
        # Call to select slice plane
        ipyw.interact(self.views)
    
    def views(self):
        self.volCT1 = np.transpose(self.volumeCT, [1,2,0])
        self.volCT2 = np.rot90(np.transpose(self.volumeCT, [2,0,1]), 3) #rotate 270 degrees
        self.volCT3 = np.transpose(self.volumeCT, [0,1,2])
        self.volPET1 = np.transpose(self.volumePET, [1,2,0])
        self.volPET2 = np.rot90(np.transpose(self.volumePET, [2,0,1]), 3) #rotate 270 degrees
        self.volPET3 = np.transpose(self.volumePET, [0,1,2])
        maxZ1 = self.volCT1.shape[2] - 1
        maxZ2 = self.volCT2.shape[2] - 1
        maxZ3 = self.volCT3.shape[2] - 1
        
        ipyw.interact(self.plot_slice, 
            z1=ipyw.IntSlider(min=0, max=maxZ1, step=1, continuous_update=False, 
            description='Axial:'), 
            z2=ipyw.IntSlider(min=0, max=maxZ2, step=1, continuous_update=False, 
            description='Coronal:'),
            z3=ipyw.IntSlider(min=0, max=maxZ3, step=1, continuous_update=False, 
            description='Sagittal:'))
    def plot_slice(self, z1, z2, z3):
        # Plot slice for the given plane and slice
        f,ax = plt.subplots(1,3, figsize=self.figsize)
        #print(self.figsize)
        #self.fig = plt.figure(figsize=self.figsize)
        #f(figsize = self.figsize)
        ax[0].imshow(self.volCT1[:,:,z1], cmap=plt.get_cmap(self.cmapCT), 
            vmin=self.vCT[0], vmax=self.vCT[1])
        ax[1].imshow(np.flip(self.volCT2[:,:,z2],1), cmap=plt.get_cmap(self.cmapCT), 
            vmin=self.vCT[0], vmax=self.vCT[1])
        ax[2].imshow(self.volCT3[:,:,z3], cmap=plt.get_cmap(self.cmapCT), 
            vmin=self.vCT[0], vmax=self.vCT[1])
        
        ax[0].imshow(self.volPET1[:,:,z1], cmap=plt.get_cmap(self.cmapPET), 
            vmin=self.vPET[0], vmax=self.vPET[1], alpha=0.5)
        ax[1].imshow(np.flip(self.volPET2[:,:,z2],1), cmap=plt.get_cmap(self.cmapPET), 
            vmin=self.vPET[0], vmax=self.vPET[1], alpha=0.5)
        ax[2].imshow(self.volPET3[:,:,z3], cmap=plt.get_cmap(self.cmapPET), 
            vmin=self.vPET[0], vmax=self.vPET[1], alpha=0.5)
        
        ax[0].axvline(x=z3,color='red',linewidth=5.0)
        ax[0].axhline(y=z2,color='green',linewidth=5.0)
        
        ax[1].axhline(y=z1,color='blue',linewidth=5.0)
        ax[1].axvline(x=z3,color='red',linewidth=5.0)
        
        ax[2].axhline(y=z1,color='blue',linewidth=5.0)
        ax[2].axvline(x=z2,color='green',linewidth=5.0)
        st.pyplot(f)
