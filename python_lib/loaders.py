from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
import random
warnings.filterwarnings("ignore")

def spill(lst):
    new = []
    for i in lst:
        new += i
    return new

def pad(arr, final_length = 64):
    new_arr = []
    for i in arr:
        length = i.shape[0]
        difference = final_length - length
        left = difference // 2
        right = difference - left
        new_arr.append(np.pad(i, (left, right)))
    return new_arr

def index_ascend(arr):
    new = [0] * len(arr)
    for i in range(len(arr)):
        new[i] = sum(arr[:i+1])
    new.insert(0,0)
    return new

class RadChar_Dataset(Dataset):
    """RadioMapSeer Loader for accurate buildings and no measurements (RadioUNet_c)"""

    def __init__(self, phase="train",
                 dir_dataset="RadChar/RadChar-Tiny.h5",
                 SNR_range = list(range(-20,20 + 1)),
                 segmentation = False,
                 seed = 100,
                 sps = 3.2e6,
                 ):
        """
        Note: 
        1. Dataset has packets of SNR (discrete int dB) from -20 to 20
        2. If Dataset has been segmented, then 6 classes will be present instead of 5. An additional
           class called "noise" will be created.
        """
        
        # Set seed for randomisation of IQ samples
        random.seed(seed)
        
        self.sps = sps
        
        ## Obtain the index of the different signals - 5 classes
        with h5py.File(dir_dataset, "r") as f:
            h5_iqs = f['iq']
            h5_labels = f['labels']
            size = h5_labels.shape[0]
            ds_idx = [0,1,2,3,4]
            ds_idx = [int(size/5 * i) for i in ds_idx]
            
            self.input_x = h5_iqs[...]
            self.output_labels = h5_labels[...]
        
        self.segmentation = segmentation

        
        
#         # 8:2 split
#         if phase == "train":
#             for idx in range(5):
#                 # 0 to 80
#                 self.pre_segment_index += list(range(ds_idx[idx], ds_idx[idx] + int(8/50*size))) 
                
#         elif phase == "test":
#             for idx in range(5):                
#                 # 80 to 100
#                 self.pre_segment_index += list(range(ds_idx[idx] + int(8/50*size), ds_idx[idx] + int(10/50*size))) 
        
    
        # Filter the required Signals out
        ds_snr_filtered = [[] for i in range(5)]
        for idx, label in enumerate(self.output_labels):
            if label[6] in SNR_range:
                ds_snr_filtered[label[1]].append(idx)
        
        
        
        # Shuffling
        for packet_list in ds_snr_filtered:
            # print("Packet List Length: ", len(packet_list))
            random.shuffle(packet_list)
        
        # Pre Segmentation of Signals
        if not self.segmentation:
            self.pre_segment_index = []
            # 0 to 80
            if phase == "train":
                for packet_list in ds_snr_filtered:
                    self.pre_segment_index += packet_list[:int(0.8 * len(packet_list))]
            
            # 80 to 100
            elif phase == "test":
                for packet_list in ds_snr_filtered:
                    self.pre_segment_index += packet_list[int(0.8 * len(packet_list)):]
        
        # Post Segmentation of Signals:
        elif self.segmentation:
            ds_segmented = [[] for i in range(6)] # Last lst is for noise
            for packet_no, packet_list in enumerate(ds_snr_filtered):
                generated_pulses = self.generate_pulse(packet_list, self.input_x, self.output_labels)
                ds_segmented[packet_no] += generated_pulses[0] # Add the Pulses
                ds_segmented[-1] += generated_pulses[1] # Add the noise
            ds_segmented_len = [len(i) for i in ds_segmented]
            print(ds_segmented_len)
    
            # Segmented Y 
            self.segmented_Y = []
            for idx, i in enumerate(ds_segmented_len):
                self.segmented_Y += [idx] * i
            
            post_segment_index1 = list(range(len(self.segmented_Y)))
            post_segment_index2 = []
            
            
            # currently ds_segmented_len = [a,b,c,d,e]
            # ascending_ds_len = [0, a, a+b, a+b+c, a+b+c+d, a+b+c+d+e]
            ascending_ds_len = index_ascend(ds_segmented_len)
            
            
            # for idx, i in enumerate(ascending_ds_len):
            #     if idx + 1 >= len(ascending_ds_len):
            #         break
            #     post_segment_index2.append(post_segment_index1[i:ds_segmented_len[idx+1]])
            
            for i in range(len(ascending_ds_len)):
                if i >= len(ascending_ds_len)-1:
                    break
                post_segment_index2.append(post_segment_index1[ascending_ds_len[i]:ascending_ds_len[i+1]])
                
            # Segmented X
            self.segmented_X = pad(spill(ds_segmented))
            
            self.post_segment_index = []
            
            # 0 to 80
            if phase == "train":
                # print("Packet List Length: ", len(packet_list))
                for packet_list in post_segment_index2:
                    self.post_segment_index += packet_list[:int(0.8 * len(packet_list))]
            
            # 80 to 100
            elif phase == "test":
                for packet_list in post_segment_index2:
                    self.post_segment_index += packet_list[int(0.8 * len(packet_list)):]
            

    def __len__(self):
        if not self.segmentation:
            return len(self.pre_segment_index)
        else:
            return len(self.post_segment_index)

    def __getitem__(self, idx):
        
        if not self.segmentation:
            # Targets
            y = self.output_labels[self.pre_segment_index[idx]][1]

            # Inputs
            x = self.input_x[self.pre_segment_index[idx]]
            x_real = np.real(x)
            x_imag = np.imag(x)
            x = np.stack((x_real, x_imag))
            # x = x.reshape((1,x.shape[0]))
            # return (x.astype(np.float32), np.array(z))
            return (x.astype(np.float32), np.array(y))
        
        else:
            # Targets
            y = self.segmented_Y[self.post_segment_index[idx]]
            
            # Inputs
            x = self.segmented_X[self.post_segment_index[idx]]
            x_real = np.real(x)
            x_imag = np.imag(x)
            x = np.stack((x_real, x_imag))
            
            return (x.astype(np.float32), np.array(y))

    
    def generate_pulse(self, packet_list, iq_signals, iq_labels):
        pulses_lst = []
        noise_lst = []
        for idx in packet_list:
            _, signal_type, no_of_pulse,pulse_width, time_delay, PRI, SNR = iq_labels[idx]
            PRI_ = int(PRI*self.sps)
            time_delay_half = int(time_delay*self.sps*0.5)
            
            for i in range(no_of_pulse):
                l = int(i*(PRI_ * 0.82))
                r = int((i + 1)*(PRI_) * 0.82)
                pulses_lst.append(iq_signals[idx][l + time_delay_half : r + time_delay_half])
            noise_lst.append(iq_signals[idx][int(no_of_pulse*(PRI_ * 0.82)) + time_delay_half : int((no_of_pulse + 1)*(PRI_) * 0.82) + time_delay_half])
                
        return (pulses_lst, noise_lst) # Pulses, noise