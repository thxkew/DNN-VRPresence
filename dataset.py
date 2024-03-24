import torch.utils.data as data
import numpy as np
import random
import os
from skimage import io, transform

class MultiBehaviorDataset(data.Dataset):
    def __init__(self, behavioral_cues, EPP, VEP, label):
        
        self.behavioral_cues = behavioral_cues
        self.EPP = EPP
        self.VEP = VEP
        self.label = label

        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        
        X = self.behavioral_cues[idx, :, :]
        y = self.label[idx]

        epp = self.EPP[idx]
        epp = np.array(epp)

        vep = self.VEP[idx]
        vep = np.array(vep)
        
        return X, epp, vep, y