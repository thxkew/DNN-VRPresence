import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import torchvision.models as models
from resnet import ResNet1D
from layer import AttentionLSTM, SELayer, BahdanauAttention

class FCN(nn.Module):
    def __init__(self, input_size=81, momentum=0.99, eps=0.001):
        super(FCN, self).__init__()
        
        self.feature = nn.Sequential(  

                                        nn.Conv1d(input_size, 128, kernel_size=8),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        nn.Conv1d(128, 128, kernel_size=8),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        nn.Conv1d(128, 128, kernel_size=8),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        
                                        nn.Conv1d(128, 256, kernel_size=5),
                                        nn.BatchNorm1d(256, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        nn.Conv1d(256, 256, kernel_size=5),
                                        nn.BatchNorm1d(256, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        nn.Conv1d(256, 256, kernel_size=5),
                                        nn.BatchNorm1d(256, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        
                            
                                        nn.Conv1d(256, 128, kernel_size=3),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        nn.Conv1d(128, 128, kernel_size=3),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        nn.Conv1d(128, 128, kernel_size=3),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        
                                     )
        
    
    def forward(self, x):
        
        x = self.feature(x)
        x = torch.mean(x, 2)

        return x
    
class LSTM_FCN(nn.Module):
    def __init__(self, input_size=75, input_lstm=240, momentum=0.99, eps=0.001):
        super(LSTM_FCN, self).__init__()
        
        self.lstm = nn.LSTM(input_lstm, 100)
        self.lstm_dropout = nn.Dropout(p=0.8)
        
        self.feature = nn.Sequential(   

                                        nn.Conv1d(input_size, 128, kernel_size=8),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        
                                        nn.Conv1d(128, 256, kernel_size=5),
                                        nn.BatchNorm1d(256, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                            
                                        nn.Conv1d(256, 128, kernel_size=3),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        
                                     )
    
    def forward(self, x):

        x_fcn = self.feature(x)
        x_fcn = torch.mean(x_fcn, 2)
        
        x_lstm = x.clone().permute(1, 0, 2)
        _, (h, c) = self.lstm(x_lstm)
        x_lstm = self.lstm_dropout(h)
        x_lstm = torch.squeeze(x_lstm, dim=0)

        x = torch.cat((x_lstm, x_fcn),dim=1)

        return x
   
class ALSTM_FCN(nn.Module):
    def __init__(self, input_size=75, momentum=0.99, eps=0.001):
        super(ALSTM_FCN, self).__init__()
        
        self.lstm = nn.LSTM(240, 128)
        self.lstm_dropout = nn.Dropout(p=0.8)

        self.attention = BahdanauAttention(128)
        
        self.feature = nn.Sequential(   

                                        nn.Conv1d(input_size, 128, kernel_size=8),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        
                                        nn.Conv1d(128, 256, kernel_size=5),
                                        nn.BatchNorm1d(256, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                            
                                        nn.Conv1d(256, 128, kernel_size=3),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        
                                     )
        
        
        self.fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        x_fcn = self.feature(x)
        x_fcn = torch.mean(x_fcn, 2)
        
        x_lstm = x.clone().permute(1, 0, 2)
        outputs, (h, c) = self.lstm(x_lstm)
        context, attention_weights = self.attention(h[-1], outputs)
        x_lstm = self.lstm_dropout(context)

        x = torch.cat((x_lstm, x_fcn),dim=1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

class FCN_SE(nn.Module):
    def __init__(self, input_size=72, momentum=0.99, eps=0.001):
        super(FCN_SE, self).__init__()
        
        self.feature = nn.Sequential(  

                                        nn.Conv1d(input_size, 128, kernel_size=8),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        nn.Conv1d(128, 128, kernel_size=8),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        nn.Conv1d(128, 128, kernel_size=8),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        SELayer(128),
                                        
                                        nn.Conv1d(128, 256, kernel_size=5),
                                        nn.BatchNorm1d(256, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        nn.Conv1d(256, 256, kernel_size=5),
                                        nn.BatchNorm1d(256, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        nn.Conv1d(256, 256, kernel_size=5),
                                        nn.BatchNorm1d(256, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        SELayer(256),
                            
                                        nn.Conv1d(256, 128, kernel_size=3),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        nn.Conv1d(128, 128, kernel_size=3),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        nn.Conv1d(128, 128, kernel_size=3),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),

                                     )
        
    
    def forward(self, x):
        
        x = self.feature(x)
        x = torch.mean(x, 2)

        return x

class MLSTM_FCN(nn.Module):
    def __init__(self, input_size=75, input_lstm=240, momentum=0.99, eps=0.001):
        super(MLSTM_FCN, self).__init__()
        
        self.lstm = nn.LSTM(input_lstm, 100)
        self.lstm_dropout = nn.Dropout(p=0.8)
        
        self.feature = nn.Sequential(  

                                        nn.Conv1d(input_size, 128, kernel_size=8),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        SELayer(128),
                                        
                                        nn.Conv1d(128, 256, kernel_size=5),
                                        nn.BatchNorm1d(256, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        SELayer(256),
                            
                                        nn.Conv1d(256, 128, kernel_size=3),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        
                                     )
    
    def forward(self, x):
        
        x_fcn = self.feature(x)
        x_fcn = torch.mean(x_fcn, 2)
        
        x_lstm = x.clone().permute(1, 0, 2)
        _, (h, c) = self.lstm(x_lstm)
        x_lstm = self.lstm_dropout(h)
        x_lstm = torch.squeeze(x_lstm, dim=0)

        x = torch.cat((x_lstm, x_fcn),dim=1)

        return x

class MALSTM_FCN(nn.Module):
    def __init__(self, input_size=75, momentum=0.99, eps=0.001):
        super(MALSTM_FCN, self).__init__()
        
        self.lstm = nn.LSTM(240, 128)
        self.lstm_dropout = nn.Dropout(p=0.8)

        self.attention = BahdanauAttention(128)
        
        self.feature = nn.Sequential(  

                                        nn.Conv1d(input_size, 128, kernel_size=8),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        SELayer(128),
                                        
                                        nn.Conv1d(128, 256, kernel_size=5),
                                        nn.BatchNorm1d(256, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        SELayer(256),
                            
                                        nn.Conv1d(256, 128, kernel_size=3),
                                        nn.BatchNorm1d(128, momentum=momentum, eps=eps),
                                        nn.ReLU(),
                                        
                                     )
        
        
        self.fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):

        x_fcn = self.feature(x)
        x_fcn = torch.mean(x_fcn, 2)
        
        x_lstm = x.clone().permute(1, 0, 2)
        outputs, (h, c) = self.lstm(x_lstm)
        context, attention_weights = self.attention(h[-1], outputs)
        x_lstm = self.lstm_dropout(context)

        x = torch.cat((x_lstm, x_fcn),dim=1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x

class Model(nn.Module):
    def __init__(self, input_size=72, momentum=0.99, eps=0.001):
        super(Model, self).__init__()
        
        self.feature_hand = ResNet1D(input_size=12)
        self.feature_face = ResNet1D(input_size=63)
        self.feature_head = ResNet1D(input_size=3)
        
        self.epp_fc = nn.Linear(6, 128)
        self.vep_fc = nn.Linear(6, 128)
        
        self.fc = nn.Linear(1024, 1)
    
    def forward(self, x, epp, vep):

        hand = x[:, 0:12, :]
        face = x[:, 12:75, :]
        head = x[:, 75:, :]

        x_hand = self.feature_hand(hand)
        x_face = self.feature_face(face)
        x_head = self.feature_head(head)

        x_epp = self.epp_fc(epp)
        x_vep = self.vep_fc(vep)

        x_cat = torch.cat((x_hand, x_face, x_head, x_epp, x_vep), dim=1)
        pred_y = self.fc(x_cat)

        return pred_y