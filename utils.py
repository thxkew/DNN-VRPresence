import numpy as np

def loaddata (data_dir):
    '''

    Input : 
    [1] data_dir : directory path that contain data.

    Output :
    [1] behavioral_cues :   A numpy array of behavioral cues, having dimension [N, L, C].
                            Where N is number of sample, L is sequence length (240), and C is number of data feature channel (78).
                            
                            index   0 to 11     ->  hand movements
                                    12 to 74    ->  facial expressions
                                    75 to 77    ->  head movements

    [2] EPP             :   A list of Experiential Presence Profile, having legth equal to number of sample.

    [3] VEP             :   A list of Visual Entropy Profile, having legth equal to number of sample.

    [4] label           : A list of ground-truth label (IPQ score), having legth equal to number of sample.

    '''

    behavioral_cues = np.random.rand(197,240, 78)
    EPP = list(np.random.rand(240))
    VEP = list(np.random.rand(240))
    label = list(np.random.rand(240))

    return behavioral_cues, EPP, VEP, label