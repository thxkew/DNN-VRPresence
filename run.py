import torch
import numpy as np
from dataset import MultiBehaviorDataset
from torch.utils import data
import torch.nn as nn
from model import Model
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from scipy.stats import pearsonr, spearmanr
import utils
import json
import gc

random_seed = 0
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

def test(testloader, net, criterion, device):
    
    net.eval()
    
    y_pred = []
    y_true = []
    envi = []
    running_loss = []
    with torch.no_grad():  # Disable gradient computation during testing
        for i_batch, sample in enumerate(testloader):
            
            X, EPP, VEP, y = sample
            
            X = X.to(device, dtype=torch.float32) #[B, L, C]
            X = X.permute(0, 2, 1)
            EPP = EPP.to(device, dtype=torch.float32)
            VEP = VEP.to(device, dtype=torch.float32)
            
            y = y.to(device, dtype=torch.float32) #[B]

            outputs = net(X, EPP, VEP)
                
            loss = criterion(outputs, y)
            running_loss.append(loss.item())
            
            y_true.extend(y.reshape(-1,).cpu().detach().to(dtype=torch.float32).tolist())
            y_pred.extend(outputs.reshape(-1,).cpu().detach().to(dtype=torch.float32).tolist())
            
    test_loss = np.mean(running_loss)

    return y_true, y_pred, test_loss

def train(trainloader, net, criterion, optimizer, device):
    
    net.train()
    
    y_pred = []
    y_true = []
    envi = []
    running_loss = []
    for i_batch, sample in enumerate(trainloader):

        optimizer.zero_grad()
        
        X, EPP, VEP, y = sample
        
        X = X.to(device, dtype=torch.float32) #[B, L, F]
        X = X.permute(0, 2, 1) #[B, F, L]
        EPP = EPP.to(device, dtype=torch.float32)
        VEP = VEP.to(device, dtype=torch.float32)
        
        y = y.to(device, dtype=torch.float32) #[B]

        outputs = net(X, EPP, VEP)
            
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.item())
       
        y_true.extend(y.reshape(-1,).cpu().detach().to(dtype=torch.float32).tolist())
        y_pred.extend(outputs.reshape(-1,).cpu().detach().to(dtype=torch.float32).tolist())
        
    train_loss = np.mean(running_loss)
    
    return y_true, y_pred, train_loss

def main():
    
    data_dir='./data2'
    log_interval = 100
    input_size = 78
    num_epoch = 2000
    learning_rate = 0.001
    batch_size = 32
    k_folds = 5
    load_img = False
    load_existing_data = True
        
    device = torch.device('cuda')

    behavioral_cues, EPP, VEP, label = utils.loaddata(data_dir=data_dir)
    
    kfold = RepeatedKFold(n_splits=k_folds, n_repeats=5, random_state=random_seed)
    
    stacked_y_true = []
    stacked_y_pred = []
    stacked_envi = []

    for fold, (train_ids, test_ids) in enumerate(kfold.split(behavioral_cues, label)):
        gc.collect()
        torch.cuda.empty_cache() #clear memory before starting new fold. 

        X_train, y_train = behavioral_cues[train_ids], label[train_ids]       
        EPP_train = [EPP[idx] for idx in train_ids]
        VEP_train = [VEP[idx] for idx in train_ids]
        
        X_test,  y_test = behavioral_cues[test_ids], label[test_ids]
        EPP_test = [EPP[idx] for idx in test_ids]
        VEP_test = [VEP[idx] for idx in test_ids]
              
        trainset = MultiBehaviorDataset(X_train, EPP_train, VEP_train, y_train)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    
        testset = MultiBehaviorDataset(X_test, EPP_test, VEP_test, y_test)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)

        criterion = nn.MSELoss()

        #Reinitialize the model for each fold
        net = Model(input_size=input_size).to(device)
        optimizer =  torch.optim.Adam(net.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.79370052598, patience=100, threshold=0.0001, min_lr=0.0001)

        best_result = {'epoch':0, 'plcc':0, 'p-value':0, 'mae':0, 'r2':0, 'loss':10000000, 'y_true':0, 'y_pred':0}
        for epoch in range(num_epoch):  # loop over the dataset multiple times

            y_train_true, y_train_pred, train_loss = train(trainloader, net, criterion, optimizer, device)
            y_test_true, y_test_pred, test_loss = test(testloader, net, criterion, device)

            scheduler.step(test_loss)

            if test_loss <= best_result['loss']:

                plcc_test, _ = pearsonr(y_test_true, y_test_pred)
                srocc_test, _ = spearmanr(y_test_true, y_test_pred)
                mae_test = mean_absolute_error(y_test_true, y_test_pred)
                r2_test = r2_score(y_test_true, y_test_pred)

                best_result['epoch'] = epoch
                best_result['plcc'] = plcc_test
                best_result['srocc'] = srocc_test
                best_result['mae'] = mae_test
                best_result['r2'] = r2_test
                best_result['loss'] = test_loss
                best_result['y_true'] = y_test_true
                best_result['y_pred'] = y_test_pred

            if epoch%log_interval == 0:
                
                print('\n')
                print(" --- Fold {}, Epoch --- {}".format(fold, epoch))

                plcc_train, _ = pearsonr(y_train_true, y_train_pred)
                srocc_train, _ = spearmanr(y_train_true, y_train_pred)
                mae_train = mean_absolute_error(y_train_true, y_train_pred)
                r2_train = r2_score(y_train_true, y_train_pred)
                print('\n')
                print("train plcc : {}".format(plcc_train))
                print("train srocc : {}".format(srocc_train))
                print("train mae : {}".format(mae_train))
                print("train r2 : {}".format(r2_train))
                print("train loss : {}".format(train_loss))
                
                plcc_test, _ = pearsonr(y_test_true, y_test_pred)
                srocc_test, _ = spearmanr(y_test_true, y_test_pred)
                mae_test = mean_absolute_error(y_test_true, y_test_pred)
                r2_test = r2_score(y_test_true, y_test_pred)
                print('\n')
                print("test plcc : {}".format(plcc_test))
                print("test srocc : {}".format(srocc_test))
                print("test mae : {}".format(mae_test))
                print("test r2 : {}".format(r2_test))
                print("test loss : {}".format(test_loss))

        print('\n')
        print(" --- Fold --- {}".format(fold))

        print('\n')
        print("best epoch : {}".format(best_result['epoch']))
        print("test plcc : {}".format(best_result['plcc']))
        print("test srocc : {}".format(best_result['srocc']))
        print("test mae : {}".format(best_result['mae']))
        print("test r2 : {}".format(best_result['r2']))
        print("test loss : {}".format(best_result['loss']))

        stacked_y_true.extend(best_result['y_true'])
        stacked_y_pred.extend(best_result['y_pred'])
        stacked_envi.extend(best_result['envi'])
        
        net.to('cpu')
        del net, optimizer, scheduler, trainset, testset, trainloader, testloader, criterion, X_train, y_train, X_test, y_test

    plcc, _ = pearsonr(stacked_y_true, stacked_y_pred)
    srocc, _ = spearmanr(stacked_y_true, stacked_y_pred)
    mae = mean_absolute_error(stacked_y_true, stacked_y_pred)
    r2 = r2_score(stacked_y_true, stacked_y_pred)

    print("\n --- Overall ---")
    print("plcc : {}".format(plcc))
    print("srocc : {}".format(srocc))
    print("mae : {}".format(mae))
    print("r2 : {}".format(r2))

    #save result
    np.save('.\\stacked_y_true', np.stack(stacked_y_true, axis=0))
    np.save('.\\stacked_y_pred', np.stack(stacked_y_pred, axis=0))

if __name__ == '__main__':
    main()
