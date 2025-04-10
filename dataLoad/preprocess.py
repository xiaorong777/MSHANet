import os 
import sys
current_path = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(os.path.split(current_path)[0])[0]
sys.path.append(current_path)
sys.path.append(rootPath)

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from LoadData import load_data_2a, Load_BCIC_2b
from LoadData import load_data_LOSO
from LoadData import load_data_onLine2a



def calculate_distances(channel_positions):
    num_channels = len(channel_positions)
    positions = np.array([channel_positions[i+1] for i in range(num_channels)])  # 将位置转换为数组
    distances = np.zeros((num_channels, num_channels))  # 初始化距离矩阵
    
    for i in range(num_channels):
        for j in range(num_channels):
            distances[i, j] = np.linalg.norm(positions[i] - positions[j])  # 计算欧几里得距离
    
    return distances


 # 生成时间位置编码
def get_time_positional_encoding(seq_len, d_model):
     pe = np.zeros((seq_len, d_model))
     position = np.arange(0, seq_len).reshape(-1, 1)
     div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
     pe[:, 0::2] = np.sin(position * div_term)
     pe[:, 1::2] = np.cos(position * div_term)
     return pe


def create_adjacency_matrix(threshold):

    channel_positions = {
    1: (0, 7), 2: (-7, 3.5), 3: (-3.5, 3.5), 4: (0, 3.5), 5: (3.5, 3.5), 6: (7, 3.5),
    7: (-10.5,0), 8: (-7, 0), 9: (-3.5, 0), 10: (0, 0), 11: (3.5, 0), 12: (7, 0), 13: (10.5, 0),
    14: (-7, -3.5), 15: (-3.5,-3.5), 16: (0, -3.5), 17: (3.5, -3.5), 18: (7, -3.5), 19: (-3.5, -7), 20: (0, -7),
    21: (3.5, -7), 22: (0, -10.5)
    }

    coords = np.array([channel_positions[i+1] for i in range(22)])  # 获取位置数组
    n = coords.shape[0]
    adj_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dist = np.sqrt((coords[i, 0] - coords[j, 0]) ** 2 + (coords[i, 1] - coords[j, 1]) ** 2)
                if dist < threshold:
                    adj_matrix[i, j] = 1
    # 添加自连接
    np.fill_diagonal(adj_matrix, 1)
    degree_matrix = np.sum(adj_matrix, axis=1)
    norm_matrix = adj_matrix / degree_matrix[:, np.newaxis]
    return norm_matrix




#%%
def standardize_data(X_train, X_test, channels):
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, j, :])
          X_train[:, j, :] = scaler.transform(X_train[:, j, :])
          X_test[:, j, :] = scaler.transform(X_test[:, j, :])

    return X_train, X_test

#%%
def standardize_data_trans(X_train, X_test, X_train_trans, channels): 
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, j, :])
          X_train[:, j, :] = scaler.transform(X_train[:, j, :])
          X_test[:, j, :] = scaler.transform(X_test[:, j, :])
          X_train_trans[:, j, :] = scaler.transform(X_train_trans[:, j, :])

    return X_train, X_test, X_train_trans

#%%
def standardize_data_onLine2a(X_train, channels):
    # X_train & X_test :[Trials, MI-tasks, Channels, Time points]
    for j in range(channels):
          scaler = StandardScaler()
          scaler.fit(X_train[:, j, :])
          X_train[:, j, :] = scaler.transform(X_train[:, j, :])

    return X_train


#%%
def get_data(path, subject=None, LOSO=False, Transfer=False, trans_num=1, onLine_2a=False,  data_model='one_session', isStandard=True,PE=False, data_type='2a'):
    # Define dataset parameters
    fs = 250          # sampling rate
    t1 = int(2*fs)    # start time_point
    t2 = int(6*fs)    # end time_point
    T = t2-t1         # length of the MI trial (samples or time_points)
 
    # Load and split the dataset into training and testing 
    if LOSO:
        # Loading and Dividing of the data set based on the 
        # 'Leave One Subject Out' (LOSO) evaluation approach. 
        X_train, y_train, X_test, y_test, X_train_trans, y_train_trans = load_data_LOSO(path, subject, data_model, Transfer, trans_num)
    elif onLine_2a:
        X_train, y_train = load_data_onLine2a(path, data_model)
        X_test = []
        y_test = []
    else:
        # Loading and Dividing of the data set based on the subject-specific (subject-dependent) approach.
        # In this approach, we used the same training and testing data as the original competition, 
        # i.e., trials in session 1 for training, and trials in session 2 for testing.  
        path = path + 's{:}/'.format(subject)
        if data_type == '2a':
            X_train, y_train = load_data_2a(path, subject, True)
            X_test, y_test = load_data_2a(path, subject, False)
        elif data_type == '2b':
            load_raw_data = Load_BCIC_2b(path, subject)
            eeg_data = load_raw_data.get_epochs_train(tmin=0., tmax=4.)
            X_train, y_train = eeg_data['x_data'], eeg_data['y_labels']
            eeg_data = load_raw_data.get_epochs_test(tmin=0., tmax=4.)
            X_test, y_test = eeg_data['x_data'], eeg_data['y_labels']

    # Prepare training data
    N_tr, N_ch, samples = X_train.shape 
    if  data_type == '2a':
        X_train = X_train[:, :, t1:t2]
        y_train = y_train-1

    # Prepare testing data 
    if onLine_2a == False:
        if  data_type == '2a':
            X_test = X_test[:, :, t1:t2]
            y_test = y_test-1

    if Transfer:
        X_train_trans = X_train_trans[:, :, t1:t2]
        y_train_trans = y_train_trans-1
    else:
        X_train_trans = []
        y_train_trans = []

    # Standardize the data
    if (isStandard == True):
        if Transfer:
            X_train, X_test, X_train_trans = standardize_data_trans(X_train, X_test, X_train_trans, N_ch)
        elif onLine_2a:
            X_train = standardize_data_onLine2a(X_train, N_ch)
        else:
            X_train, X_test = standardize_data(X_train, X_test, N_ch)


            
    if (PE == True):
        d_model = 22
        channel_pe = get_channel_positional_encoding(22,d_model,alpha=0.7)
        #time_pe = get_time_positional_encoding(1000,d_model)
        for s in range(288):  # 遍历每个样本
            for i in range(22):  # 遍历每个通道
                for t in range(1000):  # 遍历每个时间点
                    X_train[s, i, t] = X_train[s, i, t] + channel_pe[i, t % d_model] #+ time_pe[t,i % d_model]
                    X_test[s, i, t] = X_test[s, i, t] + channel_pe[i, t % d_model] #+ time_pe[t,i % d_model] 

        # threshold = 5
        # norm_adj_matrix = create_adjacency_matrix(threshold)
        # # 归一化邻接矩阵


        # X_train = np.einsum('ij,bjk->bik',norm_adj_matrix,X_train)
        # X_test = np.einsum('ij,bjk->bik',norm_adj_matrix,X_test)



    return X_train, y_train, X_test, y_test, X_train_trans, y_train_trans

#%%
def cross_validate(x_data, y_label, kfold, data_seed=20250228):
    '''
    This version dosen't use early stoping.
    Arg:
        sub:Subject number.
        data_path:The data path of all subjects.
        augment(bool):Data augment.Take care that this operation will change the size in the temporal dimension.
        validation_rate:The percentage of validation data in the data to be divided. 
        data_seed:The random seed for shuffle the data.
    @author:Guangjin Liang
    '''

    skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=data_seed)
    for split_train_index,split_validation_index in skf.split(x_data,y_label):
        split_train_x       = x_data[split_train_index]
        split_train_y       = y_label[split_train_index]
        split_validation_x  = x_data[split_validation_index]
        split_validation_y  = y_label[split_validation_index]

        split_train_x,split_train_y = torch.FloatTensor(split_train_x),torch.LongTensor(split_train_y).reshape(-1)
        split_validation_x,split_validation_y = torch.FloatTensor(split_validation_x),torch.LongTensor(split_validation_y).reshape(-1)
   
        split_train_dataset = TensorDataset(split_train_x,split_train_y)
        split_validation_dataset = TensorDataset(split_validation_x,split_validation_y)
    
        yield split_train_dataset,split_validation_dataset


#%%
def BCIC_DataLoader(x_train, y_train, batch_size=64, num_workers=1, shuffle=True):
    '''
    Cenerate the batch data.

    Args:
        x_train: data to be trained
        y_train: label to be trained
        batch_size: the size of the one batch
        num_workers: how many subprocesses to use for data loading
        shuffle: shuffle the data
    '''
    # 将数据转换为TensorDataset类型
    dataset  = TensorDataset(x_train, y_train)
    # 分割数据，生成batch
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # 函数返回值
    return dataloader

# %%
