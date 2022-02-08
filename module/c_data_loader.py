import pickle
import pandas as pd

def data_load(t_path, d_id, s_id, e_id, case, n_fold, s_len):
    """
    abstract: load the learning dataset with train & test index
    return:
            1. trnx:            train X                           : numpy
            2. trny:            train y                           : numpy
            3. tstx:            test X                            : numpy
            4. tsty:            test y                            : numpy
    
    parameter:
            1. t_path:          total absolute path               : string
            2. d_id:            dataset id                        : string
            3. s_id:            detail id                         : string
            4. e_id:            experiment id                     : string
            5. case:            subject id                        : string
            6. n_fold:          fold number                       : int
            7. s_len:           sequence length (time step)       : int
    """
    
    l_path = f'{t_path}/l-dataset/{d_id}/{s_id}' # load path

    data = pd.read_csv(f'{l_path}/{case}/{case}.csv')
    X = data.drop(['sys', 'dia'], axis=1)
    y = data.loc[:, ['sys', 'dia']]
    
    X = reshape_X(X, e_id, s_len)

    handle = open(f'{l_path}/{case}/{case}_idx.pickle', 'rb')
    p_load = pickle.load(handle)
    tr_idx = p_load[n_fold]['train']     # train index
    ts_idx = p_load[n_fold]['test']      # test index

    trnx = X.loc[tr_idx].values
    tstx = X.loc[ts_idx].values
    trny = y.loc[tr_idx].values
    tsty = y.loc[ts_idx].values
    
    return trnx, trny, tstx, tsty



def reshape_X(X, e_id, s_len):
    """
    abstract: reshape X each experiment id
    return:
            1. X:               X data reshaped                   : pandas
    
    parameter:
            1. X:               X data                            : pandas
            2. e_id:            experiment id                     : string
            3. s_len:           sequence length (time step)       : int
    """
    
    if (e_id == 'e01') | (e_id == 'e06'):
        return X.iloc[:, :s_len]
    
    elif (e_id == 'e02') | (e_id == 'e07'):
        return X.iloc[:, :s_len*3]
    
    elif (e_id == 'e03') | (e_id == 'e08'):
        return X.iloc[:, s_len*3:]
    
    elif (e_id == 'e04') | (e_id == 'e09'):
        features = X.iloc[:, s_len*3:]
        return pd.concat([X.iloc[:, :s_len], X.iloc[:, s_len*3:]], axis=1)
    
    elif (e_id == 'e05') | (e_id == 'e10'):
        return X