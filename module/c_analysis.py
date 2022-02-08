import os
import json
import glob
import numpy as np
import pandas as pd

def get_best_result():
    """
    abstract: return best results from ray results about all experiment
    return:
            1. b_results:     best result each experiment         : pandas
    """
    
    r_path = './ray_results'              # ray path
    r_res_dir_list = os.listdir(r_path)   # ray result directory list
    r_res_dir_list.sort()

    e_results = [] # experiment results
    for folder_name in r_res_dir_list:
        try:
            e_state = glob.glob(f'{r_path}/{folder_name}/experiment_state-*.json')[0]
            e_state = open(e_state, 'r')
            e_state = json.load(e_state)
            e_result = e_state['checkpoints']
            e_result = [json.loads(res) for res in e_result]

            e_results.append(e_result)
        except Exception as e:
            continue

    b_results = [] # best results
    for res in e_results:
        try:
            r2 = [r['_last_result']['r2'] for r in res]
            val_r2 = [r['_last_result']['val_r2'] for r in res]
            mean_r2 = np.mean([r2, val_r2], axis=0)

            best = np.argmax(mean_r2)
            b_res = res[best]['_last_result']

            config = b_res.pop('config')
            e_config = config['exp']
            m_config = str(config['model'])

            b_res.update(e_config)
            b_res['m_config'] = m_config

            b_results.append(b_res)
            
        except Exception as e:
            continue


    columns = ['d_id', 's_id', 'm_id', 'e_id', 'case', 'n_fold', 
               'mse', 'mae', 'r2', 'val_mse', 'val_mae', 'val_r2', 
               'm_config']

    b_results = pd.DataFrame(b_results)
    b_results = b_results.loc[:, columns]
    
    return b_results



def get_case_result(results, d_id, s_id, m_id, e_id, case):
    """
    abstract: return case results
    return:
            1. c_result:      best result each experiment         : pandas
    
    parameter:
            1. results:       best result each experiment         : pandas
            2. d_id:          dataset id                          : string
            3. s_id:          detail id                           : string
            4. m_id:          model id                            : string
            5. e_id:          experiment id                       : string
            6. case:          subject id                          : string
    """
    
    cond = (results.d_id == d_id)
    cond &= (results.s_id == s_id)
    cond &= (results.m_id == m_id)
    cond &= (results.e_id == e_id)
    cond &= (results.case == case)
    
    c_result = results[cond] # case result
    
    return c_result



def get_case_list(results, d_id, s_id, m_id, e_id):
    """
    abstract: return case list
    return:
            1. c_list:        case list                           : list
            
    parameter:
            1. results:       best result each experiment         : pandas
            2. d_id:          dataset id                          : string
            3. s_id:          detail id                           : string
            4. m_id:          model id                            : string
            5. e_id:          experiment id                       : string
    """
    
    cond = (results.d_id == d_id)
    cond &= (results.s_id == s_id)
    cond &= (results.m_id == m_id)
    cond &= (results.e_id == e_id)
    
    c_list = results[cond].case.unique() # case list
    
    return c_list


import pickle
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def get_overall_metrics(results, d_id, s_id, m_id, e_id):
    """
    abstract: return overall metric results
    return:
            1. o_results:     overall metric results              : pandas
            
    parameter:
            1. results:       best result each experiment         : pandas
            2. d_id:          dataset id                          : string
            3. s_id:          detail id                           : string
            4. m_id:          model id                            : string
            5. e_id:          experiment id                       : string
    """
    
    metrics = ['mse', 'mae', 'r2', 'val_mse', 'val_mae', 'val_r2']
    c_list = get_case_list(results, d_id, s_id, m_id, e_id)              # case list
    
    o_results = [] # overall results
    for case in c_list:
        c_res = get_case_result(results, d_id, s_id, m_id, e_id, case)   # case result
        c_res_mean = c_res.loc[:, metrics].mean()                        # case metrics mean
        o_results.append(c_res_mean)
        
    o_results = pd.DataFrame(o_results)
    o_results.insert(0, 'case', c_list)

    # add test result
    handle = open(f'test_result/{d_id}_{s_id}_{m_id}_{e_id}.pickle', 'rb')
    pred_tsty = pickle.load(handle)
    o_results['tst_mse'] = [mean_squared_error(tsty, pred) for pred, tsty in pred_tsty]
    o_results['tst_mae'] = [mean_absolute_error(tsty, pred) for pred, tsty in pred_tsty]
    o_results['tst_r2'] = [r2_score(tsty, pred) for pred, tsty in pred_tsty]

    
    o_results.loc[o_results['r2'] < 0, 'r2'] = 0
    o_results.loc[o_results['val_r2'] < 0, 'val_r2'] = 0
    o_results.loc[o_results['tst_r2'] < 0, 'tst_r2'] = 0
    
    return o_results



def get_overall_config(results, d_id, s_id, m_id, e_id):
    """
    abstract: return overall model configuration
    return:
            1. o_config:      overall model configuration         : pandas
            
    parameter:
            1. results:       best result each experiment         : pandas
            2. d_id:          dataset id                          : string
            3. s_id:          detail id                           : string
            4. m_id:          model id                            : string
            5. e_id:          experiment id                       : string
    """
    
    c_list = get_case_list(results, d_id, s_id, m_id, e_id)              # case list

    o_config = [] # overall model configuration
    for case in c_list:
        c_res = get_case_result(results, d_id, s_id, m_id, e_id, case)   # case result
        o_config.append(c_res.loc[:, ['case', 'n_fold', 'm_config']])

    o_config = pd.concat(o_config)

    return o_config



from .c_data_loader import data_load
from .c_model import get_model
from .common import set_device, make_tag

def get_pred_tsty(d_id, s_id, m_id, e_id, case, n_fold, m_config, d_number):
    """
    abstract: get model prediction and test y values
    return:
            1. pred:          model prediction                    : list
            2. tsty:          true y values                       : list
            
    parameter:
            1. d_id:          dataset id                          : string
            2. s_id:          detail id                           : string
            3. m_id:          model id                            : string
            4. e_id:          experiment id                       : string
            5. case:          subject id                          : string
            6. n_fold:        # of fold                           : int
            7. m_config:      model configuration                 : dict
            8. d_number:      gpu device number                   : int
    """
        
    set_device(d_number)
    
    t_path = '.'
    m_c_path = './model_checkpoint'  # model checkpoint path
    
    # prepare test data
    _, _, tstx, tsty = data_load(t_path, d_id, s_id, e_id, case, n_fold, m_config['s_len'])
    tstx = tstx.reshape(-1, m_config['s_len'], m_config['n_dim'])
    
    # model build
    model = get_model(m_id, m_config)
    
    # weight load
    d_name = f'{d_id}_{s_id}_{m_id}_{e_id}_{case}_{n_fold}' # directory name
    m_c_file = make_tag(m_config) + '.h5'                   # model checkpoint file name
    
    model.load_weights(f'{m_c_path}/{d_name}/{m_c_file}')
    pred = model.predict(tstx)
    
    return pred, tsty



from tqdm import tqdm

def get_overall_pred_tsty(results, d_id, s_id, m_id, e_id, d_number):
    """
    abstract: get overall experimence of model prediction and test y values
    return:
            1. pred_tsty:     model prediction and true y values  : list
            
    parameter:
            1. results:       best result each experiment         : pandas
            2. d_id:          dataset id                          : string
            3. s_id:          detail id                           : string
            4. m_id:          model id                            : string
            5. e_id:          experiment id                       : string
            6. d_number:      gpu device number                   : int
    """
    
    o_config = get_overall_config(results, d_id, s_id, m_id, e_id) # overall model configuration

    pred_tsty = []
    for case, n_fold, m_config in tqdm(o_config.values):
        m_config = eval(m_config)

        pred, tsty = get_pred_tsty(d_id, s_id, m_id, e_id, case, n_fold, m_config, d_number)
        pred_tsty.append([pred, tsty])
        
    # merge fold results each case
    o_config = o_config.reset_index()
    c_group = o_config.groupby('case').groups

    temp = []
    for case in c_group:
        c_idx = c_group[case]
        pred = []
        tsty = []
        for i in c_idx:
            pred.append(pred_tsty[i][0])
            tsty.append(pred_tsty[i][1])

        pred = np.concatenate(pred)
        tsty = np.concatenate(tsty)

        temp.append([pred, tsty])

    pred_tsty = temp
    
    return pred_tsty



import matplotlib.pyplot as plt

def show_overall_metrics(results, d_id, s_id, m_id, e_id):
    """
    abstract: show the plot about overall metric results (MAE, R-squared)
    return:
    parameter:
            1. results:       best result each experiment         : pandas
            2. d_id:          dataset id                          : string
            3. s_id:          detail id                           : string
            4. m_id:          model id                            : string
            5. e_id:          experiment id                       : string
    """
    
    o_result = get_overall_metrics(results, d_id, s_id, m_id, e_id) # overall result
    
    metrics = [['mae', 'val_mae', 'tst_mae'], ['r2', 'val_r2', 'tst_r2']]

    fig, axes = plt.subplots(1, 2, figsize=(30, 5))
    idx = np.arange(len(o_result))*4
    for i, [t, v, k] in enumerate(metrics):
        axes[i].bar(idx-1, o_result[t], width=1, label=t)
        axes[i].bar(idx, o_result[v], width=1, label=v)
        axes[i].bar(idx+1, o_result[k], width=1, label=k)
        axes[i].set_xticks(idx)
        axes[i].set_xticklabels(o_result.case)
        axes[i].legend()

    plt.suptitle(f'{d_id} {s_id} {m_id} {e_id}', y=0.95)
    plt.show()
    

    
def show_pred_true(pred, true, target, p_title, mode=0):
    """
    abstract: show the plot about prediction and true y
    return:
    parameter:
            1. pred:          model prediction                    : list
            2. true:          true y                              : list
            3. target:        systolic bp or diastolic bp         : string
            4. p_title:       plot title                          : string
            5. mode:          plot mode                           : int
    """
    
    if target == 'sys':
        t_number = 0 # target index
    elif target == 'dia':
        t_number = 1 # target index
    else:
        return
    
    if mode == 0:
        plt.figure(figsize=(10, 5))
        plt.plot(pred[:, t_number], label='pred')
        plt.plot(true[:, t_number], label='true', alpha=0.7)
        plt.title(f'{p_title} ({target})')
        plt.legend()
        plt.show()
        
    elif mode == 1:
        plt.figure(figsize=(5, 5))
        plt.scatter(true[:, t_number], pred[:, t_number])
        plt.title(f'{p_title} ({target})')
        plt.xlabel('true')
        plt.ylabel('pred')
        plt.show()
        
    else:
        return