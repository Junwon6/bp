import os
import tensorflow as tf
import tensorflow.keras as keras

def set_device(d_number):
    """
    abstract: set gpu device
    return:
    parameter:
            1. d_number:      gpu device number                   : int
    """
    
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{d_number}'
    gpu = tf.config.experimental.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.backend.set_floatx('float64')

def red_print(words):
    # abstract: print red color text
    #
    # parameter:
    #         1. words:         words for print                    : string
    
    print(f'\033[31m{words}\033[0m')



def make_tag(_dict):
    """
    abstract: make and return tag
    return: tag                                                   : string
    
    parameter:
            1. _dict:         dictionary                          : dict
    """
    
    tag = _dict.items()
    tag = [str(t[1]) for t in tag]
    tag = '_'.join(tag)
    
    return tag
    

    
from .c_model import l01, c01, l02, l03, l04

# ***
def get_model_func(m_id):
    """
    abstract: get model building function matched model id
    return: model building function                               : function
    
    parameter:
            1. m_id:          model id                            : string
    """
    
    if m_id == 'm01':
        return l01
    elif m_id == 'm02':
        return c01
    elif m_id == 'm03':
        return l02
    elif m_id == 'm04':
        return l03
    elif m_id == 'm05':
        return l04
    else:
        print('[ERROR] Not exist model id')
        return None
    
    
    
def get_param_name(func):
    """
    abstract: get model building function matched model id
    return: function parameter names                               : list
    
    parameter:
            1. func:          model building function              : function
    """
    
    f_varnames = func.__code__.co_varnames    # function variable names
    f_param_cnt = func.__code__.co_argcount   # function parameter count
    p_names = f_varnames[:f_param_cnt]             # parameter names
    
    return p_names