from .common import set_device, red_print
from .c_model import get_model

# ***
def test_build(is_all=True, targets=[]):
    """
    abstract: test build models
    return:
    parameter:
            1. is_all:        if you want test all, enter True    : bool
            2. target:        target model                        : list (model id: str)
    """
    
    if (is_all == False) & (len(targets) == 0):
        print('[ERROR] Please enter the target')
        return
    
    # choose all model
    if is_all:
        targets = ['m01', 'm02', 'm03', 'm04', 'm05']
    
    # get test model configuration each model
    configs = []
    for target in targets:
        m_config = get_test_model_config(target)
        configs.append(m_config)
    
    # set gpu device
    set_device(0)
    
    # build each model and print result
    for target, config in zip(targets, configs):
        is_ok, err_code = test_build_model(target, config)
        if is_ok:
            print(f'[{target}] CHECK COMPLETE')
        else:
            red_print(f'[{target}] HAVE PROBLEMS [ERROR CODE: {err_code}]')
    
    
    
def test_build_model(m_id, m_config):
    """
    abstract: build and return the model
    return:
            1. is_ok:         test success or not                 : bool
            2. err_code:      error code                          : int
    
    parameter:
            1. m_id:          model id                            : string
            2. m_config:      model configuration                 : dict
    """
    
    try:
        m = get_model(m_id, m_config)
        return True, 0
    
    except Exception as e:
#         print(e)
        return False, 1874
      
    
            
# ***
def get_test_model_config(m_id):
    """
    abstract: build and return test model configuration
    return:
            1. m_config:      model configuration                 : dict
    
    parameter:
            1. m_id:          model id                            : string
    """
    
    s_len = 100
    n_dim = 1
    n_unit = 10
    drop_rate = 0.2
    n_output = 2

    m_config = {}
    if m_id == 'm01':
        m_config['s_len'] = s_len
        m_config['n_dim'] = n_dim
        m_config['n_unit'] = n_unit
        m_config['drop_rate'] = drop_rate
        m_config['n_output'] = n_output
        
    elif m_id == 'm02':
        m_config['s_len'] = s_len
        m_config['n_dim'] = n_dim
        m_config['n_output'] = n_output
        
    elif m_id == 'm03':
        m_config['s_len'] = s_len
        m_config['n_dim'] = n_dim
        m_config['n_unit1'] = n_unit
        m_config['drop_rate1'] = drop_rate
        m_config['n_unit2'] = n_unit
        m_config['drop_rate2'] = drop_rate
        m_config['n_output'] = n_output
        
    elif m_id == 'm04':
        m_config['s_len'] = s_len
        m_config['n_dim'] = n_dim
        m_config['n_unit'] = n_unit
        m_config['drop_rate'] = drop_rate
        m_config['n_output'] = n_output
        
    elif m_id == 'm05':
        m_config['s_len'] = s_len
        m_config['n_dim'] = n_dim
        m_config['n_unit'] = n_unit
        m_config['n_output'] = n_output
    
    return m_config


if __name__ == "__main__":
    test_build(is_all=True)
    test_build(is_all=False, targets=['m01', 'm04'])