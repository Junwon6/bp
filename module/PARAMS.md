- s_len:         sequence length (time step)         : int
- values:        learning dataset                    : 1d-numpy
- l_dataset:     learning dataset                    : numpy
                                                            - unit bvp: unit bvp                            : float
                                                            - unit bvp diff 1: bvp 1-differential values    : float
                                                            - unit bvp diff 2: bvp 2-differential values    : float
                                                            - features:   bvp unit features                 : object
                                                            - sys: systolic blood pressure                  : float
                                                            - dia: diastolic blood pressure                 : float
                                                            
- f_name:        list of feature name                : list (string)

- v_bndry:       validation boundary                 : pandas
                                                            - start_time: bvp boundaries' start time index  : int 
                                                            - end_time:   bvp boundaries' end time index    : int
                                                            - features:   bvp unit features                 : object
                                                            
- bvp_bp:        bvp and bp values                   : pandas
                                                            - sys: systolic blood pressure                  : float
                                                            - dia: diastolic blood pressure                 : float
                                                            - bvp: blood volumn pulse                       : float
                                                            
- case:          subject id                          : int
- case:          subject id                          : string

- n_fold:        # of fold                           : int
- n_fold:        fold number                         : int

- idx:           fold index                          : dict
- t_datset:      total datsets                       : list
- t_idx:         total fold index                    : list



- trnx:          train X                             : numpy
- trny:          train y                             : numpy
- tstx:          test X                              : numpy
- tsty:          test y                              : numpy
- t_path:        total absolute path                 : string

- d_id:          dataset id                          : string
- s_id:          detail id                           : string
- e_id:          experiment id                       : string

- X:             X data reshaped                     : pandas
- X:             X data                              : pandas



- y_true:        ground truth                        : float
- y_pred:        model prediction                    : float



- n_dim:         data dimension                      : int
- n_unit:        # of lstm unit                      : int
- drop_rate:     drop out rate                       : float
- n_output:      # of output                         : int
- n_unit1:       # of first layer lstm unit          : int
- n_unit2:       # of second layer lstm unit         : int
- drop_rate1:    drop out rate of first layer lstm   : float
- drop_rate2:    drop out rate of second layer lstm  : float
- m_id:          model id                            : string
- m_config:      model configuration                 : dict



- is_all:        if you want test all, enter True    : bool
- target:        target model                        : list (model id: str)
- is_ok:         test success or not                 : bool
- err_code:      error code                          : int



- d_number:      gpu device number                   : int
- _dict:         dictionary                          : dict
- func:          model building function             : function


- results:       best result each experiment         : pandas
- o_results:     overall metric results              : pandas
- o_config:      overall model configuration         : pandas
- pred:          model prediction                    : list
- tsty:          true y values                       : list
- true:          true y                              : list
- target:        systolic bp or diastolic bp         : string
- p_title:       plot title                          : string
- mode:          plot mode                           : int