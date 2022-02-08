import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models, optimizers



#-----------------------------[MODEL INFO]-------------------------#
#         m01-l01: 1-layer lstm                                    #
#         m02-c01: multi-layer cnn                                 #
#         m03-l02: 2-layer lstm                                    #
#         m04-l03: 1-layer bidirectional lstm                      #
#         m05-l04: stacked, residual, bidirectional lstm           #
#------------------------------------------------------------------#



def l01(s_len, n_dim, 
        n_unit, drop_rate, 
        n_output):
    """
    m01
    abstract: 1-layer lstm
    return: model                                                 : object
    
    parameter:
            1. s_len:         sequence length (time step)         : int
            2. n_dim:         data dimension                      : int
            3. n_unit:        # of lstm unit                      : int
            4. drop_rate:     drop out rate                       : float
            5. n_output:      # of output                         : int
    """
    
    i_tensor = layers.Input(shape=(s_len, n_dim))   # input tensor

    lstm = layers.LSTM(n_unit)(i_tensor)            # lstm
    drop = layers.Dropout(drop_rate)(lstm)          # drop out
    actv = layers.Activation('relu')(drop)          # activation (relu)
    btch = layers.BatchNormalization()(actv)        # batch norm

    o_tensor = layers.Dense(n_output)(btch)   
    
    model = models.Model(i_tensor, o_tensor)
    
    return model



def c01(s_len, n_dim, n_output):
    """
    m02
    abstract: multi-layer cnn
    return: model                                                 : object
    
    parameter:
            1. s_len:         sequence length (time step)         : int
            2. n_dim:         data dimension                      : int
            3. n_output:      # of output                         : int
    """
    
    i_tensor = layers.Input(shape=(s_len, n_dim)) # input tensor

    conv1_1 = layers.Conv1D(50, 21)(i_tensor)
    conv1_2 = layers.BatchNormalization()(conv1_1)
    conv1_3 = layers.Activation('relu')(conv1_2)

    conv2_1 = layers.Conv1D(50, 21)(conv1_3)
    conv2_2 = layers.BatchNormalization()(conv2_1)
    conv2_3 = layers.Activation('relu')(conv2_2)
    conv2_4 = layers.Dropout(0.2)(conv2_3)
    conv2_5 = layers.MaxPooling1D(2)(conv2_4)

    conv3_1 = layers.Conv1D(50, 11)(conv2_5)
    conv3_2 = layers.BatchNormalization()(conv3_1)
    conv3_3 = layers.Activation('relu')(conv3_2)

    flatted = layers.Flatten()(conv3_3)

    dense1 = layers.Dense(128, activation='relu')(flatted)
    dense2 = layers.Dense(64, activation='relu')(dense1)
    o_tensor = layers.Dense(n_output, activation='linear')(dense2)

    model = models.Model(i_tensor, o_tensor)
    
    return model



def l02(s_len, n_dim,
        n_unit1, drop_rate1, 
        n_unit2, drop_rate2, 
        n_output):
    """
    m03
    abstract: 2-layer lstm
    return: model                                                 : object
    
    parameter:
            1. s_len:         sequence length (time step)         : int
            2. n_dim:         data dimension                      : int
            3. n_unit1:       # of first layer lstm unit          : int
            4. n_unit2:       # of second layer lstm unit         : int
            5. drop_rate1:    drop out rate of first layer lstm   : float
            6. drop_rate2:    drop out rate of second layer lstm  : float
            7. n_output:      # of output                         : int
    """
    
    i_tensor = layers.Input(shape=(s_len, n_dim)) # input tensor

    lstm01 = layers.LSTM(n_unit1, return_sequences=True)(i_tensor)
    drop01 = layers.Dropout(drop_rate1)(lstm01)
    
    lstm02 = layers.LSTM(n_unit2)(drop01)
    drop02 = layers.Dropout(drop_rate2)(lstm02)
    b_norm = layers.BatchNormalization()(drop02)
            
    o_tensor = layers.Dense(n_output)(b_norm)
    
    model = models.Model(i_tensor, o_tensor)
    
    return model



def l03(s_len, n_dim,
        n_unit, drop_rate, 
        n_output):
    """
    m04
    abstract: 1-layer bidirectional lstm
    return: model                                                 : object
        
    parameter:
            1. s_len:         sequence length (time step)         : int
            2. n_dim:         data dimension                      : int
            3. n_unit:        # of lstm unit                      : int
            4. drop_rate:     drop out rate                       : float
            5. n_output:      # of output                         : int
    """
    
    i_tensor = layers.Input(shape=(s_len, n_dim)) # input tensor


    l01 = layers.LSTM(n_unit, dropout=drop_rate, return_sequences=True) # forward lstm
    l02 = layers.LSTM(n_unit, dropout=drop_rate, return_sequences=True, go_backwards=True) # backward lstm
    b_lstm = layers.Bidirectional(l01, backward_layer=l02)(i_tensor) # bidirectional lstm

    flatten = layers.Flatten()(b_lstm)
        
    o_tensor = layers.Dense(n_output, activation='linear')(flatten)
    
    model = models.Model(i_tensor, o_tensor)
    
    return model



class l04:
    """
    m05
    abstract: stacked, residual, bidirectional lstm
    return: model                                                 : object
    
    parameter:
            1. s_len:         sequence length (time step)         : int
            2. n_dim:         data dimension                      : int
            3. n_unit:        # of lstm unit                      : int
            4. n_output:      # of output                         : int
    """
    
    def __init__(self, s_len, n_dim, n_unit, n_output):
        self.n_unit = n_unit
        
        i_tensor = layers.Input(shape=(s_len, n_dim))
        
        r_net01 = self.r_net(i_tensor) # residual network
        b_unit01 = self.b_unit(r_net01, r_seq=True) # bidirectional lstm unit
        
        l01 = layers.LSTM(self.n_unit*2)(b_unit01) # forward lstm
        l01 = layers.Activation('relu')(l01)
        
        l_output = b_unit01[:, -1, :] # b_unit01's last output
        
        e_wise = tf.math.multiply(l01, l_output) # element wise multiplication
        batch = layers.BatchNormalization()(e_wise)
        
        o_tensor = layers.Dense(n_output)(batch)
        
        self.model = models.Model(i_tensor, o_tensor)
        
    # residual network
    def r_net(self, i_tensor):
        # i_tensor: input tensor
        b_unit01 = self.b_unit(i_tensor, r_seq=True)
        b_unit02 = self.b_unit(b_unit01, r_seq=True)

        e_wise = tf.math.multiply(b_unit01, b_unit02) # element wise multiplication
    
        return layers.BatchNormalization()(e_wise)
    
    # bidirectional lstm unit
    def b_unit(self, i_tensor, r_seq=False):
        # i_tensor: input tensor
        # r_seq: return sequences
        l01 = layers.LSTM(self.n_unit, return_sequences=r_seq) # forward lstm
        l02 = layers.LSTM(self.n_unit, return_sequences=r_seq, go_backwards=True) # backward lstm
        b_lstm = layers.Bidirectional(l01, backward_layer=l02)(i_tensor) # bidirectional lstm
        b_lstm = layers.Activation('relu')(b_lstm)
        
        return b_lstm
    
    def get_model(self):
        return self.model

    
    
from tensorflow.keras import optimizers
from tensorflow.keras.metrics import MeanAbsoluteError
from .c_metric import r2_score



# ***
def get_model(m_id, m_config):
    """
    abstract: build and return the model
    return: model                                                 : object
    
    parameter:
            1. m_id:          model id                            : string
            2. m_config:      model configuration                 : dict
    """
    
    if m_id == 'm01':
        s_len = m_config['s_len']
        model = l01(m_config['s_len'], 
                    m_config['n_dim'], 
                    m_config['n_unit'], 
                    m_config['drop_rate'], 
                    m_config['n_output'])
        
    elif m_id == 'm02':
        model = c01(m_config['s_len'], 
                    m_config['n_dim'], 
                    m_config['n_output'])
        
    elif m_id == 'm03':
        model = l02(m_config['s_len'], 
                    m_config['n_dim'], 
                    m_config['n_unit1'], 
                    m_config['drop_rate1'], 
                    m_config['n_unit2'], 
                    m_config['drop_rate2'], 
                    m_config['n_output'])
        
    elif m_id == 'm04':
        model = l03(m_config['s_len'], 
                    m_config['n_dim'],
                    m_config['n_unit'], 
                    m_config['drop_rate'], 
                    m_config['n_output'])
        
    elif m_id == 'm05':
        model = l04(m_config['s_len'], 
                    m_config['n_dim'], 
                    m_config['n_unit'], 
                    m_config['n_output']).get_model()
        
    else:
        print('[ERROR] Not exist model id')
        return None
    
    model.compile(optimizer='Adam', 
                  loss='mean_squared_error', 
                  metrics=[MeanAbsoluteError(name='mae'), r2_score])
    
    return model