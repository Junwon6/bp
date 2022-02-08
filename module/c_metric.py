import tensorflow.keras.backend as K

def r2_score(y_true, y_pred):
    """
    abstract: calculate r squared score
    return: r squared score
    
    parameter:
            1. y_true:        ground truth                        : float
            2. y_pred:        model prediction                    : float
    """
    
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )