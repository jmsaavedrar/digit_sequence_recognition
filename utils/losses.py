import tensorflow as tf

def l2_regularization_loss(model, weight_decay):
    variable_list = []
    for variable in model.trainable_variables :
        if 'kernel' in variable.name :             
            variable_list.append(tf.nn.l2_loss(variable))        
    val_loss = tf.add_n(variable_list)
    return val_loss*weight_decay; 


def my_crossentropy_loss(y_true, y_pred):
    mask = tf.equal(y_true,1)
    y_pred = tf.keras.activations.softmax(y_pred, axis = -11)
    vals = tf.cast(tf.boolean_mask(y_pred, mask), tf.float32)    
    ce = tf.reduce_mean(-tf.math.log(vals))
    return ce

def multiple_crossentropy_loss(y_true, y_pred):
    mask = tf.equal(y_true,1)
    y_pred = tf.keras.activations.softmax(y_pred, axis = -1)
    vals = tf.cast(tf.boolean_mask(y_pred, mask), tf.float32)    
    ce = tf.reduce_mean(-tf.math.log(vals + 0.00001))
    return ce
    
def crossentropy_loss(y_true, y_pred):
    """
    shape of y_true = [Bx10]
    shape of y_pred = [Bx10]
    This is the classical categorical crossentropy
    """
    ce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(y_true, y_pred)    
    return ce

def crossentropy_l2_loss(model, weight_decay = 0):
    def loss(y_true, y_pred):
        """ 
        This uses crossentropy plus l2 regularization
        """
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)
        l2_loss = l2_regularization_loss(model, weight_decay)
        return ce + l2_loss
    return loss

#constrastive loss for triplet larning
def triplet_loss(margin = 20):
    def loss(y_true, y_pred):
        #y_true will be used for training cross_entropy
        e_a, e_p, e_n = tf.split(y_pred, axis = 1, num_or_size_splits = 3)    
        d_p = tf.sqrt(tf.reduce_sum(tf.square(e_a - e_p), 2))
        d_p_hard = tf.math.reduce_mean(d_p)
        d_n = tf.sqrt(tf.reduce_sum(tf.square(e_a - e_n), 2))
        d_n_hard = tf.math.reduce_mean(d_n)
        #hardest negative and hardest positive
        return tf.maximum(1e-10, d_p_hard + margin - d_n_hard)
    return loss

#crossentropy loss for triplets    
def crossentropy_triplet_loss(y_true, y_pred):
    y_true_a, y_true_p, y_true_n = tf.split(y_true, axis = 1, num_or_size_splits = 3)
    cl_a, cl_p, cl_n = tf.split(y_pred, axis = 1, num_or_size_splits = 3)            
    ce_a = tf.keras.losses.categorical_crossentropy(tf.squeeze(y_true_a), tf.squeeze(cl_a), from_logits=True)
    ce_p = tf.keras.losses.categorical_crossentropy(tf.squeeze(y_true_p), tf.squeeze(cl_p), from_logits=True)
    ce_n = tf.keras.losses.categorical_crossentropy(tf.squeeze(y_true_n), tf.squeeze(cl_n), from_logits=True)    
    ce = (ce_a + ce_p + ce_n) / 3.0 
    return ce


if __name__ == '__main__' :
    y_true = tf.constant([[0,1,0],[0,0,1]], tf.float32)
    y_pred = tf.constant([[0,1,0],[0,0,1]], tf.float32)
    print(y_true)
    print(y_pred)
    ce = my_crossentropy_loss(y_true, y_pred)
    
    print(ce)
