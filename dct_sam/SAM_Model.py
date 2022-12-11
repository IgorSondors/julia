from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Dense, GlobalAveragePooling2D, Conv2D, Input, Flatten, concatenate
import tensorflow as tf

from train_dct_sam import *

# creating a custom train_step to override the train_step method of sub_classing model
def train_step_sam(self, data, rho=0.05):
    """
    Overrides the train_step method of Model
    
    Args:
        data : Data on which model is to be trained
        rho  : Hyperparameter Rho indicating the size of neighborhood
    """
    
    sample_weight = None
    x, y = data

    # Opening Gradient Tape scope to record operations during 1st forward pass
    with tf.GradientTape() as tape:
        y_pred = self(x, training=True)
        # Calculating loss to calculate gradients
        loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)

    
    trainable_vars = self.trainable_variables
    # Calculating gradients with respect trainable variable
    gradients = tape.gradient(loss, trainable_vars)

    """
    This is the first step which involves calculating the point w_adv with highest loss and virtually moving to that point so that we can get gradient at that point. 
    """
    eps_w_ls = [] # list to store the updates done to trainable variables in first step
    
    #computing the norm of gradients which is required for computing eps_w
    grad_norm = tf.linalg.global_norm(gradients)
    
    # Iterating over trainable_vars
    for i in range(len(trainable_vars)):
        # we will calculate eps_w to find w_adv point having highest loss in rho neighborhood
        eps_w = tf.math.multiply(gradients[i], rho / grad_norm )
        # temporarily moving to w_adv point
        trainable_vars[i].assign_add(eps_w)
        # storing updates done in eps_w_ls list 
        eps_w_ls.append(eps_w)

    # Opening Gradient Tape scope to record operations during 2nd forward pass
    with tf.GradientTape() as tape:
        y_pred = self(x, training=True) 
        # Calculating loss to calculate gradient at w_adv point
        loss = self.compiled_loss(y, y_pred, sample_weight=sample_weight, regularization_losses=self.losses)
        
    trainable_vars = self.trainable_variables
    #computing gradient at w_adv which is our objective in this first step
    gradients = tape.gradient(loss, trainable_vars)

    """
    This is the second step in SAM where we will do actual update at the initial point from the gradient calculated at adversial point w_adv 
    """
    
    for i in range(len(trainable_vars)):
        # Going back to orignal parameters
        trainable_vars[i].assign_sub(eps_w_ls[i])
    
    # Updating parameters with gradients computed at w_adv
    self.optimizer.apply_gradients(zip(gradients, trainable_vars))

    # Updating the metrics.
    self.compiled_metrics.update_state(y, y_pred, sample_weight=sample_weight)

    # returns a dictionary mapping metric names (including the loss) to their current value.
    return {m.name: m.result() for m in self.metrics}

# using model subclassing so we can override the train_step
class MyModel(Model):

    # defining the model, for this notebook I will be using ResNet50 model and changing the output layer
    def __init__(self, base_model):
        super(MyModel, self).__init__(name='julia_sam')
        self.base=base_model
     
    def call(self, x):
        x = self.base(x)
        return x
    
    # overriding the train_step method  with our custom train_step_sam created in earlier cell
    def train_step(self, data):
        return train_step_sam(self, data, rho=0.05) # using rho as 0.05 you can tune this hyperparameter


base_model=create_model(128)
# Instantiating our ResNet Model
model = MyModel(base_model)
model.build((None, 128,128, 3))

# compiling our model with SGD as base optimizer
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

tf.keras.utils.plot_model(
    model,
    to_file="{}/sam_custom.png".format('/home/yandex/igor/julia/dct_sam'),
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
    )

print(model.summary())