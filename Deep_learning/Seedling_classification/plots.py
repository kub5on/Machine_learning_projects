import os
import random
import numpy as np
import tensorflow as tf
import keras

# Stała wartość seed
seed = 42

# Ustawienie seedów dla reprodukowalności
os.environ["PYTHONHASHseed"] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
keras.utils.set_random_seed(seed)

# Wymuszenie deterministycznych operacji w TensorFlow
os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.config.experimental.enable_op_determinism()

# Parametryzacja
train_dir = "plants_train"
val_dir = "plants_test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

