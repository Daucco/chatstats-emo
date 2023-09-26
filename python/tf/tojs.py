# Simple script to convert a TF model into TFJS Layers format
import sys
import tensorflow as tf
import tensorflowjs as tfjs

if __name__ == "__main__":
    tf_model_path = sys.argv[1]
    tfjs_model_dir = sys.argv[2]

    tf_model = tf.keras.models.load_model(tf_model_path)
    tfjs.converters.save_keras_model(tf_model, tfjs_model_dir)

