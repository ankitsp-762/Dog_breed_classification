import tensorflow as tf
import tensorflow_hub as hub

def load_model(model_path):
  model=tf.keras.models.load_model(model_path,
                                   custom_objects={"KerasLayer":hub.KerasLayer})
  return model

def get_model():
  return load_model("20230117-195757-full_image_set_mobilenetv2-Adam.h5")
