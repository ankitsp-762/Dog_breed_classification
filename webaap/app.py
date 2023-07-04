import numpy as np
import gradio as gr
import tensorflow as tf
import tensorflow_hub as hub
# from labels import get_label
from model import get_model

labels = ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale','american_staffordshire_terrier', 'appenzeller','australian_terrier', 'basenji', 'basset', 'beagle','bedlington_terrier', 'bernese_mountain_dog','black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound','bluetick', 'border_collie', 'border_terrier', 'borzoi','boston_bull', 'bouvier_des_flandres', 'boxer','brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff','cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua','chow', 'clumber', 'cocker_spaniel', 'collie','curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo','doberman', 'english_foxhound', 'english_setter','english_springer', 'entlebucher', 'eskimo_dog','flat-coated_retriever', 'french_bulldog', 'german_shepherd','german_short-haired_pointer', 'giant_schnauzer','golden_retriever', 'gordon_setter', 'great_dane','great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael','ibizan_hound', 'irish_setter', 'irish_terrier','irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound','japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier','komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier','leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog','mexican_hairless', 'miniature_pinscher', 'miniature_poodle','miniature_schnauzer', 'newfoundland', 'norfolk_terrier','norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog','otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian','pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler','saint_bernard', 'saluki', 'samoyed', 'schipperke','scotch_terrier', 'scottish_deerhound', 'sealyham_terrier','shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier','soft-coated_wheaten_terrier', 'staffordshire_bullterrier','standard_poodle', 'standard_schnauzer', 'sussex_spaniel','tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier','vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel','west_highland_white_terrier', 'whippet','wire-haired_fox_terrier', 'yorkshire_terrier']
labels = np.array(labels)
BATCH_SIZE=32
IMG_SIZE=224
model = get_model()

def preprocess_image(img,img_size=IMG_SIZE):
  img=tf.io.read_file(img)
  img=tf.image.decode_jpeg(img,channels=3)
  image=tf.image.convert_image_dtype(img,tf.float32)
  image=tf.image.resize(image,size=[img_size,img_size])

  return image

def create_data_batches(X,y=None,batch_size=BATCH_SIZE,valid_data=False,test_data=False):
    data=tf.data.Dataset.from_tensor_slices(tf.constant(X))
    data_batch=data.map(preprocess_image).batch(BATCH_SIZE)
    return data_batch



def dog_breed(input):
   custom_image_paths = []
   custom_image_paths.append(input)
   custom_data = create_data_batches(custom_image_paths, test_data=True)
   pred_prob = model.predict(custom_data).flatten()
   pred_prob = np.array(pred_prob)
   top_preds_label=labels[pred_prob.argsort()[-5:][::-1]]
   top_preds_value=pred_prob[pred_prob.argsort()[-5:][::-1]]
   confidences = {top_preds_label[i]: float(top_preds_value[i]) for i in range(5)}
   return confidences


gr.Interface(fn=dog_breed, 
             inputs=gr.Image(type = "filepath", shape=(224, 224)),
             outputs=gr.Label(num_top_classes=3),
             title = "Dog Breed Classification").launch()
