# import the opencv library 
import cv2 
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np


IMG_SIZE = 256
resize_resacle = tf.keras.Sequential([
  tf.keras.layers.Resizing(IMG_SIZE, IMG_SIZE),
  tf.keras.layers.Rescaling(1./255)
])

class_names = ['Keshav', 'Rohit', 'pakhi']
loaded_model = load_model("my_model")


vid = cv2.VideoCapture(0) 

while(True): 
	ret, image = vid.read() 
	image = cv2.flip(image, 1)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
	face = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )
	x, y, width, height = 100, 50, 200, 150

	for (x,y,width,height) in face:
		face_roi = image[y:y + height, x:x + width]
		image_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
		preprocessed_image = resize_resacle.predict(tf.expand_dims(image_rgb, axis=0))[0]
		tensor_shape_32 = (32, 256, 256, 3)
		tensor_32 = tf.expand_dims(preprocessed_image, axis=0) 
		predictions = loaded_model.predict(tensor_32.numpy().astype('uint8'))
		# print(predictions[0])
		text = class_names[np.argmax(predictions[0])] + " Accuracy : "+str(round(predictions[0][np.argmax(predictions[0])]*100,2))+" %"
		font = cv2.FONT_HERSHEY_SIMPLEX
		font_scale = 0.8
		font_thickness = 1
		text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
		text_position = (x + (width - text_size[0]) // 2, y + height + text_size[1] + 10)
		cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)  # (0, 255, 0) is the color in BGR format, and 2 is the thickness
		cv2.putText(image,text, text_position, font, font_scale, (0, 255, 0), font_thickness)

	cv2.imshow('frame', image) 


	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

# After the loop release the cap object 
vid.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
