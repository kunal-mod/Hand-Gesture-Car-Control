import os

#from carcode import detect_fn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

PATH_TO_SAVED_MODEL = os.path.join('Tensorflow','workspace','models','my_ssd_mobnet','export','saved_model')
model = tf.saved_model.load(PATH_TO_SAVED_MODEL)
detect_fn=model.signatures['serving_default']





import cv2 
import numpy as np

lb=["ThubmsUp","ThumbsDown", "Peace", "Livelong"]


cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while cap.isOpened(): 
    ret, frame = cap.read()
    image_np = np.array(frame)
    
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    if(detections['detection_scores'][0]>0.80):
        print(lb[detections['detection_classes'][0]])
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
