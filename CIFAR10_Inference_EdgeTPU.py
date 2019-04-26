import argparse
import io
import time
from datetime import datetime
import numpy as np

from PIL import Image, ImageDraw, ImageFont

from edgetpu.classification.engine import ClassificationEngine

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--model', help='File path of Tflite model.', required=True)
    parser.add_argument(
      '--image', help='File path of file.', required=True)
    args = parser.parse_args()
    
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    img = Image.open(args.image)
    #np_image = np.array(img)

    # Load Engine
    engine = ClassificationEngine(args.model)

    lap_time = time.time()

    # Run inference. 
    for result in engine.ClassifyWithImage(img, top_k=3):
        print ('---------------------------')
        print (label_names[result[0]])
        print ('Score : ', result[1])

    previous_time = lap_time
    lap_time = time.time()
    print("Elapsed time for the last inference: ", lap_time - previous_time)


if __name__ == '__main__':
    main()    
