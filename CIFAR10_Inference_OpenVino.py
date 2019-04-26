import argparse
import io
import time
from datetime import datetime
import numpy as np
import cv2 as cv

def pre_process_image(image, img_height=32):

    # Normalize to keep data between 0 - 1
    processedImg = (np.array(image) - 0) / 255.0
    processedImg = processedImg.astype('float32')

    return processedImg

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--modelxml', help='File path of model.xml file.', required=True)
    parser.add_argument(
      '--modelbin', help='File path of model.bin file.', required=True)
    parser.add_argument(
      '--imagePath', help='File path of the image.', required=True)
    args = parser.parse_args()
    
    label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Plugin initialization for specified device and load extensions library if specified.
    model_xml = args.modelxml
    model_bin = args.modelbin

    # Read IR
    net = cv.dnn.readNet(model_xml,model_bin)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_MYRIAD)
    #net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU) #Not working on Rasp.
   
    # Load the image
    image = cv.imread(args.imagePath)
    processedImg = pre_process_image(image)
    
    # Run inference   
    blob = cv.dnn.blobFromImage(processedImg, size=(32, 32), ddepth=cv.CV_32F)  
    net.setInput(blob)
    
    lap_time = time.time()
    
    out = net.forward()
    
    previous_time = lap_time
    lap_time = time.time()
    
    # Access the results 
    print("Elapsed time for the last inference: ", lap_time - previous_time)
    results = np.column_stack((label_names, out.flatten()))
    print(results)
    

if __name__ == '__main__':
    main()    
