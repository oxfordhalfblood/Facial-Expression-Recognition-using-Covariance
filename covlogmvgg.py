# /***************************************************************************************
# *    Title: Facial Expression Recognition using covariance pooling
# *    Author: Afrida Tabassum
# *    Date: 20.11.2019
# *    Code version: 1.1
# ***************************************************************************************/

import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import applications
from sklearn.preprocessing import scale
import argparse
import os
import numba
from keras.models import load_model
from keras.models import Model
from keras.applications.vgg16 import preprocess_input
from numba import cuda
#numba.cuda.select_device(1)
import sys
import math
import pickle
from scipy.linalg import logm
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import time
from tqdm import tqdm
from memory_profiler import profile

@profile
def main(args):
    print("Loading Pretrained model VGG16")
    model = applications.VGG16(include_top=False, weights='imagenet')
    model_pre = Model(model.inputs, model.layers[-2].output)

    print("Got Weights")

    data_dir = args.data_dir
    #datagen_top = ImageDataGenerator(rescale=1. / 255,rotation_range=9,horizontal_flip=True)
    datagen_top = ImageDataGenerator()
    generator_top = datagen_top.flow_from_directory(
        data_dir,
        target_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False)
    
    nb_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)
    labels = generator_top.classes
    print(labels.shape)
   
    predict_size_data = int(math.ceil(nb_samples / args.batch_size))
    print("Extracting features...")
    bottleneck_features = model_pre.predict_generator(
        generator_top, predict_size_data, verbose=1)
    

    #Covariance Calculation
    print("Getting covariance matrix ")
    covarianceDimension = bottleneck_features.shape[3]
    longVectorDimension = ((covarianceDimension * (covarianceDimension + 1))/2)
    longVector = np.zeros((bottleneck_features.shape[0],int(longVectorDimension)))
    for i in tqdm(range(0,bottleneck_features.shape[0])):
        X=bottleneck_features[i]
        y=X.reshape(X.shape[0]*X.shape[1],X.shape[2])   #Shape (14*14, 512)
        yt=np.transpose(y)
        #print(y.shape)
        #covarianceMatrix = np.cov(y)
        covarianceMatrix = yt.dot(y)
        #Matrix Logarithm
        identityMatrix = np.identity(covarianceDimension)
        lambdaValue = 1.e-3
        covarianceMatrixNew = ((identityMatrix*lambdaValue) + covarianceMatrix)
        matrixLogarithm = logm(covarianceMatrixNew)

        flattenVector = list(matrixLogarithm[np.triu_indices(bottleneck_features.shape[3])])  #If matrix logarithm is applied change covariance matrix to matrixLogarithm
        longVector[i] = flattenVector
    print('Long Vector Dimension ',longVector.shape)
    print("Scaling Features")
    bottleneck_features = longVector
    #bottleneck_features = scale(longVector) 
    classifier_filename_exp = os.path.expanduser(args.classifier_filename)
    ##After long Vector    
    if (args.mode=='TRAIN'):
        
        print('Training classifier')
        start= time.time()
        modelSVC = SVC(kernel='linear',probability=True,verbose=True)
        modelSVC.fit(bottleneck_features, labels)
        end = time.time()
        print("Time taken", (end - start))
        with open(classifier_filename_exp, 'wb') as outfile:
            joblib.dump(modelSVC,outfile)
        print('Saved classifier model to file "%s"' % classifier_filename_exp)

    elif (args.mode=='CLASSIFY'):
    
        print('Testing classifier')
        with open(classifier_filename_exp, 'rb') as infile:
            modelSVC = joblib.load(infile)
        print('Loaded classifier model from file "%s"' % classifier_filename_exp)     
        print("Predicting Images")
        predictions = modelSVC.predict_proba(bottleneck_features)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        
        arr=[]
        for i in range(len(best_class_indices)):
            arr.append((predictions[i],best_class_indices[i],labels[i]))
        print(arr)
        # pickle.dump(arr, open('classification_data.p','wb'))
        accuracy = 100*np.mean(np.equal(best_class_indices, labels))
        print('Total Accuracy: %.3f' % accuracy)
    '''
    train_data = np.load('bottleneck_features_train.npy')
    train_data = train_data.reshape(train_data.shape[0],-1)
    train_data = scale(train_data)
    print("Shape of train data ",train_data.shape)
    '''


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:])) 
