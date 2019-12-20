"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import SVC


# Face Recognition opencv import start
import time
import cv2
import imutils







def main(args):

    with tf.Graph().as_default():

        with tf.Session() as sess:

            np.random.seed(seed=args.seed)

            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
                if (args.mode=='TRAIN'):
                    dataset = train_set
                elif (args.mode=='CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')


            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            # nrof_images = len(paths)
            # nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            # emb_array = np.zeros((nrof_images, embedding_size))
            # for i in range(nrof_batches_per_epoch):
            #     start_index = i*args.batch_size
            #     end_index = min((i+1)*args.batch_size, nrof_images)
            #     paths_batch = paths[start_index:end_index]
            #     images = facenet.load_data(paths_batch, False, False, args.image_size)
            #     feed_dict = { images_placeholder:images, phase_train_placeholder:False }
            #     emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if (args.mode=='TRAIN'):
                # Train classifier
                print('Training classifier')
                model = SVC(kernel='linear', probability=True)
                model.fit(emb_array, labels)

                # Create a list of class names
                class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                # Saving classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

            elif (args.mode=='CLASSIFY'):


                vs = cv2.VideoCapture(0)
                time.sleep(2.0)


                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)


                # load our serialized face detector from disk
                print("[INFO] loading face detector...")
                protoPath = os.path.sep.join([args.detector, "deploy.prototxt"])
                modelPath = os.path.sep.join([args.detector,
                                              "res10_300x300_ssd_iter_140000.caffemodel"])
                detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

                while True:
                    ret, frame = vs.read()

                    frame = imutils.resize(frame, width=600)
                    (h, w) = frame.shape[:2]

                    # construct a blob from the image
                    imageBlob = cv2.dnn.blobFromImage(
                        cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                        (104.0, 177.0, 123.0), swapRB=False, crop=False)


                    detector.setInput(imageBlob)
                    detections = detector.forward()


                    # loop over the detections
                    for i in range(0, detections.shape[2]):
                        # extract the confidence (i.e., probability) associated with
                        # the prediction
                        confidence = detections[0, 0, i, 2]

                        # filter out weak detections
                        if confidence > 0.5:
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")

                            # extract the face ROI
                            face = frame[startY:endY, startX:endX]

                            cv2.imwrite("datasets/exntu_faces/result2/Young_Yoo/1.png", face)

                            nrof_images = len(paths)
                            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
                            emb_array = np.zeros((nrof_images, embedding_size))

                            print('nrof_images',nrof_images)
                            print('nrof_batches_per_epoch',nrof_batches_per_epoch)
                            for i in range(nrof_batches_per_epoch):
                                start_index = i*args.batch_size
                                end_index = min((i+1)*args.batch_size, nrof_images)
                                paths_batch = paths[start_index:end_index]
                                images = facenet.load_data(paths_batch, False, False, args.image_size)
                                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)


                                # Start classify
                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                                for i in range(len(best_class_indices)):
                                    if best_class_probabilities[i] > 0.70:
                                        print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                                    else:
                                        print('Unknown')


                            #face = frame[startY - 20 : endY + 20, startX - 20:endX + 20]
                            (fH, fW) = face.shape[:2]
                            # ensure the face width and height are sufficiently large
                            if fW < 5 or fH < 5:
                                continue




                            # start prediction
                            # feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                            # emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                            #
                            # predictions = model.predict_proba(emb_array)
                            # best_class_indices = np.argmax(predictions, axis=1)
                            # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            #
                            # for i in range(len(best_class_indices)):
                            #     if best_class_probabilities[i] > 0.70:
                            #         print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                            #     else:
                            #         print('Unknown')




                            y = startY - 10 if startY - 10 > 10 else startY + 10

                            text = "{}".format('unknown')
                            cv2.rectangle(frame, (startX, startY), (endX, endY),
                                          (0, 0, 255), 2)
                            cv2.putText(frame, text, (startX, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)



                    # show the output frame
                    cv2.imshow("Frame", frame)
                    key = cv2.waitKey(1) & 0xFF

                    # if the `q` key was pressed, break from the loop
                    if key == ord("q"):
                        break

                        # Start classify
                        # predictions = model.predict_proba(emb_array)
                        # best_class_indices = np.argmax(predictions, axis=1)
                        # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                        #
                        # for i in range(len(best_class_indices)):
                        #     if best_class_probabilities[i] > 0.70:
                        #         print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                        #     else:
                        #         print('Unknown')
                        #
                        # accuracy = np.mean(np.equal(best_class_indices, labels))
                        # print('Accuracy: %.3f' % accuracy)


def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    for cls in dataset:
        paths = cls.image_paths
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
    return train_set, test_set


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help='Indicates if a new classifier should be trained or a classification ' +
                             'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. ' +
                             'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset',
                        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +
                             'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true')
    parser.add_argument('--test_data_dir', type=str,
                        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
                        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
                        help='Use this number of images from each class for training and the rest for testing', default=10)
    parser.add_argument("-d", "--detector", required=True,
                        help="path to OpenCV's deep learning face detector")

    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    #embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

    # Face Recognition opencv import end


    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
