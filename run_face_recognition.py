from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2

import facenet
import detect_face
import os

import pickle

import requests
import multiprocessing
from multiprocessing.dummy import Pool
from threading import Timer


from multiprocessing.pool import ThreadPool


import numpy as np
import datetime

import entry_exit
import threading

tp = ThreadPool(processes=4)

# Arduino Address URL
ARDUINO_API = 'http://XXX.XXX.XXX.XXX/'
# Please add your arduino access id
ARDUINO_ID = ''
# Please add your arduino access password
ARDUINO_PASS = ''
ARDUINO_FLAG = False
ARDUINO_TIME = 5.0
pool = Pool(10)
flag_pool = Pool(5)

path_collect = './collect_images'

bytes=b''


def main():
    # Create Collect Image Folder
    collect_dir = os.path.expanduser(path_collect)
    if not os.path.isdir(collect_dir):
        # Create directory
        os.makedirs(path_collect)

if __name__ == '__main__':
    main()

def arduino_timer():
    global ARDUINO_FLAG
    ARDUINO_FLAG = False


def update_time_table(name):
    # try:
    name_arr = name.split(" ")
    if len(name_arr) > 0:
        name_str = name_arr[1]

        time_res = pool.apply_async(entry_exit.update_time_sheet, [name_str])
        try:
            print('update_time_table', time_res.get(timeout=2))

        except multiprocessing.TimeoutError:
            print("time table timeout error")

def command_door_open():

    door_result = pool.apply_async(requests.get, [ARDUINO_API], dict(auth=(ARDUINO_ID, ARDUINO_PASS)))

    try:
        print('open_door', door_result.get(timeout=1))

    except multiprocessing.TimeoutError:
        print("Abort open_door timeout")


def open_door(name):
    global ARDUINO_FLAG
    if ARDUINO_FLAG:
        return
    ARDUINO_FLAG = True

    # Open door by Aruduino
    command_door_open()

    # update excel sheet
    # update_time_table(name)
    Timer(ARDUINO_TIME, arduino_timer).start()


def write_image_file(image_name, probability, frame):
    if len(probability) > 0:
        probability = probability[0]
        probability = str(int(probability * 10000) / 100) + '%'

    # Store the image into a classified folder
    str_path = path_collect + '/' + image_name
    str_path = os.path.expanduser(str_path)
    if not os.path.isdir(str_path):
        # Create directory
        os.makedirs(str_path)
    img_item = str_path + '/' + '{date:%Y-%m-%d %H:%M:%S}_{name}_{probability}.jpg'.format( date=datetime.datetime.now(), name = image_name, probability = probability)
    cv2.imwrite(img_item, frame)

    # save log in text file
    # with open('new.txt', 'a') as outfile:
    #     outfile.write(" {0} \n".format(img_item))
    # outfile.close()

current_started = datetime.datetime.now()
end = False

import time
import os
def remove(path):
    """
    Remove the file or directory
    """
    if os.path.isdir(path):
        try:
            os.rmdir(path)
        except OSError:
            print ("Unable to remove folder: %s" % path)
    else:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            print ("Unable to remove file: %s" % path)

#----------------------------------------------------------------------
def cleanup(number_of_days):
    print('start cleanup old images')
    """
    Removes files from the passed in path that are older than or equal
    to the number_of_days
    """
    # delete files less than 60 seconds
    time_in_secs = time.time() - (number_of_days * 24 * 60 * 60)
    path =  os.getcwd() + '/collect_images'

    for root, dirs, files in os.walk(path, topdown=False):
        for file_ in files:
            full_path = os.path.join(root, file_)
            stat = os.stat(full_path)
            if stat.st_mtime <= time_in_secs:
                remove(full_path)

        if not os.listdir(root):
            remove(root)

def run():
    # app runs every 12 hours, so delete files whenever app start running
    cleanup(30) # store images for 30 days

    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './src/align/')

            minsize = 20  # minimum size of face
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            margin = 44
            frame_interval = 3
            batch_size = 1000
            image_size = 182
            input_image_size = 160

            print('Loading feature extraction model')
            modeldir = './20180402-114759/20180402-114759.pb'
            facenet.load_model(modeldir)

            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            classifier_filename = './classifier/my_classifier.pkl'
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, class_names) = pickle.load(infile)
                # print('load classifier file-> %s' % classifier_filename_exp)

            print('class_names',class_names)

            video_capture = cv2.VideoCapture(0)
            c = 0
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            # #video writer
            print('Start Recognition!')
            flag = False
            time_started = datetime.datetime.now()

            while True:
                # destroy app period time
                timenow = datetime.datetime.now()
                elapsed_time = int(timenow.timestamp() - time_started.timestamp())

                if elapsed_time > 60 * 60 * 12:
                   break

                if flag == False:
                    flag = True
                else:
                    flag = False

                    ret, frame = video_capture.read()

                    #Optional roate frame by 90 degrees
                    num_rows, num_cols = frame.shape[:2]
                    rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 90, 1)
                    frame = cv2.warpAffine(frame, rotation_matrix, (num_cols, num_rows))

                    frame = cv2.resize(frame, (0,0), fx=1, fy=1)    #resize frame (optional)

                    timeF = frame_interval

                    if (c % timeF == 0):

                        if frame.ndim == 2:
                            frame = facenet.to_rgb(frame)
                        frame = frame[:, :, 0:3]
                        bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]

                        if nrof_faces > 0:
                            det = bounding_boxes[:, 0:4]
                            bb = np.zeros((nrof_faces,4), dtype=np.int32)

                            for i in range(nrof_faces):
                                cropped =[]
                                scaled = []
                                scaled_reshape = []
                                emb_array = np.zeros((1, embedding_size))

                                bb[i][0] = det[i][0]
                                bb[i][1] = det[i][1]
                                bb[i][2] = det[i][2]
                                bb[i][3] = det[i][3]

                                # ensure the face width and height are sufficiently large
                                if abs(bb[i][0] - bb[i][2]) < 40 or abs(bb[i][1] - bb[i][3]) < 40:
                                    continue

                                # inner exception
                                if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                                    continue

                                cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                                cropped[0] = facenet.flip(cropped[0], False)
                                scaled.append(misc.imresize(cropped[0], (image_size, image_size), interp='bilinear'))
                                scaled[0] = cv2.resize(scaled[0], (input_image_size,input_image_size),
                                                       interpolation=cv2.INTER_CUBIC)
                                scaled[0] = facenet.prewhiten(scaled[0])
                                scaled_reshape.append(scaled[0].reshape(-1,input_image_size,input_image_size,3))
                                feed_dict = {images_placeholder: scaled_reshape[0], phase_train_placeholder: False}
                                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                predicted_probability = predictions[np.arange(len(best_class_indices)), best_class_indices]
                                # cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                                #plot result idx under box
                                text_x = bb[i][0]
                                text_y = bb[i][3] + 20

                                result_names = 'unknown'

                                if len(best_class_indices) > 0 and predicted_probability[0] > 0.6:
                                    result_names = class_names[best_class_indices[0]]

                                    # write Images
                                    write_image_file(result_names, predicted_probability, frame)

                                    g = 255; r = 0
                                    if result_names.lower() == 'unknown':
                                        g = 0; r = 255
                                    else:
                                        print('%s: %.3f, %s' % (result_names, predicted_probability, datetime.datetime.now()))
                                        # open_door(result_names)
                                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, g, r), 2)    #boxing face
                                        #cv2.rectangle(frame, (endX, startY + int((endY - startY)*int((1-proba)*100)/100)), (endX + 10, endY), (0, 0, 255), cv2.FILLED)
                                        cv2.rectangle(frame, (bb[i][2], bb[i][1] + int((bb[i][3] - bb[i][1]) * int((1 - predicted_probability[0]) * 100) / 100)), (bb[i][2] + 10, bb[i][3]), (0, g, r), cv2.FILLED)

                                        cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                    1, (0, g, r), thickness=1, lineType=2)

                                else:
                                    # Save Unknown Images
                                    write_image_file(result_names, predicted_probability, frame)

                                    cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                                1, (0, 0, 255), thickness=1, lineType=2)
                                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 2)    #boxing face

                    # if you want to display realtime video
                    #cv2.imshow('Video', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            video_capture.release()

run()