from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import os
import tensorflow as tf
import numpy as np
import facenet
import detect_face
import argparse
import shutil

# Add argument
argument = argparse.ArgumentParser()
argument.add_argument("-r", "--raw_image", default=False,
                      help="refine raw image data")
args = vars(argument.parse_args())

# Output directory name
output_dir_path = './output/'
output_dir = os.path.expanduser(output_dir_path)

# Check output images directory exits
if os.path.isdir(output_dir):
    shutil.rmtree(output_dir_path)
# Create directory
os.makedirs(output_dir_path)

datadir = './dataset'
dataset = facenet.get_dataset(datadir)

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './src/align/')

minsize = 20  # minimum size of face
threshold = [0.6, 0.7, 0.7]  # three steps's threshold
factor = 0.709  # scale factor
margin = 44
image_size = 182

# Add a random key to the filename to allow alignment using multiple processes
random_key = np.random.randint(0, high=99999)
bounding_boxes_filename = os.path.join(output_dir, 'bounding_boxes_%05d.txt' % random_key)
print('images alignment started')

with open(bounding_boxes_filename, "w") as text_file:
    nrof_images_total = 0
    nrof_successfully_aligned = 0
    for cls in dataset:
        output_class_dir = os.path.join(output_dir, cls.name)
        if not os.path.exists(output_class_dir):
            os.makedirs(output_class_dir)
        for image_path in cls.image_paths:
            nrof_images_total += 1
            filename = os.path.splitext(os.path.split(image_path)[1])[0]
            # print('filename', filename)
            output_filename = os.path.join(output_class_dir, filename + '.png')
            # print(image_path)
            if not os.path.exists(output_filename):
                try:
                    img = misc.imread(image_path)
                    # print('read data dimension: ', img.ndim)
                except (IOError, ValueError, IndexError) as e:
                    errorMessage = '{}: {}'.format(image_path, e)
                    print(errorMessage)
                else:
                    if img.ndim < 2:
                        # print('Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))
                        continue
                    if img.ndim == 2:
                        img = facenet.to_rgb(img)
                        # print('to_rgb data dimension: ', img.ndim)
                    img = img[:, :, 0:3]
                    # print('after data dimension: ', img.ndim)

                    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                    nrof_faces = bounding_boxes.shape[0]
                    # print('detected_face: %d' % nrof_faces)

                    if nrof_faces > 0:

                        det_data = bounding_boxes[:, 0:4]

                        for i in range(nrof_faces):
                            det = np.squeeze(det_data[i])
                            img_size = np.asarray(img.shape)[0:2]
                            bb_temp = np.zeros(4, dtype=np.int32)

                            x1 = 0; x2=0; y1=0; y2 =0
                            if args["raw_image"] == True:
                                x1 = - 25; y1 = - 35; x2 = + 25; y2 = + 20

                            bb_temp[0] = det[0] + x1
                            bb_temp[1] = det[1] + y1
                            bb_temp[2] = det[2] + x2
                            bb_temp[3] = det[3] + y2

                            # out of range images
                            flag = True
                            for j in range(len(bb_temp)):
                                if bb_temp[j] < 0:
                                    flag = False

                            if flag == True:
                                cropped_temp = img[bb_temp[1]:bb_temp[3], bb_temp[0]:bb_temp[2], :]
                                scaled_temp = misc.imresize(cropped_temp, (image_size, image_size), interp='bilinear')
                                nrof_successfully_aligned += 1
                                misc.imsave(output_filename, scaled_temp)
                                text_file.write('%s %d %d %d %d\n' % (output_filename, bb_temp[0], bb_temp[1], bb_temp[2], bb_temp[3]))
                    else:
                        print('Unable to align "%s"' % image_path)
                        text_file.write('%s\n' % (output_filename))

print('Total number of images: %d' % nrof_images_total)
print('Number of successfully aligned images: %d' % nrof_successfully_aligned)



