
import cv2
import cv2 as cv
import numpy as np
from numpy.random import rand
import tensorflow
import tensorflow.keras
from tensorflow.keras.applications.inception_v3 import InceptionV3,preprocess_input
from scipy import *

def preprocess_image(imgc,img_size=224):
    gray = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
    gray[gray > 200] = 0
    gray[gray > 0] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    imgc = cv2.dilate(gray, kernel, iterations=1)
    imgc = cv.resize(imgc, (img_size, img_size),interpolation=cv.INTER_LINEAR)
    return imgc

def traingen_no_leave_users_out(batch,path_a,mode):

    while(1):
        input1 = []
        input2 = []
        outputs = []
        for c in range(0, batch, 1):
            try:
                i = rand() * 39 + 1
                if(i==3):
                    i = rand() * 39 + 1
                k = rand() * 90 + 1
                path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
                img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
                imgc = cv.imread(img_path)
                imgc = preprocess_image(imgc)
                imgc1_1 = np.expand_dims(imgc, axis=2)

                k = rand() * 90 + 1
                path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
                img_path = path_b + '/'+mode[1]+'%dB%d.jpg' %(i, k)
                imgc = cv.imread(img_path)
                imgc = preprocess_image(imgc)
                imgc1_2 = np.expand_dims(imgc, axis=2)

                k = rand() * 90 + 1
                path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
                img_path = path_b + '/'+mode[1]+'%dB%d.jpg'% (i, k)
                imgc = cv.imread(img_path)
                imgc = preprocess_image(imgc)
                imgc1_3 = np.expand_dims(imgc, axis=2)


                #########################################################33
                k = rand() * 90 + 1
                img_path = path_b + '/'+mode[1]+'%dB%d.jpg'% (i, k)
                imgc2 = cv.imread(img_path)
                imgc2 = preprocess_image(imgc2)
                imgc2_1 = np.expand_dims(imgc2, axis=2)

                k = rand() * 90 + 1
                img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
                imgc2 = cv.imread(img_path)
                imgc2 = preprocess_image(imgc2)
                imgc2_2 = np.expand_dims(imgc2, axis=2)

                k = rand() * 90 + 1
                img_path = path_b + '/'+mode[1]+'%dB%d.jpg'% (i, k)
                imgc2 = cv.imread(img_path)
                imgc2 = preprocess_image(imgc2)
                imgc2_3 = np.expand_dims(imgc2, axis=2)


                ################################################################3
                input1.append(preprocess_input(np.concatenate((imgc1_1, imgc1_2, imgc1_3), axis=2) ))
                input2.append(preprocess_input(np.concatenate((imgc2_1, imgc2_2, imgc2_3), axis=2) ))
                outputs.append(1.)
                ###############################################################

                iant=i
                i = rand() * 39 + 1
                if(i==iant or np.int(i)==3):
                    i = rand() * 39 + 1
                k = rand() * 90 + 1
                path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
                img_path = path_b + '/'+mode[1]+'%dB%d.jpg'% (i, k)
                imgc3 = cv.imread(img_path)
                imgc3 = preprocess_image(imgc3)
                imgc3_1 = np.expand_dims(imgc3, axis=2)

                k = rand() * 90 + 1
                path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
                img_path = path_b + '/'+mode[1]+'%dB%d.jpg'% (i, k)
                imgc3 = cv.imread(img_path)
                imgc3 = preprocess_image(imgc3)
                imgc3_2 = np.expand_dims(imgc3, axis=2)

                k = rand() * 90 + 1
                path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
                img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
                imgc3 = cv.imread(img_path)
                imgc3 = preprocess_image(imgc3)
                imgc3_3 = np.expand_dims(imgc3, axis=2)

                input1.append(preprocess_input(np.concatenate((imgc1_1, imgc1_2, imgc1_3), axis=2)))
                input2.append(preprocess_input(np.concatenate((imgc3_1, imgc3_2, imgc3_3), axis=2)))
                outputs.append(0.)
            except:
                print(i)
                print(k)
        input1 = np.array(input1)
        input2 = np.array(input2)
        outputs = np.array(outputs)

        yield [input1, input2], outputs
def valgen_no_leave_users_out(batch,path_a,mode):

    while(1):
        input1 = []
        input2 = []
        outputs = []
        for c in range(0, batch, 1):
            i = rand() * 39 + 1

            k = rand() * 49 + 90
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg'% (i, k)
            imgc = cv.imread(img_path)
            imgc = preprocess_image(imgc)
            imgc1_1 = np.expand_dims(imgc, axis=2)

            k = rand() * 49 + 90
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
            imgc = cv.imread(img_path)
            imgc = preprocess_image(imgc)
            imgc1_2 = np.expand_dims(imgc, axis=2)

            k = rand() * 49 + 90
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
            imgc = cv.imread(img_path)
            imgc = preprocess_image(imgc)
            imgc1_3 = np.expand_dims(imgc, axis=2)
            #########################################################33
            k = rand() * 49 + 90
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
            imgc2 = cv.imread(img_path)
            imgc2 = preprocess_image(imgc2)
            imgc2_1 = np.expand_dims(imgc2, axis=2)

            k = rand() * 49 + 90
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg'% (i, k)
            imgc2 = cv.imread(img_path)
            imgc2 = preprocess_image(imgc2)
            imgc2_2 = np.expand_dims(imgc2, axis=2)

            k = rand() * 49 + 90
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg'% (i, k)
            imgc2 = cv.imread(img_path)
            imgc2 = preprocess_image(imgc2)
            imgc2_3 = np.expand_dims(imgc2, axis=2)
            ################################################################3
            input1.append(preprocess_input(np.concatenate((imgc1_1, imgc1_2, imgc1_3), axis=2) ))
            input2.append(preprocess_input(np.concatenate((imgc2_1, imgc2_2, imgc2_3), axis=2) ))
            outputs.append(1.)
            ###############################################################
            iant=i
            i = rand() * 39 + 1
            if (i == iant or np.int(i) == 3):
                i = rand() * 39 + 1
            k = rand() * 49 + 90
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
            imgc3 = cv.imread(img_path)
            imgc3 = preprocess_image(imgc3)
            imgc3_1 = np.expand_dims(imgc3, axis=2)

            k = rand() * 49 + 90
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
            imgc3 = cv.imread(img_path)
            imgc3 = preprocess_image(imgc3)
            imgc3_2 = np.expand_dims(imgc3, axis=2)

            k = rand() * 49 + 90
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
            imgc3 = cv.imread(img_path)
            imgc3 = preprocess_image(imgc3)
            imgc3_3 = np.expand_dims(imgc3, axis=2)

            input1.append(preprocess_input(np.concatenate((imgc1_1, imgc1_2, imgc1_3), axis=2) ))
            input2.append(preprocess_input(np.concatenate((imgc3_1, imgc3_2, imgc3_3), axis=2) ))
            outputs.append(0.)
        input1 = np.array(input1)
        input2 = np.array(input2)
        outputs = np.array(outputs)

        yield [input1, input2], outputs
def traingen_leave_users_out(batch,path_a,mode):

    while(1):
        input1 = []
        input2 = []
        outputs = []
        for c in range(0, batch, 1):
            i = rand() * 24 + 1
            if(i==3):
                i = rand() * 24 + 1
            k = rand() * 149 + 1
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
            imgc = cv.imread(img_path)
            imgc = preprocess_image(imgc)
            imgc1_1 = np.expand_dims(imgc, axis=2)

            k = rand() * 149 + 1
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg' %(i, k)
            imgc = cv.imread(img_path)
            imgc = preprocess_image(imgc)
            imgc1_2 = np.expand_dims(imgc, axis=2)

            k = rand() * 149 + 1
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg'% (i, k)
            imgc = cv.imread(img_path)
            imgc = preprocess_image(imgc)
            imgc1_3 = np.expand_dims(imgc, axis=2)


            #########################################################33
            k = rand() * 149 + 1
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg'% (i, k)
            imgc2 = cv.imread(img_path)
            imgc2 = preprocess_image(imgc2)
            imgc2_1 = np.expand_dims(imgc2, axis=2)

            k = rand() * 149 + 1
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
            imgc2 = cv.imread(img_path)
            imgc2 = preprocess_image(imgc2)
            imgc2_2 = np.expand_dims(imgc2, axis=2)

            k = rand() * 149 + 1
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg'% (i, k)
            imgc2 = cv.imread(img_path)
            imgc2 = preprocess_image(imgc2)
            imgc2_3 = np.expand_dims(imgc2, axis=2)

            ################################################################3
            input1.append(preprocess_input(np.concatenate((imgc1_1, imgc1_2, imgc1_3), axis=2) ))
            input2.append(preprocess_input(np.concatenate((imgc2_1, imgc2_2, imgc2_3), axis=2) ))
            outputs.append(1.)
            ###############################################################

            iant=i
            i = rand() * 24 + 1
            if(i==iant or i==3):
                i = rand() * 24 + 1
            k = rand() * 149 + 1
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg'% (i, k)
            imgc3 = cv.imread(img_path)
            imgc3 = preprocess_image(imgc3)
            imgc3_1 = np.expand_dims(imgc3, axis=2)

            k = rand() * 149 + 1
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg'% (i, k)
            imgc3 = cv.imread(img_path)
            imgc3 = preprocess_image(imgc3)
            imgc3_2 = np.expand_dims(imgc3, axis=2)

            k = rand() * 149 + 1
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
            imgc3 = cv.imread(img_path)
            imgc3 = preprocess_image(imgc3)
            imgc3_3 = np.expand_dims(imgc3, axis=2)

            input1.append(preprocess_input(np.concatenate((imgc1_1, imgc1_2, imgc1_3), axis=2)))
            input2.append(preprocess_input(np.concatenate((imgc3_1, imgc3_2, imgc3_3), axis=2)))
            outputs.append(0.)
        input1 = np.array(input1)
        input2 = np.array(input2)
        outputs = np.array(outputs)

        yield [input1, input2], outputs
def valgen_leave_users_out(batch,path_a,mode):

    while(1):
        input1 = []
        input2 = []
        outputs = []
        for c in range(0, batch, 1):
            i = rand() * 16 + 24

            k = rand() * 149 + 1
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg'% (i, k)
            imgc = cv.imread(img_path)
            imgc = preprocess_image(imgc)
            imgc1_1 = np.expand_dims(imgc, axis=2)

            k = rand() * 149 + 1
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
            imgc = cv.imread(img_path)
            imgc = preprocess_image(imgc)
            imgc1_2 = np.expand_dims(imgc, axis=2)

            k = rand() * 149 + 1
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
            imgc = cv.imread(img_path)
            imgc = preprocess_image(imgc)
            imgc1_3 = np.expand_dims(imgc, axis=2)
            #########################################################33
            k = rand() * 149 + 1
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
            imgc2 = cv.imread(img_path)
            imgc2 = preprocess_image(imgc2)
            imgc2_1 = np.expand_dims(imgc2, axis=2)

            k = rand() * 149 + 1
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg'% (i, k)
            imgc2 = cv.imread(img_path)
            imgc2 = preprocess_image(imgc2)
            imgc2_2 = np.expand_dims(imgc2, axis=2)

            k = rand() * 149 + 1
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg'% (i, k)
            imgc2 = cv.imread(img_path)
            imgc2 = preprocess_image(imgc2)
            imgc2_3 = np.expand_dims(imgc2, axis=2)
            ################################################################3
            input1.append(preprocess_input(np.concatenate((imgc1_1, imgc1_2, imgc1_3), axis=2) ))
            input2.append(preprocess_input(np.concatenate((imgc2_1, imgc2_2, imgc2_3), axis=2) ))
            outputs.append(1.)
            ###############################################################
            iant=i
            i = rand() * 16 + 24
            if(i==iant):
                i = rand() * 16 + 24
            k = rand() * 149 + 1
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
            imgc3 = cv.imread(img_path)
            imgc3 = preprocess_image(imgc3)
            imgc3_1 = np.expand_dims(imgc3, axis=2)

            k = rand() * 149 + 1
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
            imgc3 = cv.imread(img_path)
            imgc3 = preprocess_image(imgc3)
            imgc3_2 = np.expand_dims(imgc3, axis=2)

            k = rand() * 149 + 1
            path_b = path_a+ mode[0] +'/figuras P%dB' % (i)
            img_path = path_b + '/'+mode[1]+'%dB%d.jpg' % (i, k)
            imgc3 = cv.imread(img_path)
            imgc3 = preprocess_image(imgc3)
            imgc3_3 = np.expand_dims(imgc3, axis=2)

            input1.append(preprocess_input(np.concatenate((imgc1_1, imgc1_2, imgc1_3), axis=2) ))
            input2.append(preprocess_input(np.concatenate((imgc3_1, imgc3_2, imgc3_3), axis=2) ))
            outputs.append(0.)
        input1 = np.array(input1)
        input2 = np.array(input2)
        outputs = np.array(outputs)

        yield [input1, input2], outputs
def create_dataset(path_a,mode):
    input1 = []
    input2 = []
    outputs = []
    for c in range(0, 500, 1):
        i = np.random.rand() * 16 + 24
        k = np.random.rand() * 149 + 1
        path_b = path_a + mode[0] + '/figuras P%dB' % (i)
        img_path = path_b + '/' + mode[1] + '%dB%d.jpg' % (i, k)
        imgc = cv.imread(img_path)
        imgc = preprocess_image(imgc)
        imgc1_1 = np.expand_dims(imgc, axis=2)

        k = np.random.rand() * 149 + 1
        path_b = path_a + mode[0] + '/figuras P%dB' % (i)
        img_path = path_b + '/' + mode[1] + '%dB%d.jpg' % (i, k)
        imgc = cv.imread(img_path)
        imgc = preprocess_image(imgc)
        imgc1_2 = np.expand_dims(imgc, axis=2)

        k = np.random.rand() * 149 + 1
        path_b = path_a + mode[0] + '/figuras P%dB' % (i)
        img_path = path_b + '/' + mode[1] + '%dB%d.jpg' % (i, k)
        imgc = cv.imread(img_path)
        imgc = preprocess_image(imgc)
        imgc1_3 = np.expand_dims(imgc, axis=2)
        #########################################################33
        k = np.random.rand() * 149 + 1
        img_path = path_b + '/' + mode[1] + '%dB%d.jpg' % (i, k)
        imgc2 = cv.imread(img_path)
        imgc2 = preprocess_image(imgc2)
        imgc2_1 = np.expand_dims(imgc2, axis=2)

        k = np.random.rand() * 149 + 1
        img_path = path_b + '/' + mode[1] + '%dB%d.jpg' % (i, k)
        imgc2 = cv.imread(img_path)
        imgc2 = preprocess_image(imgc2)
        imgc2_2 = np.expand_dims(imgc2, axis=2)

        k = np.random.rand() * 149 + 1
        img_path = path_b + '/' + mode[1] + '%dB%d.jpg' % (i, k)
        imgc2 = cv.imread(img_path)
        imgc2 = preprocess_image(imgc2)
        imgc2_3 = np.expand_dims(imgc2, axis=2)
        ################################################################3
        input1.append(preprocess_input(np.concatenate((imgc1_1, imgc1_2, imgc1_3), axis=2)))
        input2.append(preprocess_input(np.concatenate((imgc2_1, imgc2_2, imgc2_3), axis=2)))
        outputs.append(1.)
        ###############################################################
        iant = i
        i = np.random.rand() * 16 + 24
        if (i == iant):
            i = np.random.rand() * 5 + 35
        k = np.random.rand() * 149 + 1
        path_b = path_a + mode[0] + '/figuras P%dB' % (i)
        img_path = path_b + '/' + mode[1] + '%dB%d.jpg' % (i, k)
        imgc3 = cv.imread(img_path)
        imgc3 = preprocess_image(imgc3)
        imgc3_1 = np.expand_dims(imgc3, axis=2)

        k = np.random.rand() * 149 + 1
        path_b = path_a + mode[0] + '/figuras P%dB' % (i)
        img_path = path_b + '/' + mode[1] + '%dB%d.jpg' % (i, k)
        imgc3 = cv.imread(img_path)
        imgc3 = preprocess_image(imgc3)
        imgc3_2 = np.expand_dims(imgc3, axis=2)

        k = np.random.rand() * 149 + 1
        path_b = path_a + mode[0] + '/figuras P%dB' % (i)
        img_path = path_b + '/' + mode[1] + '%dB%d.jpg' % (i, k)
        imgc3 = cv.imread(img_path)
        imgc3 = preprocess_image(imgc3)
        imgc3_3 = np.expand_dims(imgc3, axis=2)

        input1.append(preprocess_input(np.concatenate((imgc1_1, imgc1_2, imgc1_3), axis=2)))
        input2.append(preprocess_input(np.concatenate((imgc3_1, imgc3_2, imgc3_3), axis=2)))
        outputs.append(0.)
    input1 = np.array(input1)
    input2 = np.array(input2)
    outputs = np.array(outputs)

    return [input1, input2], outputs
