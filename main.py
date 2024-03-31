import numpy as np
import tensorflow
import matplotlib.pyplot as plt
from scipy import *
from utils import *
from model_scoring import *
######################################
            # PARAMS #
#######################################
pathh=r'data/'
size = 224
mode=3
batch = 10
options=[['crudo','PQPPG'],['filtradas','PQF001PPG'],['filtradas2','PQF05PPG'],['normalizadas','PQF05PPG'],['solapadas','PQF05PPG']]
train_test=False
#######################################


biomodel=create_model(size)


######################################
            # TRAIN MODE #
#######################################
if(train_test==False):
    check = tensorflow.keras.callbacks.ModelCheckpoint('PQ_' + options[mode][0] + '.h5', monitor='val_accuracy',verbose=1, save_best_only=True, save_weights_only=True,mode='auto', period=1)
    biomodel.fit_generator(traingen_leave_users_out(batch,pathh,options[mode]),steps_per_epoch=500,validation_data=valgen_leave_users_out(batch,pathh,options[mode]),validation_steps=250,epochs=4000,verbose=1, callbacks=[check])


######################################
            # TEST MODE #
#######################################
elif(train_test==True):
    biomodel.load_weights('models/PQ_'+options[mode][0]+'.h5')
    [inputs,outputs]=create_dataset(pathh,options[mode])
    output=biomodel.predict(inputs,verbose=1)
    FPR=[]
    FNR=[]
    TPR=[]
    gtTn=[]
    gtFn=[]
    threshold=[]
    aux=output.copy()
    for k in range(0,100,1):
        output=aux.copy()
        thres=k / 100.
        threshold.append(thres)
        output[output>=thres]=1.
        output[output < thres] = 0.
        TPR.append(np.array(np.where(np.equal(output[:,0], outputs)==1)).shape[1])
        FPR.append(np.array(np.where(np.greater(output[:,0], outputs)==1)).shape[1])
        FNR.append(np.array(np.where(np.less(output[:,0], outputs)==1)).shape[1])
        gtTn.append(np.array(np.where( outputs == 1)).shape[1])
        gtFn.append(np.array(np.where( outputs == 0)).shape[1])
    TPR=np.array(TPR)
    FPR=np.array(FPR)
    FNR=np.array(FNR)
    gtFn=np.array(gtFn)
    gtTn = np.array(gtTn)
    threshold=np.array(threshold)


    plt.figure(1)
    plt.plot(threshold,FNR/gtFn)
    plt.plot(threshold,FPR/gtTn)
    plt.xlabel('Threshold')
    plt.ylabel('Error')


    plt.figure(2)
    precision=TPR/(TPR+FPR)
    recall=TPR/(TPR+FNR)
    plt.plot(recall,precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')


    plt.figure(3)
    precision=TPR/(TPR+FPR)
    recall=TPR/(TPR+FNR)
    plt.plot(threshold,(2*recall*precision)/(precision+recall))
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score')


    plt.figure(4)
    plt.plot(FPR,TPR)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
