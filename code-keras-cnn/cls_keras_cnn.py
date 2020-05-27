import random
import sklearn.neural_network
import numpy
import matplotlib.pyplot
import matplotlib
import math
import h5py
import random
import pickle
import glob
import ntpath 
import joblib
import os
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Activation, Dense, Dropout, LSTM, PReLU, Conv1D, MaxPooling2D, Flatten, Embedding, Reshape,Conv2D
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras import regularizers
from itertools import combinations_with_replacement as cb

sc = StandardScaler()

#$ INPUTS
epochs           = 50
batch_size       = 256
verbose          = 1
shuffle          = True
validation_split = 0.01
dropout          = 0.0 #0.1
reg_rate         = 0.000 #0.005 # regularization rate 
optimizer        = 'adam'
activation_1st   = 'relu'
activation_hid   = 'relu'
activation_lst   = 'relu'
loss             = 'mean_squared_error'
plot_train_cond  = True
plot_test_cond   = True
standard_val     = False
lb               = 0
ub               = +1
filter_num       = [96]
kernel_val       = [5,]
embedding_dims   = 10
disl_pack        = 100
data_format      = 'channels_last' # 'channels_first'

#directories and files
dir_current      = os. getcwd()
dir_parent       = os.path.split(dir_current)[0]
dir_code_keras   = os.path.join(dir_parent,'code-keras-cnn')
dir_code_scikit  = os.path.join(dir_parent,'code-scikit')
dir_data_matlab  = os.path.join(dir_parent,'data-matlab')
dir_data_python  = os.path.join(dir_parent,'data-python')

file_data_python = os.path.join(dir_data_python,'data.npy')
file_ind_python  = os.path.join(dir_data_python,'ind.npy')
file_data_matlab = os.path.join(dir_data_matlab,'data_inou.mat')

############################

def fun_scale(x,lb,ub):
	return ((x-numpy.min(x,axis=0))/(numpy.max(x,axis=0)-numpy.min(x,axis=0)))*(ub-lb) + lb

def fun_cnnreshape(datain):
    col_size = int(datain.shape[1]/3)
    r_val    = datain[:,0:col_size]
    a_val    = datain[:,col_size:2*col_size]
    t_val    = datain[:,2*col_size:]
    #print(a_val.shape)
    #print(r_val.shape)
    #print(t_val.shape)
    ind   = numpy.array(list(cb(list(range(col_size)),2)))
    ind1  = ind
    ind2  = ind[:,[1,0]]
    ind = numpy.concatenate((ind1,ind2),axis=0)
    ind = numpy.unique(ind,axis=0)

    #print(ind.shape)

    azimuth0     = numpy.reshape(a_val[:,ind[:,0]],(datain.shape[0],col_size,col_size,1))
    azimuth1     = numpy.reshape(a_val[:,ind[:,1]],(datain.shape[0],col_size,col_size,1))
    azimuth_all  = numpy.concatenate((azimuth0,azimuth1),axis=3)

    radial0      = numpy.reshape(r_val[:,ind[:,0]],(datain.shape[0],col_size,col_size,1))
    radial1      = numpy.reshape(r_val[:,ind[:,1]],(datain.shape[0],col_size,col_size,1))
    radial_all   = numpy.concatenate((radial0,radial1),axis=3)

    rotation0    = numpy.reshape(t_val[:,ind[:,0]],(datain.shape[0],col_size,col_size,1))
    rotation1    = numpy.reshape(t_val[:,ind[:,1]],(datain.shape[0],col_size,col_size,1))
    rotation_all = numpy.concatenate((rotation0,rotation1),axis=3)  

    datain       = numpy.concatenate((radial_all,azimuth_all,rotation_all),axis=3)

    return datain


class ClsFuns():
    """docstring for Cls_Funs"""

    statement = "thi is in cls"

    def __init__(self,i0,i1,lay_neuron):
        self.i0 = i0
        self.i1 = i1
        self.lay_neuron = lay_neuron

    #i0 = rtt
    #i1 = rrs


    def fun_run(self,datain,dataou,ind_train,ind_test):

        # requires: 
        i0 = self.i0
        i1 = self.i1
        lay_neuron = self.lay_neuron

        #normalization

        # transform values between -1 and +1 without any interfere in rotational information

        disl_num  = int(datain.shape[1]/3)
        r_all     = datain[:,0:disl_num]
        a_all     = datain[:,disl_num:disl_num*2]
        t_all     = datain[:,disl_num*2:disl_num*3]

        #r_tmp     = fun_scale(r_all.reshape((r_all.shape[0]*r_all.shape[1],1)),-1,1)
        #r_scl     = r_tmp.reshape((r_all.shape[0],r_all.shape[1]))
        a_sin     = numpy.sin(a_all)
        t_sin     = numpy.sin(t_all)

        datain    = numpy.concatenate((numpy.log10(r_all),a_all,t_all),axis=1)

        # relax the outputs by dividing values by 10**9
        #max_out   = numpy.abs(dataou).max()
        #dataou    = dataou/max_out


        if standard_val:
            datain = sc.fit_transform(datain)
            dataou = sc.fit_transform(dataou)
            
        else:
            datain = fun_scale(datain,lb,ub)
            dataou = fun_scale(dataou,lb,ub)



        #datain = datain[0:100,:]
        #dataou = datain[0:100,:]
        
        # CNN re-arrangement of datain
        col_size = int(datain.shape[1]/3)
        datain = fun_cnnreshape(datain)
        #print(datain.shape)       
        #print(datain.shape)
        #print(col_size)
        # get the indices 
        ind_train = ind_train[i0-1][i1-1]
        ind_test  = ind_test[i0-1][i1-1]
        
        # get the training data
        datain_train = datain[ind_train,:,:,:]
        dataou_train = dataou[ind_train,:]
        
        # get the testing data
        datain_test  = datain[ind_test,:,:,:]
        dataou_test  = dataou[ind_test,:]

        # rearrange for 4d cnn
        #datain_train = numpy.expand_dims(datain_train, axis = 2)
        #datain_train = numpy.expand_dims(datain_train, axis = 2)

        #dataou_train = numpy.expand_dims(dataou_train, axis = 2)
        #dataou_train = numpy.expand_dims(dataou_train, axis = 2)

        #datain_test  = numpy.expand_dims(datain_test, axis = 2)
        #datain_test  = numpy.expand_dims(datain_test, axis = 2)

        #dataou_test  = numpy.expand_dims(dataou_test, axis = 2) 
        #dataou_test  = numpy.expand_dims(dataou_test, axis = 2)       


        tail_val = ''
        for jj in range(len(lay_neuron)):
            tail_val = tail_val + str(lay_neuron[jj])+ '_'

        tail_val = tail_val[:-1]

        filename = "model_keras_cnn_{}_{}_{}.h5".format(i0,i1,tail_val)
        filename = os.path.join(dir_data_python,filename)

        filename_dir = glob.glob(filename)
        if len(filename_dir) == 0:
            print("Initiate new Sequential network from scratch")

            model = Sequential()

            # first cnn layer
            model.add(Conv2D(filter_num[0], kernel_size=kernel_val[0], strides = 1,activation=activation_1st, data_format=data_format, input_shape=(datain.shape[1],datain.shape[2],datain.shape[3]) ))
            #model.add(Dropout(dropout))
            #model.add(Embedding(max_features,embedding_dims,input_length=datain.shape[1]))
            # second to last cnn layers

            for k0 in range(len(filter_num)-1):
                model.add(Conv2D(filter_num[k0+1], kernel_size=kernel_val[k0+1], strides=1, activation=activation_hid, data_format = data_format))
                #model.add(MaxPooling2D(pool_size=2))
                #model.add(Dropout(dropout))
                #model.add(Flatten())
                #model.add(Reshape((model.output_shape[1],1),input_shape=(model.output_shape[1],)))
            #model.add(MaxPooling1D(pool_size=2))
            
            model.add(Flatten())
            
            # dense layers
            for l0 in lay_neuron:
                print(l0)
                model.add(Dense(l0, activation=activation_hid, kernel_regularizer=regularizers.l2(reg_rate)))
                #model.add(Dense(l0, kernel_regularizer=regularizers.l2(reg_rate)))
                #model.add(PReLU(alpha_initializer="zeros",alpha_regularizer=None,alpha_constraint=None,shared_axes=None,))
                model.add(Dropout(dropout))

            # last layer

            model.add(Dense(dataou_train.shape[1], activation=activation_lst))
            #sgd = optimizers.SGD(lr=0.5, decay=1e-9, momentum=0.9, nesterov=True)
            model.compile(loss=loss, optimizer=optimizer)
        else:
            print("Initiate Sequential transfer learning")
            model = load_model(filename)

        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.002,
        #                      patience=5, min_lr=0.000000000000000001)
        
        # train

        print("-------------------------------")
        print("TR-LEARN-KERAS-CNN | rrt {} | rrs {} | Net {}".format(i0,i1,tail_val))
        print("-------------------------------")

        model.summary()

        history       = model.fit(datain_train, dataou_train, epochs=epochs, batch_size=batch_size, verbose=verbose, shuffle=shuffle, validation_split=validation_split) #callbacks=[reduce_lr]
        # save the model to disk
        #pickle.dump(my_neural_network, open(filename, 'wb'))
        model.save(filename)

        loss_vals     = history.history['loss'] 
        val_loss_vals = history.history['val_loss'] 
        
        
        filename = "model_keras_cnn_loss_{}_{}_{}.txt".format(i0,i1,tail_val)
        filename = os.path.join(dir_data_python,filename)
        
        try:
        	f = open(filename,'x')
        except:
        	f = open(filename,'w')

        f.write("{}".format(loss_vals))
        f.close()
        
        filename = "model_keras_cnn_val_loss_{}_{}_{}.txt".format(i0,i1,tail_val)
        filename = os.path.join(dir_data_python,filename)
        try:
            f = open(filename,'x')
        except:
            f = open(filename,'w')

        f.write("{}".format(val_loss_vals))
        f.close()

        print("-------------------------------")
        print("TE-LEARN-KERAS-CNN | rrt {} | rrs {} | Net {}".format(i0,i1,tail_val))
        print("-------------------------------")

        # estimate train and test data
        dataes_train = model.predict(datain_train)
        dataes_test  = model.predict(datain_test)


        trou = numpy.reshape(dataou_train,(dataou_train.shape[0]*dataou_train.shape[1],1))
        tres = numpy.reshape(dataes_train,(dataes_train.shape[0]*dataes_train.shape[1],1))
        teou = numpy.reshape(dataou_test ,(dataou_test.shape[0] *dataou_test.shape[1],1)) 
        tees = numpy.reshape(dataes_test ,(dataes_test.shape[0] *dataes_test.shape[1],1))
 
        if plot_train_cond:
            print("TR-PLOT-KERAS-CNN  | rrt {} | rrs {} | Net {}".format(i0,i1,tail_val))
    
            matplotlib.pyplot.figure(figsize=[10,10])
            matplotlib.rc('xtick', labelsize=20)
            matplotlib.rc('ytick', labelsize=20) 
            matplotlib.rc('font',family='Times New Roman')
            matplotlib.pyplot.plot(trou, tres,'.',markersize=1)
            if standard_val:
                lb_val = -2.5 #-10**10
                ub_val = +2.5 #+10**10
            else:
            	lb_val = lb #-10**10
            	ub_val = ub #+10**10

            matplotlib.pyplot.plot([lb_val,ub_val],[lb_val,ub_val],'-g')
            matplotlib.pyplot.xlabel('Real', fontsize=20, fontname='Times New Roman')
            matplotlib.pyplot.ylabel('Estimated', fontsize=20, fontname='Times New Roman')
            matplotlib.pyplot.title('Train | RTT = {} | RRS = {} | Net_{}'.format(i0,i1,tail_val), fontsize=20, fontname='Times New Roman')
            
            filename = 'tr_keras_cnn_{}_{}_{}.png'.format(i0,i1,tail_val)
            filename = os.path.join(dir_data_python,filename)
            matplotlib.pyplot.savefig(filename, dpi=300)  
        
        if plot_test_cond:
            print("TE-PLOT-KERAS-CNN  | rrt {} | rrs {} | Net {}".format(i0,i1,tail_val))
            matplotlib.pyplot.figure(figsize=[10,10])
            matplotlib.rc('xtick', labelsize=20)
            matplotlib.rc('ytick', labelsize=20) 
            matplotlib.rc('font',family='Times New Roman')
            matplotlib.pyplot.plot(teou, tees,'.',markersize=1)
            if standard_val:
                lb_val = -2.5 #-10**10
                ub_val = +2.5 #+10**10
            else:
                lb_val = lb #-10**10
                ub_val = ub #+10**10

            matplotlib.pyplot.plot([lb_val,ub_val],[lb_val,ub_val],'-g')
            matplotlib.pyplot.xlabel('Real', fontsize=20, fontname='Times New Roman')
            matplotlib.pyplot.ylabel('Estimated', fontsize=20, fontname='Times New Roman')
            matplotlib.pyplot.title('Test | RTT = {} | RRS = {} | Net_{}'.format(i0,i1,tail_val), fontsize=20, fontname='Times New Roman')
            
            filename = 'te_keras_cnn_{}_{}_{}.png'.format(i0,i1,tail_val)
            filename = os.path.join(dir_data_python,filename)
            matplotlib.pyplot.savefig(filename, dpi=300)  
        
    def fun_losscurve(self):
        i0 = self.i0
        i1 = self.i1
        lay_neuron = self.lay_neuron

        tail_val = ''
        for jj in range(len(lay_neuron)):
            tail_val = tail_val + str(lay_neuron[jj])+ '_'

        tail_val = tail_val[:-1]

        filename1     = "model_keras_cnn_loss_{}_{}_{}.txt".format(i0,i1,tail_val)
        filename1     = os.path.join(dir_data_python,filename1)
        
        filename2     = "model_keras_cnn_val_loss_{}_{}_{}.txt".format(i0,i1,tail_val)
        filename2     = os.path.join(dir_data_python,filename2)
        
        f1            = open(filename1,'r')
        f2            = open(filename2,'r')
        
        f1_val = eval(f1.read())
        f2_val = eval(f2.read())

        a1 = range(1,len(f1_val)+1)
        a1 = numpy.array(a1)
        b1 = numpy.log10(numpy.array(f1_val))

        a2 = range(1,len(f2_val)+1)
        a2 = numpy.array(a2)
        b2 = numpy.log10(numpy.array(f2_val))

        f1.close()
        f2.close()

        print("LS-PLOT-KERAS-CNN  | rrt {} | rrs {} | Net {}".format(i0,i1,tail_val))
        
        matplotlib.pyplot.figure()
        matplotlib.pyplot.figure(figsize=[10,10])
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20) 
        matplotlib.rc('font',family='Times New Roman')
        matplotlib.pyplot.plot(a1, b1,'-b',markersize=1)
        matplotlib.pyplot.plot(a2, b2,'-r',markersize=1)
        matplotlib.pyplot.legend(['loss', 'val_loss'])
        matplotlib.pyplot.xlabel('Iterations', fontsize=20, fontname='Times New Roman')
        matplotlib.pyplot.ylabel('Natural Logarithm of Mean Squared Error (MSE)', fontsize=20, fontname='Times New Roman')
        matplotlib.pyplot.title('Train_Progress | RTT = {} | RRS = {} | Net_{}'.format(i0,i1,tail_val), fontsize=20, fontname='Times New Roman')
        
        filename = 'ls_keras_cnn_{}_{}_{}.png'.format(i0,i1,tail_val)
        filename = os.path.join(dir_data_python,filename)
        matplotlib.pyplot.savefig(filename, dpi=300) 
