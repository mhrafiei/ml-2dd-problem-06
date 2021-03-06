import random
import sklearn.neural_network
import numpy
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
from keras.layers import Activation, Dense, Dropout, LSTM
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers
from sklearn.preprocessing import StandardScaler
from keras import regularizers

sc = StandardScaler()

#$ INPUTS
epochs           = 20000
batch_size       = 2024
verbose          = 1
shuffle          = True
validation_split = 0.01
optimizer        = 'sgd'
activation       = 'relu' # logistic
learning_init    = 0.02
early_stopping   = False
reg_rate         = 0.05 # regularization rate 
plot_train_cond  = True
plot_test_cond   = True
standard_val     = False
lb               = 0
ub               = +1


#directories and files
dir_current      = os. getcwd()
dir_parent       = os.path.split(dir_current)[0]
dir_code_keras   = os.path.join(dir_parent,'code-keras')
dir_code_scikit  = os.path.join(dir_parent,'code-scikit')
dir_data_matlab  = os.path.join(dir_parent,'data-matlab')
dir_data_python  = os.path.join(dir_parent,'data-python')

file_data_python = os.path.join(dir_data_python,'data.npy')
file_ind_python  = os.path.join(dir_data_python,'ind.npy')
file_data_matlab = os.path.join(dir_data_matlab,'data_inou.mat')

############################

def fun_scale(x,lb,ub):
    return ((x-numpy.min(x,axis=0))/(numpy.max(x,axis=0)-numpy.min(x,axis=0)))*(ub-lb) + lb

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


        # get the indices 
        ind_train = ind_train[i0-1][i1-1]
        ind_test  = ind_test[i0-1][i1-1]
        
        # get the training data
        datain_train = datain[ind_train]
        dataou_train = dataou[ind_train]
        
        # get the testing data
        datain_test  = datain[ind_test]
        dataou_test  = dataou[ind_test]

        tail_val = ''
        for jj in range(len(lay_neuron)):
            tail_val = tail_val + str(lay_neuron[jj])+ '_'

        tail_val = tail_val[:-1]

        filename = "model_scikit_{}_{}_{}.sav".format(i0,i1,tail_val)
        filename = os.path.join(dir_data_python,filename)
        filename_dir = glob.glob(filename)
        print(filename_dir)

        if len(filename_dir) == 0:
            print("Initiate new network from scratch")
            model = sklearn.neural_network.MLPRegressor(hidden_layer_sizes   = tuple(lay_neuron),  \
                                                        max_iter             = epochs, \
                                                        verbose              = verbose,\
                                                        batch_size           = batch_size,\
                                                        shuffle              = shuffle, \
                                                        solver               = optimizer,\
                                                        n_iter_no_change     = 10000,\
                                                        activation           = activation,\
                                                        learning_rate        = 'adaptive',\
                                                        early_stopping       = early_stopping, \
                                                        alpha                = reg_rate, \
                                                        tol                  = 0.00000000000000001,\
                                                        validation_fraction  = validation_split,\
                                                        learning_rate_init   = learning_init)
        else:
            # transfer learning 
            print("Initiate transfer learning")
            model = joblib.load(filename)
            model.warm_start = True

        #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.002,
        #                      patience=5, min_lr=0.000000000000000001)
        
        # train
        print("-------------------------------")
        print("TR-LEARN-SCIKIT | rrt {} | rrs {} | Net {}".format(i0,i1,tail_val))
        print("-------------------------------")

        model.fit(X=datain_train, y=dataou_train)

        print("-------------------------------")
        print("TE-LEARN-SCIKIT | rrt {} | rrs {} | Net {}".format(i0,i1,tail_val))
        print("-------------------------------")

        # estimate train and test data
        dataes_train = model.predict(datain_train)
        dataes_test  = model.predict(datain_test)

        # save the model to disk
        #pickle.dump(model, open(filename, 'wb'))
        joblib.dump(model, filename)


        trou = numpy.reshape(dataou_train,(dataou_train.shape[0]*dataou_train.shape[1],1))
        tres = numpy.reshape(dataes_train,(dataes_train.shape[0]*dataes_train.shape[1],1))
        teou = numpy.reshape(dataou_test ,(dataou_test.shape[0] *dataou_test.shape[1],1)) 
        tees = numpy.reshape(dataes_test ,(dataes_test.shape[0] *dataes_test.shape[1],1))


 
        if plot_train_cond:
            print("TR-PLOT-SCIKIT  | rrt {} | rrs {} | Net {}".format(i0,i1,tail_val))
    
            matplotlib.pyplot.figure(figsize=[10,10])
            matplotlib.rc('xtick', labelsize=20)
            matplotlib.rc('ytick', labelsize=20) 
            matplotlib.rc('font',family='Times New Roman')
            matplotlib.pyplot.plot(trou, tres,'.',markersize=1)
            if standard_val:
                lb_val = -2.5
                ub_val = +2.5
            else:
                lb_val = lb
                ub_val = ub

            matplotlib.pyplot.plot([lb_val,ub_val],[lb_val,ub_val],'-g')
            matplotlib.pyplot.xlabel('Real', fontsize=20, fontname='Times New Roman')
            matplotlib.pyplot.ylabel('Estimated', fontsize=20, fontname='Times New Roman')
            matplotlib.pyplot.title('Train | RTT = {} | RRS = {} | Net_{}'.format(i0,i1,tail_val), fontsize=20, fontname='Times New Roman')

            filename = 'tr_scikit_{}_{}_{}.png'.format(i0,i1,tail_val)
            filename = os.path.join(dir_data_python,filename)
            matplotlib.pyplot.savefig(filename, dpi=300)  
        
        if plot_test_cond:
            print("TE-PLOT-SCIKIT  | rrt {} | rrs {} | Net {}".format(i0,i1,tail_val))
        
            matplotlib.pyplot.figure(figsize=[10,10])
            matplotlib.rc('xtick', labelsize=20)
            matplotlib.rc('ytick', labelsize=20) 
            matplotlib.rc('font',family='Times New Roman')
            matplotlib.pyplot.plot(teou, tees,'.',markersize=1)
            if standard_val:
                lb_val = -2.5
                ub_val = +2.5
            else:
                lb_val = lb
                ub_val = ub

            matplotlib.pyplot.plot([lb_val,ub_val],[lb_val,ub_val],'-g')
            matplotlib.pyplot.xlabel('Real', fontsize=20, fontname='Times New Roman')
            matplotlib.pyplot.ylabel('Estimated', fontsize=20, fontname='Times New Roman')
            matplotlib.pyplot.title('Test | RTT = {} | RRS = {} | Net_{}'.format(i0,i1,tail_val), fontsize=20, fontname='Times New Roman')

            filename = 'te_scikit_{}_{}_{}.png'.format(i0,i1,tail_val)
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

        filename = "model_scikit_{}_{}_{}.sav".format(i0,i1,tail_val)
        filename = os.path.join(dir_data_python,filename)
        model    = joblib.load(filename)
        
        a1 = range(1,len(model.loss_curve_)+1)
        a1 = numpy.array(a1)
        b1 = numpy.log10(model.loss_curve_)

        if early_stopping :
            a2 = range(1,len(model.validation_scores_)+1)
            a2 = numpy.array(a2)
            b2 = numpy.array(model.validation_scores_)

        print("LS-PLOT-SCIKIT  | rrt {} | rrs {} | Net {}".format(i0,i1,tail_val))
        
        matplotlib.pyplot.figure(figsize=[10,10])
        matplotlib.rc('xtick', labelsize=20)
        matplotlib.rc('ytick', labelsize=20) 
        matplotlib.rc('font',family='Times New Roman')
        matplotlib.pyplot.plot(a1, b1,'-b',markersize=1)
        matplotlib.pyplot.legend(['loss', 'val_loss'])
        matplotlib.pyplot.xlabel('Iterations', fontsize=20, fontname='Times New Roman')
        matplotlib.pyplot.ylabel('Natural Logarithm of Mean Squared Error (MSE)', fontsize=20, fontname='Times New Roman')
        matplotlib.pyplot.title('Train_Progress | RTT = {} | RRS = {} | Net_{}'.format(i0,i1,tail_val), fontsize=20, fontname='Times New Roman')
        
        filename = 'ls_scikit_{}_{}_{}.png'.format(i0,i1,tail_val)
        filename = os.path.join(dir_data_python,filename)
        matplotlib.pyplot.savefig(filename, dpi=300) 

        if early_stopping :
            matplotlib.pyplot.figure(figsize=[10,10])
            matplotlib.rc('xtick', labelsize=20)
            matplotlib.rc('ytick', labelsize=20) 
            matplotlib.rc('font',family='Times New Roman')
            matplotlib.pyplot.plot(a2, b2,'-r',markersize=1)
            matplotlib.pyplot.legend(['loss', 'val_loss'])
            matplotlib.pyplot.xlabel('Iterations', fontsize=20, fontname='Times New Roman')
            matplotlib.pyplot.ylabel('Natural Logarithm of Mean Squared Error (MSE)', fontsize=20, fontname='Times New Roman')
            matplotlib.pyplot.title('Validation_Scores | RTT = {} | RRS = {} | Net_{}'.format(i0,i1,tail_val), fontsize=20, fontname='Times New Roman')
            
            filename = 'vl_scikit_{}_{}_{}.png'.format(i0,i1,tail_val)
            filename = os.path.join(dir_data_python,filename)
            matplotlib.pyplot.savefig(filename, dpi=300) 
