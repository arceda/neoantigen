from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import math, os, time, sys, re, datetime
from datetime import timedelta
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy import stats
#import matplotlig.pylot as plt

print("Tensorflow version " + tf.__version__)

#testset = 'A0202'
#INPUTS: python DLMHC.py 0 A0202
testset = sys.argv[2] # e.g. takes A0202 as input allele name
print("Test set is ", testset)
#
runindx = sys.argv[1] #0 for new run, 1 for follow up runs to use the previous best weights IF WISHED IN THE FUTURE.
runindx = int(runindx)


#this gives one hot encoded values
import new_read_data  #our custom functions script to get one hot encoded peptide sequence datasets
transform_start = time.time()
#数据准备
data=new_read_data.getdata_onehot(datafile=testset)
transform_end = time.time()
transform_time = transform_end - transform_start

#optional: shuffle the training data and its labels
shuffle_ = np.arange(len(data['Y_train']))
np.random.shuffle(shuffle_)
data['Y_train']=data['Y_train'][shuffle_]
data['X_train']=data['X_train'][shuffle_]

print("X_Train size ", data['X_train'].shape)
print("Y_Train size ", data['Y_train'].shape)
print("Train data value=1 ", np.sum(data['Y_train']==1))
print("X_Test size " , data['X_test'].shape)
print("Y_Test size " , data['Y_test'].shape)
print("Test data value=1 ", np.sum(data['Y_test']==1))
print(" val1 data size ", len(data['Y_val1']))
print("val1 data value=1 ", np.sum(data['Y_val1']==1))
print(" val2 data size ", len(data['Y_val2']))
print("val2 data value=1 ", np.sum(data['Y_val2']==1))
print(" val3 data size ", len(data['Y_val3']))
print("val3 data value=1 ", np.sum(data['Y_val3']==1))

# Training and optimisation Parameters
lr_max = 0.003
lr_min = 0.001
lr_ultimate_min= 0.0001
# learning rate decay
decay_speed = 40.0 # 0.003-0.0001-2000=>0.9826 done in 5000 iterations
learning_rate = tf.placeholder(tf.float32, shape=[])
batch_size = 64 #was 40 for most
epochs = 100
###
save_dir = 'checkpoints/' + testset + '/'
numofinput_channels = 1 # 1 data input per feature
numofclasses=2  # data labels are binary.
#for dropout probability
prob_ = tf.placeholder( dtype=tf.float32, shape=() )
keep_prob_rate=0.4 #0.4
nnodes_f1= 100

#
input_height = data['X_train'][0].shape[0] #11, depends onpeptide length
input_width = data['X_train'][0].shape[1]  #21 , comes from size of unique peptide sequence letters
# Tensor graph input is 4-D: [Batch Size, Height, Width, Channel]
X = tf.placeholder(tf.float32, shape=[None, input_height, input_width])
# dynamically reshape the input
X_shaped = tf.reshape(X, [-1, input_height, input_width, 1])
#None: 'number of' (#) input is dynamic, not decied yet. numofinput_channels is 1 input channel
# now declare the output data placeholder - 2 digits
y_outputholder = tf.placeholder(tf.float32, shape=[None, numofclasses])
#numofclasses is 2: the outout is binary; binding or not binding.  #None: #input is dynamic. Will be same as #input.

# Make some wrappers for simplicity
def conv2d_layer(input_data, num_input_channels, num_filters, filter_shape,strides_, name):   #(x, W, b, strides_=[2,2]):
    # Conv2D wrapper, with bias and relu activation
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]
    # initialise weights and bias for the filter
    weights = tf.Variable(tf.random.truncated_normal(conv_filt_shape, stddev=0.01), name=name+'_W') #, seed=myseed[2]
    bias = tf.Variable(tf.random.truncated_normal([num_filters]), name=name+'_b')  #, seed=myseed[3],
    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input=input_data, filter=weights, strides=[1, strides_[0], strides_[1], 1], padding='SAME')  #see below, Exp1, for the explanations of conv2d if needed.
    # add the bias
    out_layer = tf.nn.bias_add(out_layer, bias)
    # apply a ReLU non-linear activation (or change as you like)
    return tf.nn.leaky_relu(features=out_layer,alpha=0.2)



######### Design the DL model   #########
# Tensor inputs for 4-D: [Batch Size, Height, Width, Channel]
def DL_model(inputData):
    # add a custom 2D Convolution Layer
    #output1a has different filter shape
    nfilters1a=128 #can be 512
    output1a= conv2d_layer(input_data=inputData , num_input_channels=numofinput_channels,
                num_filters=nfilters1a, filter_shape=[2,2],strides_=[1,1], name='CNN2d_1_a')
    print("CNN2d_1_a output shape: ", output1a.get_shape())
    nfilters2a=128
    output1a= conv2d_layer(input_data=output1a , num_input_channels=nfilters1a,
                num_filters=nfilters2a, filter_shape=[2,2],strides_=[2,2], name='CNN2d_2_a')
    print("CNN2d_2_a output shape: ", output1a.get_shape())
    #**
    nfilters3a=256
    output1a= conv2d_layer(input_data=output1a , num_input_channels=nfilters2a,
                num_filters=nfilters3a, filter_shape=[2,2],strides_=[2,2], name='CNN2d_3_a')
    print("CNN2d_3_a output shape: ", output1a.get_shape())
    nfilters4a=256
    output1a= conv2d_layer(input_data=output1a , num_input_channels=nfilters3a,
                num_filters=nfilters4a, filter_shape=[2,2],strides_=[2,2], name='CNN2d_4_a')
    print("CNN2d_4_a output shape: ", output1a.get_shape())
    nfilters5a=256
    output1a= conv2d_layer(input_data=output1a , num_input_channels=nfilters4a,
                num_filters=nfilters5a, filter_shape=[2,2],strides_=[2,2], name='CNN2d_5_a')
    print("CNN2d_5_a output shape: ", output1a.get_shape())
    nfilters6a=512
    output1a= conv2d_layer(input_data=output1a , num_input_channels=nfilters5a,
                num_filters=nfilters6a, filter_shape=[1,2],strides_=[1,2], name='CNN2d_6_a')
    print("CNN2d_6_a output shape: ", output1a.get_shape())
    nfilters7a=512
    output1a= conv2d_layer(input_data=output1a , num_input_channels=nfilters6a,
                num_filters=nfilters7a, filter_shape=[1,1],strides_=[1,1], name='CNN2d_7_a')
    print("CNN2d_7_a output shape: ", output1a.get_shape())
    nfilters8a=256
    output1a= conv2d_layer(input_data=output1a , num_input_channels=nfilters7a,
                num_filters=nfilters8a, filter_shape=[1,1],strides_=[1,1], name='CNN2d_8_a')
    print("CNN2d_8_a output shape: ", output1a.get_shape())
    '''
    nfilters9a=256
    output1a= conv2d_layer(input_data=output1a , num_input_channels=nfilters8a,
                num_filters=nfilters9a, filter_shape=[1,1],strides_=[1,1], name='CNN2d_9_a')
    print("CNN2d_9_a output shape: ", output1a.get_shape())
    '''
    #**
    lastfiltersize_a=nfilters8a
    out1a_h = output1a.get_shape().as_list()[1]; out1a_w  = output1a.get_shape().as_list()[2]
    output1a_reshape = tf.reshape(output1a, [-1, out1a_h*out1a_w*lastfiltersize_a])
    print("CNN2d_1_a output reshaped: ", output1a_reshape.get_shape())
    #######layer B: output1b has different filter shape
    nfilters1b=128
    output1b= conv2d_layer(input_data=inputData , num_input_channels=numofinput_channels,
                num_filters=nfilters1b, filter_shape=[1,2],strides_=[1,1], name='CNN2d_1_b')
    print("CNN2d_1_b output shape: ", output1b.get_shape())
    nfilters2b=128
    output1b= conv2d_layer(input_data=output1b , num_input_channels=nfilters1b,
                num_filters=nfilters2b, filter_shape=[2,2],strides_=[2,2], name='CNN2d_2_b')
    print("CNN2d_2_b output shape: ", output1b.get_shape())
    #**
    nfilters3b=256
    output1b= conv2d_layer(input_data=output1b , num_input_channels=nfilters2b,
                num_filters=nfilters3b, filter_shape=[2,2],strides_=[2,2], name='CNN2d_3_b')
    print("CNN2d_3_b output shape: ", output1b.get_shape())
    nfilters4b=256
    output1b= conv2d_layer(input_data=output1b , num_input_channels=nfilters3b,
                num_filters=nfilters4b, filter_shape=[2,2],strides_=[2,2], name='CNN2d_4_b')
    print("CNN2d_4_b output shape: ", output1b.get_shape())
    nfilters5b=256
    output1b= conv2d_layer(input_data=output1b , num_input_channels=nfilters4b,
                num_filters=nfilters5b, filter_shape=[2,2],strides_=[2,2], name='CNN2d_5_b')
    print("CNN2d_5_b output shape: ", output1b.get_shape())
    nfilters6b=512
    output1b= conv2d_layer(input_data=output1b , num_input_channels=nfilters5b,
                num_filters=nfilters6b, filter_shape=[1,2],strides_=[1,2], name='CNN2d_6_b')
    print("CNN2d_6_b output shape: ", output1b.get_shape())
    nfilters7b=512
    output1b= conv2d_layer(input_data=output1b , num_input_channels=nfilters6b,
                num_filters=nfilters7b, filter_shape=[1,1],strides_=[1,1], name='CNN2d_7_b')
    print("CNN2d_7_b output shape: ", output1b.get_shape())
    nfilters8b=256
    output1b= conv2d_layer(input_data=output1b , num_input_channels=nfilters7b,
                num_filters=nfilters8b, filter_shape=[1,1],strides_=[1,1], name='CNN2d_8_b')
    print("CNN2d_8_b output shape: ", output1b.get_shape())
    '''
    nfilters9b=128
    output1b= conv2d_layer(input_data=output1b , num_input_channels=nfilters8b,
                num_filters=nfilters9b, filter_shape=[1,1],strides_=[1,1], name='CNN2d_9_b')
    print("CNN2d_9_b output shape: ", output1b.get_shape())
    '''
    #**
    lastfiltersize_b=nfilters8b
    out1b_h = output1b.get_shape().as_list()[1]; out1b_w  = output1b.get_shape().as_list()[2]
    output1b_reshape = tf.reshape(output1b, [-1, out1b_h*out1b_w*lastfiltersize_b])
    print("CNN2d_1_b output reshaped: ", output1b_reshape.get_shape())
    ########
    #######layer C: output1c has different filter shape
    nfilters1c=128
    output1c= conv2d_layer(input_data=inputData , num_input_channels=numofinput_channels,
                num_filters=nfilters1c, filter_shape=[2,1],strides_=[1,1], name='CNN2d_1_c')
    print("CNN2d_1_c output shape: ", output1c.get_shape())
    nfilters2c=128
    output1c= conv2d_layer(input_data=output1c , num_input_channels=nfilters1c,
                num_filters=nfilters2c, filter_shape=[2,2],strides_=[2,2], name='CNN2d_2_c')
    print("CNN2d_2_c output shape: ", output1c.get_shape())
    #**
    nfilters3c=256
    output1c= conv2d_layer(input_data=output1c , num_input_channels=nfilters2c,
                num_filters=nfilters3c, filter_shape=[2,2],strides_=[2,2], name='CNN2d_3_c')
    print("CNN2d_3_c output shape: ", output1c.get_shape())
    nfilters4c=256
    output1c= conv2d_layer(input_data=output1c , num_input_channels=nfilters3c,
                num_filters=nfilters4c, filter_shape=[2,2],strides_=[2,2], name='CNN2d_4_c')
    print("CNN2d_4_c output shape: ", output1c.get_shape())
    nfilters5c=256
    output1c= conv2d_layer(input_data=output1c , num_input_channels=nfilters4c,
                num_filters=nfilters5c, filter_shape=[2,2],strides_=[2,2], name='CNN2d_5_c')
    print("CNN2d_5_c output shape: ", output1c.get_shape())
    nfilters6c=512
    output1c= conv2d_layer(input_data=output1c , num_input_channels=nfilters5c,
                num_filters=nfilters6c, filter_shape=[1,2],strides_=[1,2], name='CNN2d_6_c')
    print("CNN2d_6_c output shape: ", output1c.get_shape())
    nfilters7c=512
    output1c= conv2d_layer(input_data=output1c , num_input_channels=nfilters6c,
                num_filters=nfilters7c, filter_shape=[1,1],strides_=[1,1], name='CNN2d_7_c')
    print("CNN2d_7_c output shape: ", output1c.get_shape())
    nfilters8c=256
    output1c= conv2d_layer(input_data=output1c , num_input_channels=nfilters7c,
                num_filters=nfilters8c, filter_shape=[1,1],strides_=[1,1], name='CNN2d_8_c')
    print("CNN2d_8_c output shape: ", output1c.get_shape())
    '''
    nfilters9c=256
    output1c= conv2d_layer(input_data=output1c , num_input_channels=nfilters8c,
                num_filters=nfilters9c, filter_shape=[1,1],strides_=[1,1], name='CNN2d_9_c')
    print("CNN2d_9_c output shape: ", output1c.get_shape())
    '''
    #**
    lastfiltersize_c=nfilters8c
    out1c_h = output1c.get_shape().as_list()[1]; out1c_w  = output1c.get_shape().as_list()[2]
    output1c_reshape = tf.reshape(output1c, [-1, out1c_h*out1c_w*lastfiltersize_c])
    print("CNN2d_1_c output reshaped: ", output1c_reshape.get_shape())
    ########

    #COMBINE THREE PARALLEL CONV CONNECTIONS OF DIFFERENT FILTER SIZES HERE
    flattened = tf.concat([output1a_reshape, output1b_reshape,output1c_reshape], axis=1) #combined the two praralel filters
    out_height = flattened.get_shape().as_list()[1]
    # Fully connected layer
    # Reshape conv2 output1 to fit fully connected layer input
    #flattened = tf.reshape(output1, [-1, out_height * out_width * nfilters2])
    print("flattened layer output shape: ", flattened.get_shape())
    # setup some weights and bias values for this layer, then activate with ReLU
    W_f1 = tf.Variable(tf.random.truncated_normal([out_height, nnodes_f1], stddev=0.01),  name='W_f1') #, seed=myseed[4]
    B_f1 = tf.Variable(tf.random.truncated_normal([nnodes_f1], stddev=0.01),  name='B_f1') #, seed=myseed[5]
    #
    dense_layer1 = tf.add(tf.matmul(flattened, W_f1), B_f1)
    dense_layer1 = tf.nn.leaky_relu(features=dense_layer1, alpha=0.2)
    print("dense_layer1 output shape: ", dense_layer1.get_shape())
    #
    # Apply Dropout
    dense_layer1 = tf.nn.dropout(x=dense_layer1, keep_prob=prob_) #Dropout process

    # another layer for the final output
    wd2 = tf.Variable(tf.random.truncated_normal([nnodes_f1, numofclasses], stddev=0.01), name='wd2') #, seed=myseed[6]
    bd2 = tf.Variable(tf.random.truncated_normal([numofclasses], stddev=0.01),  name='bd2') #, seed=myseed[7]
    final_layer = tf.add(tf.matmul(dense_layer1, wd2), bd2) #class prediction
    print("final_layer output shape: ", final_layer.get_shape())
    return final_layer

########## END of  DL model design #########

# Construct model Z2=logits shape(m,2)
logits = DL_model(inputData=X_shaped)
prediction = tf.nn.softmax(logits)

#数据转换 0-[0,1],1-[1,0]保证shape=(m,2)
Y_train_labels = new_read_data.binary2onehot(data['Y_train']) # binary output converted into two classes
Y_test_labels = new_read_data.binary2onehot(data['Y_test'])
Y_val1_labels = new_read_data.binary2onehot(data['Y_val1'])
Y_val2_labels = new_read_data.binary2onehot(data['Y_val2'])
Y_val3_labels = new_read_data.binary2onehot(data['Y_val3'])

X_train_data = data['X_train']  #already one hot encoded
X_test_data = data['X_test']

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_outputholder))
# softmax_cross_entropy_with_logits_v2 : https://github.com/tensorflow/minigo/pull/149
#loss_op = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(
#         logits=logits, labels=y_outputholder))
optimizer_ = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

# Evaluate model and  define an accuracy assessment operation
correct_predictionSum = tf.equal(tf.argmax(y_outputholder, 1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictionSum, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.compat.v1.global_variables_initializer()
############
### get total number of batches that will be run in each epoch wrt batch size defined above
total_Nbatch=new_read_data.getIndicesofMinibatchs(featuredata=data['X_train'], featurelabels=Y_train_labels, batchsize_=batch_size, isShuffle=True)
total_Nbatch= int( len(total_Nbatch)/batch_size )

# Early stopping and performance variables
best_valid_acc_min = 0.0
best_valid_acc_median = 0.0
best_valid_acc_max = 0.0

best_train_acc = 0.0
best_avg_cost=1000000
avg_test_acc = np.array([])
dyn_LR=0.1
# Iteration-number for last improvement to validation accuracy.
last_improvement = 0
# Stop optimization if no improvement found in this many iterations.
require_improvement = 100 #int(epochs/10) could change this into %10 of the maximum epoch.
# Counter for total number of iterations performed so far.
total_iterations = 0
#
saver = tf.compat.v1.train.Saver()
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_path = os.path.join(save_dir, 'best_validation')
save_thelast_path = os.path.join(save_dir, 'last_weigths')
###
import statistics #to compute median
#
shuffle_ = np.arange(len(Y_train_labels))
# Start training
with tf.compat.v1.Session() as sess:
    # Start-time used for printing time-usage below.
    start_time = time.time()
    #initialise the variables
    if(runindx==0):
        sess.run(init) 
    else:
        #Restore the previous best model if runindx is >=1
        saver.restore(sess=sess, save_path=save_path)
    for epoch in range(epochs):
        total_iterations += 1 #for Saver
        #tf.set_random_seed(myseed[0]) 
        #np.random.seed(myseed[1])
        avg_cost = 0
        avg_train_acc = 0
        np.random.shuffle(shuffle_)
        Y_train_labels=Y_train_labels[shuffle_] # shuffle data to have a different order of training in each epoch
        X_train_data=X_train_data[shuffle_]
        batchindicesAll=new_read_data.getIndicesofMinibatchs(featuredata=X_train_data, featurelabels=Y_train_labels, batchsize_=batch_size, isShuffle=True)
        tmp4=0
        maxAcc=0.0 
        if( dyn_LR > lr_min*1.05):
             dyn_LR = lr_min + (lr_max - lr_min) * math.exp(-epoch/decay_speed)
        else:
            dyn_LR =  dyn_LR*0.9995
        dyn_LR = max(dyn_LR,lr_ultimate_min)
        for i in range(total_Nbatch):
            batch_x = X_train_data[batchindicesAll[tmp4:(tmp4+batch_size)]]
            batch_y = Y_train_labels[batchindicesAll[tmp4:(tmp4+batch_size)]]
            tmp4 = tmp4 + batch_size
            _, c = sess.run([optimizer_, loss_op], feed_dict={X: batch_x, y_outputholder: batch_y,
                                                    prob_:keep_prob_rate, learning_rate:dyn_LR})  
            avg_cost += c / total_Nbatch

            train_acc = accuracy.eval(feed_dict={X: batch_x, y_outputholder: batch_y, prob_: 1.0})
            avg_train_acc += train_acc / total_Nbatch

        #val
        val1_acc= accuracy.eval(feed_dict={X: data['X_val1'], y_outputholder: Y_val1_labels, prob_: 1.0})
        val2_acc= accuracy.eval(feed_dict={X: data['X_val2'], y_outputholder: Y_val2_labels, prob_: 1.0})
        val3_acc= accuracy.eval(feed_dict={X: data['X_val3'], y_outputholder: Y_val3_labels, prob_: 1.0})
        val_medianAcc = statistics.median([val1_acc,val2_acc,val3_acc])
        val_minAcc = min(val1_acc, val2_acc, val3_acc)
        val_maxAcc = max(val1_acc, val2_acc, val3_acc)
        #test
        test_acc= accuracy.eval(feed_dict={X: X_test_data, y_outputholder: Y_test_labels, prob_: 1.0})
        avg_test_acc = np.append(avg_test_acc,test_acc)
        predict_tests_ = sess.run(prediction, feed_dict={X: X_test_data, prob_: 1.0})
        #rho_soft, P_value = stats.spearmanr(Y_test_labels, predict_tests_)
        try:
            test_auc = roc_auc_score(Y_test_labels[:,0], predict_tests_[:,0])
        except ValueError:
            test_auc = 0 #np.nan
        #
        if (((val_minAcc > best_valid_acc_min) or 
             ((val_minAcc >= best_valid_acc_min) and (val_medianAcc > best_valid_acc_median)) or 
             ((val_minAcc >= best_valid_acc_min) and (val_medianAcc > best_valid_acc_median) and (val_maxAcc > best_valid_acc_max)))
            and ((train_acc >= 0.9) and (avg_cost < 0.12) and (epoch > 40))):
            # Update the best-known validation accuracy.
            best_valid_acc_min = val_minAcc
            best_valid_acc_median = val_medianAcc
            best_valid_acc_max = val_maxAcc
            best_train_acc = train_acc
            best_avg_cost = avg_cost
            # Set the iteration for the last improvement to current.
            last_improvement = total_iterations
            # Save all variables of the TensorFlow graph to file.
            saver.save(sess=sess, save_path=save_path)
            # A string to be printed below, shows improvement found.
            improved_str = '***'
        else:
            # An empty string to be printed below.
            # Shows that no improvement was found.
            improved_str = ''
        print("Ep.:", (epoch + 1), "test ACC.: {:.6f}".format(test_acc),
                                "| test AUC: {:.6f}".format(test_auc),
                                "| train ACC: {:.6f}".format(train_acc),
                                #"| train AUC: {:.6f}".format(train_auc),
                                "| avg_cost=", "{:.10f}".format(avg_cost),
                                "| val_min_acc.: {:.6f}".format(val_minAcc),
                                "| val_max_acc.: {:.6f}".format(val_maxAcc),
                                #"| Vali_min_acc.: {:.6f}".format(val_minAUC),
                                #"| Vali_median_acc.: {:.6f}".format(val_medianAUC),
                                "| ", improved_str,
                                "| Dyn.LR: {:.6f}".format(dyn_LR)
                                )
        if ((total_iterations - last_improvement > require_improvement) and (avg_cost<0.12)):
            print("No improvement found in a while, stopping optimization.")
            # Break out from the for-loop.
            break
            # Ending time.
    end_time = time.time()
    # Difference between start and end-times.
    time_dif = end_time - start_time
    # Print the time-usage.
    print("Transform data time: " + str(timedelta(seconds=int(round(transform_time)))))
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))
    # Save all THE LAST variables of the TensorFlow graph to file.
    saver.save(sess=sess, save_path=save_thelast_path)
    saver.save(sess=sess, save_path=save_path)
    #Restore the last saved best model model
    saver.restore(sess=sess, save_path=save_path)
    #
    filename_ = './' + 'results_dir' + 'res_' + testset +'_run' + str(runindx) +'.txt'
    thetext = open(filename_, "a")

    print("\nTraining complete!")
    print("date: ", str(datetime.datetime.now()))
    print("\nTest set: ", testset)

    if runindx == 0:
        print("new run")
    else:
        print("using the previous best weights")

    thetext.write("########### START of " + testset + " #######" + '\n')
    thetext.write("\nTest set: "+ testset + '\n')
    thetext.write("Training complete!"'\n')
    thetext.write("DATE: " + str(datetime.datetime.now()) + '\n')
    thetext.write("New run if 0; follow up run if 1 : " + str(runindx) + '\n'+ '\n')
    #
    X_train_1 = X_train_data[:10000]
    Y_train_1 = Y_train_labels[:10000]
    train_acc_1 = accuracy.eval(feed_dict={X: X_train_1, y_outputholder: Y_train_1, prob_: 1.0})
    predict_trains_1 = sess.run(prediction, feed_dict={X: X_train_1, prob_: 1.0})
    train_auc_1 = roc_auc_score(Y_train_1[:,0], predict_trains_1[:,0])

    X_train_2 = X_train_data[10000:20000]
    Y_train_2 = Y_train_labels[10000:20000]
    train_acc_2 = accuracy.eval(feed_dict={X: X_train_2, y_outputholder: Y_train_2, prob_: 1.0})
    predict_trains_2 = sess.run(prediction, feed_dict={X: X_train_2, prob_: 1.0})
    train_auc_2 = roc_auc_score(Y_train_2[:,0], predict_trains_2[:,0])

    X_train_3 = X_train_data[20000:30000]
    Y_train_3 = Y_train_labels[20000:30000]
    train_acc_3 = accuracy.eval(feed_dict={X: X_train_3, y_outputholder: Y_train_3, prob_: 1.0})
    predict_trains_3 = sess.run(prediction, feed_dict={X: X_train_3, prob_: 1.0})
    train_auc_3 = roc_auc_score(Y_train_3, predict_trains_3)

    train_acc_ = (train_acc_1 + train_acc_2 +train_acc_3)/3
    train_auc_ = (train_auc_1 + train_auc_2 +train_auc_3)/3

    test_acc_ = accuracy.eval(feed_dict={X: X_test_data, y_outputholder: Y_test_labels, prob_: 1.0})
    prediction_tests_ = sess.run(prediction, feed_dict={X: X_test_data, prob_: 1.0})
    test_auc_ = roc_auc_score(Y_test_labels[:,0], prediction_tests_[:,0])
    print("!!! Train AUC:",train_auc_)
    print("!!! Train ACC:",train_acc_)
    print("!!! Test AUC: ", test_auc_)
    print("!!! Test ACC: ", test_acc_)

    ##alternatively
    #mean_fpr, mean_tpr, mean_thresholds = roc_curve(Y_test_labels[:,0], predictions_test_[:,0], pos_label=1)
    #mean_auc = auc(mean_fpr, mean_tpr)


    thetext.write(testset + "!!! Test AUC: "+ str(test_auc_) + '\n')
    thetext.write(testset + " Test ACCURACY: "+ str(test_acc_) + '\n')


    val1_acc= accuracy.eval(feed_dict={X: data['X_val1'], y_outputholder: Y_val1_labels, prob_: 1.0})
    val2_acc= accuracy.eval(feed_dict={X: data['X_val2'], y_outputholder: Y_val2_labels, prob_: 1.0})
    val3_acc= accuracy.eval(feed_dict={X: data['X_val3'], y_outputholder: Y_val3_labels, prob_: 1.0})

    print("date: ", str(datetime.datetime.now()))
    print("Val1 ACC: ",val1_acc)
    print("Val2 ACC: ",val2_acc)
    print("Val3 ACC: ",val3_acc)
    print("Avg. test ACC:",np.mean(avg_test_acc))  #shows on average how was the performance on the test dataset
    thetext.write(testset + " Val1 acc: " + str(val1_acc) + '\n')
    thetext.write(testset + " Val2 acc: " + str(val2_acc) + '\n')
    thetext.write(testset + " Val3 acc: " + str(val3_acc) + '\n')
    thetext.write(testset + " Avg. test ACC: " + str(np.mean(avg_test_acc)) +'\n')
    thetext.write(str(datetime.datetime.now()) + '\n')
    thetext.write("########### End of " + testset +" #######" +'\n' + '\n' + '\n' + '\n')
    thetext.close()

