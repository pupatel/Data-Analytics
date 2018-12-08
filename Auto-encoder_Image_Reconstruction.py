# -*- coding: utf-8 -*-

#Created by Parth Patel, DBI @ University of Delaware, Newark, Delaware 19717
#Date created: 8/07/2018

##This script takes input of images -> encode -> decond -> recontrsuct input images using autoencoder. 
#usage: python3 Auto-encoder_Image_Reconstruction.py


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#LOAD IMAGES FROM TRAINING SET
all_images = np.loadtxt('fashion-mnist_train.csv',delimiter=',', skiprows=1)[:,1:]
#print(all_images.shape)

# printing the array representation of the first image
#print("the array of the first image looks like", all_images[0])
# printing something that actually looks like an image
#print("and the actual image looks like")
#plt.imshow(all_images[0].reshape(28,28),  cmap='Greys')
#plt.show()

# SETTING UP NUMBER OF NUERONS/NODES AT EACH IN THE AUTOENCODERS
n_nodes_inputlayer = 784  #encoder
n_nodes_hiddenlayer1  = 32  #encoder
n_nodes_hiddenlayer2  = 32  #decoder
n_nodes_outlayer = 784  #decoder

#ASSIGN WEIGTHS AND BIASES TO LAYERS

# FIRST HIDDEN LAYER HAS 784*32 WEIGHTS AND 32 BIASES
hidden_1_layer_vals = {'weights':tf.Variable(tf.random_normal([n_nodes_inputlayer,n_nodes_hiddenlayer1])),'biases':tf.Variable(tf.random_normal([n_nodes_hiddenlayer1]))  }

# SECOND HIDDEN LAYER HAS 32*32 WEIGHTS AND 32 BIASES
hidden_2_layer_vals = {'weights':tf.Variable(tf.random_normal([n_nodes_hiddenlayer1, n_nodes_hiddenlayer2])),'biases':tf.Variable(tf.random_normal([n_nodes_hiddenlayer2]))  }

# OUTPUT HIDDEN LAYER HAS 32*784 WEIGHTS AND 784 BIASES
output_layer_vals = {'weights':tf.Variable(tf.random_normal([n_nodes_hiddenlayer2,n_nodes_outlayer])),'biases':tf.Variable(tf.random_normal([n_nodes_outlayer]))}


# IMAGE WITH SHAPE 784 GOES IN
input_layer = tf.placeholder('float', [None, 784])

# MULTIPLY OUTPUT OF INPUT_LAYER WTH A WEIGHT MATRIX AND ADD BIASES
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_layer,hidden_1_layer_vals['weights']),hidden_1_layer_vals['biases']))

# MULTIPLY OUTPUT OF LAYER_1 WTH A WEIGHT MATRIX AND ADD BIASES
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,hidden_2_layer_vals['weights']),hidden_2_layer_vals['biases']))

# MULTIPLY OUTPUT OF LAYER_2 WTH A WEIGHT MATRIX AND ADD BIASES
output_layer = tf.matmul(layer_2,output_layer_vals['weights']) + output_layer_vals['biases']

# OUTPUT_TRUE  HAVE THE ORIGINAL IMAGE FOR ERROR CALCULATIONS
output_original = tf.placeholder('float', [None, 784])

# DEFINE COST FUNCTION TO MINIMIZE ERROR
meansq = tf.reduce_mean(tf.square(output_layer - output_original))

# USING ADAM OPTIMIZER
learn_rate = 0.1   # how fast the model should learn
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)


# INITIATLIZE & START SESSION
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# DEFINING BATCH SIZE, NUMBER OF EPOCHS AND LEARNING RATE
batch_size = 100  # Try 128
hm_epochs =1000    # This may be too much
tot_images = 60000 # Total set


# RUNNING THE MODEL FOR A 1000 EPOCHS TAKING 100 IMAGES IN BATCHES
# TOTAL IMPROVEMENT IS PRINTED OUT AFTER EACH EPOCH

for epoch in range(hm_epochs):
    epoch_loss = 0    # initializing error as 0
    for i in range(int(tot_images/batch_size)):
        epoch_x = all_images[ i*batch_size : (i+1)*batch_size ]_, c = sess.run([optimizer, meansq],feed_dict={input_layer: epoch_x,output_true: epoch_x})
        epoch_loss += c
        
print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)




# PICK ANY IMAGE
any_image = all_images[900]

# RUN IT THROUGH THE AUTOENCODER
output_any_image = sess.run(output_layer,feed_dict={input_layer:[any_image]})

# RUN IT THROUGH JUST THE ENCODER
encoded_any_image = sess.run(layer_1,feed_dict={input_layer:[any_image]})

# PRINT THE ORIGINAL IMAGE

plt.imshow(any_image(28,28),  cmap='Greys')
plt.show()

# PRINT THE ENCODING
print(encoded_any_image)
