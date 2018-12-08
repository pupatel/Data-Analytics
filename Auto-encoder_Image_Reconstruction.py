
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Load images from training set
all_images = np.loadtxt('fashion-mnist_train.csv',delimiter=',', skiprows=1)[:,1:]
#print(all_images.shape)

# printing the array representation of the first image
print("the array of the first image looks like", all_images[0])
# printing something that actually looks like an image
print("and the actual image looks like")
plt.imshow(all_images[0].reshape(28,28),  cmap='Greys')
plt.show()

# Setting up numbers of nuerons/nodes at each in the autoEncoders
n_nodes_inputlayer = 784  #encoder
n_nodes_hiddenlayer1  = 32  #encoder
n_nodes_hiddenlayer2  = 32  #decoder
n_nodes_outlayer = 784  #decoder

#ASSIGN WEIGTHS AND BIASES

# first hidden layer has 784*32 weights and 32 biases
hidden_1_layer_vals = {'weights':tf.Variable(tf.random_normal([n_nodes_inputlayer,n_nodes_hiddenlayer1])),'biases':tf.Variable(tf.random_normal([n_nodes_hiddenlayer1]))  }

# second hidden layer has 32*32 weights and 32 biases
hidden_2_layer_vals = {'weights':tf.Variable(tf.random_normal([n_nodes_hiddenlayer1, n_nodes_hiddenlayer2])),'biases':tf.Variable(tf.random_normal([n_nodes_hiddenlayer2]))  }

# output hidden layer has 32*784 weights and 784 biases
output_layer_vals = {'weights':tf.Variable(tf.random_normal([n_nodes_hiddenlayer2,n_nodes_outlayer])),'biases':tf.Variable(tf.random_normal([n_nodes_outlayer]))}



# image with shape 784 goes in
input_layer = tf.placeholder('float', [None, 784])

# multiply output of input_layer wth a weight matrix and add biases
layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(input_layer,hidden_1_layer_vals['weights']),hidden_1_layer_vals['biases']))

# multiply output of layer_1 wth a weight matrix and add biases
layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,hidden_2_layer_vals['weights']),hidden_2_layer_vals['biases']))

# multiply output of layer_2 wth a weight matrix and add biases
output_layer = tf.matmul(layer_2,output_layer_vals['weights']) + output_layer_vals['biases']

# output_true shall have the original image for error calculations

output_true = tf.placeholder('float', [None, 784])

# define our cost function
meansq =    tf.reduce_mean(tf.square(output_layer - output_true))

# define our optimizer
learn_rate = 0.1   # how fast the model should learn
optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)



# initialising stuff and starting the session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# defining batch size, number of epochs and learning rate
batch_size = 100  # how many images to use together for training
hm_epochs =1000    # how many times to go through the entire dataset
tot_images = 60000 # total number of images


# running the model for a 1000 epochs taking 100 images in batches
# total improvement is printed out after each epoch

for epoch in range(hm_epochs):
    epoch_loss = 0    # initializing error as 0
    for i in range(int(tot_images/batch_size)):
        epoch_x = all_images[ i*batch_size : (i+1)*batch_size ]_, c = sess.run([optimizer, meansq],feed_dict={input_layer: epoch_x,output_true: epoch_x})
        epoch_loss += c
        
print('Epoch', epoch, '/', hm_epochs, 'loss:',epoch_loss)









# pick any image
any_image = all_images[999]
# run it though the autoencoder
output_any_image = sess.run(output_layer,feed_dict={input_layer:[any_image]})

# run it though just the encoder
encoded_any_image = sess.run(layer_1,feed_dict={input_layer:[any_image]})

# print the original image

plt.imshow(any_image(28,28),  cmap='Greys')
plt.show()
# print the encoding
print(encoded_any_image)
