
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


learning_rate = 0.1

# out affine transformation
def model(inputs):
    return tf.matmul(inputs,w)+b


# calculate loss with target and pred
# square can Avoid negetive loss
def square_loss(targets,predictions):
    per_sample_losses = tf.square(targets-predictions)
    return tf.reduce_mean(per_sample_losses)

# calculate grad of loss with respect to weights
def training_step(inputs,targets):
    with tf.GradientTape() as tape:
        pred = model(inputs)
        loss = square_loss(targets,pred)
    grad_loss_rsp_w,grad_loss_rsp_b = tape.gradient(loss,[w,b])
    w.assign_sub(grad_loss_rsp_w*learning_rate)
    b.assign_sub(grad_loss_rsp_b*learning_rate)
    return loss


#Main Task

num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5],[0.5, 1]],
    size=num_samples_per_class)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5],[0.5, 1]],
    size=num_samples_per_class)

#add both input to one array
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
#add both output to one array
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))

# show plot before deep learning
plt.scatter(inputs[:, 0], inputs[:, 1])
plt.show()

# true output
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()


input_dim = 2
output_dim =1
# create our weights : w and b ->  out = (input . w)+b
w= tf.Variable(tf.random.uniform((input_dim,output_dim)))
b = tf.Variable(tf.zeros(output_dim,))

# batch gd
for step in range(40):
    loss = training_step(inputs,targets)
    print(f"loss at step {step} : {loss:.4f}")
prediction = model(inputs)

# predict output
x = np.linspace(-1,4,100)
y = - w[0] / w[1] * x + (0.5 - b ) / w[1]
plt.plot(x,y,"-b")
plt.scatter(inputs[:, 0], inputs[:, 1], c=prediction[:,0]>0.5)
plt.show()


