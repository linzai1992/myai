import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
from data_batcher import DataBatcher
from vae_model import VAEModel
from time import time
import numpy as np
import tensorflow as tf

print("Finding training data...")
batcher = DataBatcher("generated_data")

print("Building model...")
model = VAEModel(50, [40, 35, 30])
batch_size = 5000
training_steps = 200000

print("Starting training...")
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(training_steps):
        batch, epoch_complete = batcher.get_batch(batch_size)
        model.train_model(session, inputs=batch)
        if epoch_complete:
            test_batch = batcher.get_test_batch()
            loss = model.get_loss(session, inputs=test_batch)
            print("Epoch complete - loss: {}".format(loss))
        if i % 500 == 0:
            test_batch = batcher.get_test_batch()
            loss = model.get_loss(session, inputs=test_batch)
            print("Step {} - loss: {}".format(i, loss))
