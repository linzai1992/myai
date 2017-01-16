from data_batcher import DataBatcher
from vae_model import VAEModel
from time import time
import numpy as np
import tensorflow as tf

print("Finding training data...")
batcher = DataBatcher("generated_data")

print("Building model...")
model = VAEModel(80, [70, 60, 50, 40])
batch_size = 500
training_steps = 150000

print("Starting training...")
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(training_steps):
        batch, epoch_complete = batcher.get_batch(batch_size)
        model.train_model(session, inputs=batch)
        if epoch_complete:
            print("Epoch complete")
        if i % 500 == 0:
            loss = model.get_loss(session, inputs=batch)
            print("Step {} - loss: {}".format(i, loss))
