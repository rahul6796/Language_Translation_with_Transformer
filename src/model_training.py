

from transformers import create_optimizer
import tensorflow as tf
from model_initialized import tf_train_dataset, tf_eval_dataset
from model_initialized import model


num_epochs = 1

num_train_steps = len(tf_train_dataset) * num_epochs 

optimizer, schedule = create_optimizer(
    init_lr=0.005,
    num_warmup_steps=0,
    num_train_steps=num_train_steps,
    weight_decay_rate=0.01
)
model.compile()

model.fit(
    tf_train_dataset,
    epochs=num_epochs,
    validation_data=tf_eval_dataset
)


model.save_pretrained('rahul-en-to-fr')
