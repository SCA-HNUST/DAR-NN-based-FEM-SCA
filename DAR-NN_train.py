import os

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models, Input
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# -----------------------------
# Custom Layer: Gradient Reversal
# -----------------------------
class GradientReversal(tf.keras.layers.Layer):
    def __init__(self, hp_lambda=1.0, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.hp_lambda = hp_lambda

    def call(self, x):
        @tf.custom_gradient
        def reverse_gradient(x):
            def grad(dy):
                return -self.hp_lambda * dy
            return x, grad
        return reverse_gradient(x)
        
# -----------------------------
# Submodel 1: Reconstructor
# -----------------------------
def build_reconstructor(input_tensor):
    x = layers.Conv1D(16, kernel_size=1, padding='same', activation='relu')(input_tensor)
    x = layers.AveragePooling1D(pool_size=2, padding='same')(x)
    x = layers.Conv1D(8, kernel_size=1, padding='same', activation='relu')(x)
    x = layers.AveragePooling1D(pool_size=2, padding='same')(x)
    latent = x  # (5, 8)

    x = layers.Conv1D(8, kernel_size=1, padding='same', activation='relu')(latent)
    x = layers.UpSampling1D(size=2)(x)
    x = layers.Conv1D(16, kernel_size=1, padding='same', activation='relu')(x)
    x = layers.UpSampling1D(size=2)(x)
    output = layers.Conv1D(1, kernel_size=1, padding='same', activation='linear', name='reconstructed')(x)
    return output, latent

# -----------------------------
# Submodel 2: Classifier
# -----------------------------
def build_classifier(reconstructed_input):
    x = layers.Conv1D(32, kernel_size=3, activation='relu')(reconstructed_input)
    x = layers.AveragePooling1D(pool_size=2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(256, activation='relu')(x)
    output = layers.Dense(256, activation='softmax', name='class_output')(x)
    return output
    
# -----------------------------
# Submodel 3: Domain Discriminator (with GRL)
# -----------------------------
def build_domain_discriminator(trace_input, hp_lambda=1.0):
    x = GradientReversal(hp_lambda)(trace_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dense(8, activation='relu')(x)
    x = layers.Dense(4, activation='relu')(x)
    x = layers.Flatten()(x)
    output = layers.Dense(2, activation='softmax', name='domain_output')(x)
    return output

def build_full_model(hp_lambda=1.0):
    noisy_traces = Input(shape=(20, 1), name='noisy_traces')
    groundtruth_traces = Input(shape=(20, 1), name='groundtruth_traces')

    reconstructed_traces, _ = build_reconstructor(noisy_traces)
    class_output = build_classifier(reconstructed_traces)

    def take_half(x):
        batch_size = tf.shape(x)[0] // 2
        return x[:batch_size]

    half_groundtruth = layers.Lambda(take_half, output_shape=(20, 1))(groundtruth_traces)
    half_reconstructed = layers.Lambda(take_half, output_shape=(20, 1))(reconstructed_traces)

    concat_traces = layers.Concatenate(axis=0)([half_groundtruth, half_reconstructed])
    domain_output = build_domain_discriminator(concat_traces, hp_lambda)

    model = Model(inputs=[noisy_traces, groundtruth_traces],
                  outputs=[reconstructed_traces, class_output, domain_output])
    return model

# -----------------------------
# Compile the Model
# -----------------------------
def compile_model(model, learning_rate):
    model.compile(
        optimizer=Adam(learning_rate),
        loss={
            'reconstructed': MeanSquaredError(),
            'class_output': CategoricalCrossentropy(),
            'domain_output': CategoricalCrossentropy()
        },
        loss_weights={
            'reconstructed': 0.03,
            'class_output': 0.95,
            'domain_output': 0.02
        },
        metrics={
            'class_output': 'accuracy',
            'domain_output': 'accuracy'
        }
    )
    return model

# -----------------------------
# Visualization of the training
# -----------------------------
def plot_training_history(history):
    # Extract all metric names from history
    metrics = list(history.history.keys())
    
    # Separate metrics into loss and accuracy categories
    loss_metrics = [m for m in metrics if 'loss' in m and not m.startswith('val_')]
    accuracy_metrics = [m for m in metrics if ('accuracy' in m or 'acc' in m) and not m.startswith('val_')]
    
    # Plot each loss metric in a separate figure
    for metric in loss_metrics:
        plt.figure(figsize=(5, 5))
        plt.plot(history.history[metric], label=metric)
        
        # Plot corresponding validation metric if exists
        val_metric = 'val_' + metric
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=val_metric)
        
        plt.title(f'{metric.capitalize().replace("_", " ")} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    # Plot each accuracy metric in a separate figure
    for metric in accuracy_metrics:
        plt.figure(figsize=(5, 5))
        plt.plot(history.history[metric], label=metric)
        
        # Plot corresponding validation metric if exists
        val_metric = 'val_' + metric
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=val_metric)
        
        plt.title(f'{metric.capitalize().replace("_", " ")} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    # If no metrics found at all
    if not loss_metrics and not accuracy_metrics:
        plt.figure(figsize=(5, 5))
        plt.text(0.5, 0.5, 'No training metrics found', 
                ha='center', va='center')
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
# -----------------------------
# Build & Compile
# -----------------------------

    learning_rate = 0.00003
    
    noisy_traces = np.load("Data/Train/noisy_traces.npy")
    noisy_traces = noisy_traces[:, 200:220]
    groundtruth_traces = np.load("Data/Train/groundtruth_traces.npy")
    groundtruth_traces = groundtruth_traces[:, 200:220]
    key_labels = np.load("Data/Train/key_labels.npy")
    domain_labels = np.load("Data/Train/domain_labels.npy")
    
    key_labels = to_categorical(key_labels, num_classes=256)
    domain_labels = to_categorical(domain_labels, num_classes=2)
    print(key_labels.shape)
    
    model = build_full_model(hp_lambda=0.5)
    model = compile_model(model, learning_rate)
    model.summary()
    
    # Train
    history = model.fit(
        x={"noisy_traces": noisy_traces, "groundtruth_traces": groundtruth_traces},
        y={
            "reconstructed": groundtruth_traces,
            "class_output": key_labels,
            "domain_output": domain_labels
        },
        validation_split=0.2,
        batch_size=256,
        epochs=200
    )
    
    plot_training_history(history)
    
    # Step 1: Get the input to the reconstructor
    noisy_input = model.get_layer('noisy_traces').input
    
    # Step 2: Get the output of the 'reconstructed' layer (from the trained model)
    reconstructed_output = model.get_layer('reconstructed').output
    
    # Step 3: Get the output of the classifier
    class_output = model.get_layer('class_output').output
    
    # Step 4: Build a new model chaining input -> reconstructor -> classifier
    combined_model = Model(inputs=noisy_input, outputs=class_output)
    combined_model.save('DAR-NN.h5')
        