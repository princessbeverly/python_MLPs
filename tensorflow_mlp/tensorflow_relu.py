

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time
import MNIST_Dataset
import Image

class MLP:
    
    
    def __init__(self, learning_rate=0.001):
        
        
        # Check GPU availability
        self.check_device()
        
        # Build model
        self.model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(784,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.metrics = {}
    
    def check_device(self):
        '''Check if running on GPU or CPU'''
        gpus = tf.config.list_physical_devices('GPU')
        
        print("=" * 60)
        print("DEVICE INFORMATION")
        print("=" * 60)
        
        if gpus:
            self.device_mode = "GPU"
            print(f"   Running on GPU")
            print(f"   Number of GPUs: {len(gpus)}")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu.name}")
        else:
            self.device_mode = "CPU"
            print(f"   Running on CPU (No GPU detected)")
        
        print("=" * 60)
    
    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
        '''Train the model and track metrics'''
        
        # Calculate batch information
        num_samples = len(X_train)
        num_batches_per_epoch = num_samples // batch_size
        total_batches = num_batches_per_epoch * epochs
        
        print(f"\nTraining Settings:")
        print(f"  Samples: {num_samples}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches per epoch: {num_batches_per_epoch}")
        print(f"  Total batches: {total_batches}")
        print(f"  Epochs: {epochs}\n")
        
        # Track training time
        start_time = time.time()
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        training_time_sec = time.time() - start_time
        training_time_ms = training_time_sec * 1000
        
        # Calculate metrics (in milliseconds)
        self.metrics['avg_training_time_ms'] = training_time_ms / epochs
        self.metrics['avg_batch_time_ms'] = training_time_ms / total_batches
        
        print(f"\n  Training completed in {training_time_sec:.2f} seconds")
        
        return history
    
    def test_classification(self, X_test):
        '''Measure classification time'''
        
        num_images = len(X_test)
        
        # Warm-up run (don't count this)
        _ = self.model.predict(X_test[:1], verbose=0)
        
        # Timed classification (this is what we measure)
        start_time = time.time()
        predictions = self.model.predict(X_test, verbose=0)
        inference_time_sec = time.time() - start_time
        
        inference_time_ms = inference_time_sec * 1000
        
        # Calculate average time per image (in milliseconds)
        self.metrics['avg_classification_time_ms'] = inference_time_ms / num_images
        
        print(f"\n  Classified {num_images} images in {inference_time_sec:.2f} seconds")
        
        return predictions
    
    def display_metrics(self):
        '''Display the three key metrics with device mode'''
        print("\n" + "=" * 60)
        print(f"PERFORMANCE METRICS ({self.device_mode} MODE)")
        print("=" * 60)
        print(f"Average Training Time: {self.metrics['avg_training_time_ms']:.2f} ms")
        print(f"Average Batching Time: {self.metrics['avg_batch_time_ms']:.2f} ms")
        print(f"Average Classifying Time: {self.metrics['avg_classification_time_ms']:.4f} ms")
        print("=" * 60)
    
    def save_metrics(self, filepath='metrics.txt'):
        '''Save metrics to a text file'''
        with open(filepath, 'w') as f:
            f.write(f"Device Mode: {self.device_mode}\n")
            f.write(f"Average Training Time: {self.metrics['avg_training_time_ms']:.2f} ms\n")
            f.write(f"Average Batching Time: {self.metrics['avg_batch_time_ms']:.2f} ms\n")
            f.write(f"Average Classifying Time: {self.metrics['avg_classification_time_ms']:.4f} ms\n")
        print(f"\n  Metrics saved to {filepath}")
    
    def test_sample_predictions(self, dataset, num_samples=5):
        '''Test predictions on sample images and display results'''
        print("\n" + "=" * 60)
        print("SAMPLE PREDICTIONS")
        print("=" * 60)
        
        import random
        sample_indices = random.sample(range(len(dataset.testing_images)), num_samples)
        
        correct_count = 0
        
        for i, idx in enumerate(sample_indices, 1):
            img = dataset.testing_images[idx]
            
            # Get prediction
            x = np.array([img.pixels])
            predictions = self.model.predict(x, verbose=0)
            predicted_digit = np.argmax(predictions[0])
            confidence = predictions[0][predicted_digit] * 100
            
            # Display image
            print(f"\nSample {i} (Image #{idx}):")
            img.display()
            
            # Display prediction
            print(f"True Label:    {img.label}")
            print(f"Predicted:     {predicted_digit}")
            print(f"Confidence:    {confidence:.2f}%")
            
            # Check if correct
            if predicted_digit == img.label:
                print("Result:          CORRECT")
                correct_count += 1
            else:
                print("Result:          INCORRECT")
        
        print(f"\n{'-' * 60}")
        print(f"Accuracy on samples: {correct_count}/{num_samples} ({correct_count/num_samples*100:.1f}%)")
        print("=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\nLoading MNIST dataset...")
    dataset = MNIST_Dataset.MNIST_Dataset('mnist_data')
    
    # Prepare data
    X_train = np.array([img.pixels for img in dataset.training_images])
    y_train = np.array([img.label for img in dataset.training_images])
    X_test = np.array([img.pixels for img in dataset.testing_images])
    y_test = np.array([img.label for img in dataset.testing_images])
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Create model
    print("\nCreating model...")
    model = MLP(learning_rate=0.001)
    
    # Train and measure
    print("\nStarting training...")
    history = model.train(X_train, y_train, X_test, y_test, epochs=10, batch_size=100)
    
    # Test classification speed
    print("\nTesting classification speed...")
    predictions = model.test_classification(X_test)
    
    # Display results
    model.display_metrics()
    
    # Test sample predictions
    print("\nTesting sample predictions...")
    model.test_sample_predictions(dataset, num_samples=5)
    
    # Optionally save to file
    model.save_metrics('tensorflow_metrics.txt')
    
    # Evaluate final accuracy
    test_loss, test_accuracy = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")
