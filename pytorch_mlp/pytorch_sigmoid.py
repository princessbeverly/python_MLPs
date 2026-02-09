

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_mnist_dataset import TorchMNIST
import numpy as np
import time

class MLP(nn.Module):
    
    
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.net(x)


class MLP_Classifier:
    '''
    MLP with GPU/CPU detection and minimal performance tracking
    '''
    
    def __init__(self, learning_rate=0.001):
        '''Initialize MLP and detect device'''
        
        # Check GPU availability
        self.check_device()
        
        # Build model
        self.model = MLP().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.metrics = {}
    
    def check_device(self):
        '''Check if running on GPU or CPU'''
        print("=" * 60)
        print("DEVICE INFORMATION")
        print("=" * 60)
        
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.device_mode = "GPU"
            print(f" Running on GPU")
            print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
        else:
            self.device = torch.device('cpu')
            self.device_mode = "CPU"
            print(f"  Running on CPU (No GPU detected)")
        
        print("=" * 60)
    
    def train(self, train_loader, test_loader, epochs=10):
        '''Train the model and track metrics'''
        
        # Calculate batch information
        num_batches_per_epoch = len(train_loader)
        total_batches = num_batches_per_epoch * epochs
        batch_size = train_loader.batch_size
        num_samples = len(train_loader.dataset)
        
        print(f"\nTraining Settings:")
        print(f"  Samples: {num_samples}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batches per epoch: {num_batches_per_epoch}")
        print(f"  Total batches: {total_batches}")
        print(f"  Epochs: {epochs}\n")
        
        # Track training time
        start_time = time.time()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
            
            avg_loss = total_loss / num_batches_per_epoch
            accuracy = 100 * correct / total
            
            print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {accuracy:.2f}%')
        
        training_time_sec = time.time() - start_time
        training_time_ms = training_time_sec * 1000
        
        # Calculate metrics (in milliseconds)
        self.metrics['avg_training_time_ms'] = training_time_ms / epochs
        self.metrics['avg_batch_time_ms'] = training_time_ms / total_batches
        
        print(f"\n Training completed in {training_time_sec:.2f} seconds")
    
    def test_classification(self, test_loader):
        '''Measure classification time'''
        
        self.model.eval()
        
        # Count total images
        num_images = len(test_loader.dataset)
        
        # Warm-up run (don't count this)
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                _ = self.model(x)
                break  # Just one batch for warm-up
        
        # Timed classification (this is what we measure)
        start_time = time.time()
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(self.device)
                predictions = self.model(x)
        
        inference_time_sec = time.time() - start_time
        inference_time_ms = inference_time_sec * 1000
        
        # Calculate average time per image (in milliseconds)
        self.metrics['avg_classification_time_ms'] = inference_time_ms / num_images
        
        print(f"\n Classified {num_images} images in {inference_time_sec:.2f} seconds")
    
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
        print(f"\n Metrics saved to {filepath}")
    
    def test_sample_predictions(self, dataset, num_samples=5):
        '''Test predictions on sample images and display results'''
        print("\n" + "=" * 60)
        print("SAMPLE PREDICTIONS")
        print("=" * 60)
        
        import random
        sample_indices = random.sample(range(len(dataset.dataset.testing_images)), num_samples)
        
        self.model.eval()
        correct_count = 0
        
        for i, idx in enumerate(sample_indices, 1):
            img = dataset.dataset.testing_images[idx]
            
            # Get prediction
            x = torch.tensor(img.pixels).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(x)
                probabilities = torch.softmax(output, dim=1)[0]
                predicted_digit = output.argmax(dim=1).item()
                confidence = probabilities[predicted_digit].item() * 100
            
            # Display image
            print(f"\nSample {i} (Image #{idx}):")
            img.display()
            
            # Display prediction
            print(f"True Label:    {img.label}")
            print(f"Predicted:     {predicted_digit}")
            print(f"Confidence:    {confidence:.2f}%")
            
            # Check if correct
            if predicted_digit == img.label:
                print("Result:        CORRECT")
                correct_count += 1
            else:
                print("Result:         INCORRECT")
        
        print(f"\n{'-' * 60}")
        print(f"Accuracy on samples: {correct_count}/{num_samples} ({correct_count/num_samples*100:.1f}%)")
        print("=" * 60)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\nLoading MNIST dataset...")
    train_dataset = TorchMNIST("mnist_data", train=True)
    test_dataset = TorchMNIST("mnist_data", train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = MLP_Classifier(learning_rate=0.001)
    
    # Train and measure
    print("\nStarting training...")
    model.train(train_loader, test_loader, epochs=10)
    
    # Test classification speed
    print("\nTesting classification speed...")
    model.test_classification(test_loader)
    
    # Display results
    model.display_metrics()
    
    # Test sample predictions
    print("\nTesting sample predictions...")
    model.test_sample_predictions(test_dataset, num_samples=5)
    
    # Optionally save to file
    model.save_metrics('pytorch_metrics.txt')
    
    # Evaluate final accuracy
    model.model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(model.device), y.to(model.device)
            outputs = model.model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    accuracy = 100 * correct / total
    print(f"\nFinal Test Accuracy: {accuracy:.2f}%")
