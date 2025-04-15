import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import cv2
import os
import time

def load_test_data():
    """
    Load test data from pickle files
    If pickle files are not available, create synthetic test data
    """
    try:
        # Try to load a small portion of the data for testing
        print("Attempting to load test data...")
        
        # Check if files exist
        if os.path.exists("full_CNN_train.p") and os.path.exists("full_CNN_labels.p"):
            # Load a small subset for testing
            test_images = pickle.load(open("full_CNN_train.p", "rb"))[:100]  # Just load first 100 for testing
            test_labels = pickle.load(open("full_CNN_labels.p", "rb"))[:100]  # Just load first 100 for testing
            
            # Convert to numpy arrays
            test_images = np.array(test_images)
            test_labels = np.array(test_labels)
            
            # Normalize labels
            test_labels = test_labels / 255.0
            
            return test_images, test_labels
        else:
            raise FileNotFoundError("Test data files not found")
            
    except Exception as e:
        print(f"Error loading test data: {e}")
        print("Creating synthetic test data instead...")
        
        # Create synthetic test data
        # 20 synthetic 80x160x3 images with random pixels
        test_images = np.random.rand(20, 80, 160, 3)
        
        # Create synthetic lane labels (simple vertical lines)
        test_labels = np.zeros((20, 80, 160, 1))
        for i in range(20):
            # Create 2 vertical lines as lane markings
            pos1 = np.random.randint(40, 60)
            pos2 = np.random.randint(100, 120)
            test_labels[i, :, pos1:pos1+2, 0] = 1.0
            test_labels[i, :, pos2:pos2+2, 0] = 1.0
            
        return test_images, test_labels

def calculate_iou(y_true, y_pred, threshold=0.5):
    """
    Calculate IoU (Intersection over Union) for lane detection
    """
    # Apply threshold to predictions
    y_pred_binary = (y_pred > threshold).astype(np.float32)
    y_true_binary = (y_true > threshold).astype(np.float32)
    
    # Calculate intersection and union
    intersection = np.sum(y_pred_binary * y_true_binary)
    union = np.sum(y_true_binary) + np.sum(y_pred_binary) - intersection
    
    # Return IoU
    if union == 0:
        return 0.0
    return intersection / union

def evaluate_model():
    """
    Evaluate the model on test data and generate performance metrics
    """
    print("Starting model evaluation...")
    
    # Create directory for results
    results_dir = "evaluation_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Load test data
    test_images, test_labels = load_test_data()
    print(f"Loaded {len(test_images)} test images with shape {test_images.shape}")
    
    # Load model
    try:
        print("Loading model...")
        model = load_model('converted_CNN_model.h5', compile=False)
        print("Model loaded successfully!")
        
        # Make predictions
        print("Making predictions...")
        start_time = time.time()
        predictions = model.predict(test_images)
        end_time = time.time()
        
        # Calculate inference time
        inference_time = (end_time - start_time) / len(test_images) * 1000  # in ms
        print(f"Average inference time: {inference_time:.2f} ms per image")
        
        # Calculate metrics
        print("Calculating performance metrics...")
        
        # Flatten predictions and labels for pixel-wise metrics
        flat_preds = predictions.flatten()
        flat_labels = test_labels.flatten()
        
        # Apply threshold to get binary predictions
        binary_preds = (flat_preds > 0.5).astype(np.int32)
        binary_labels = (flat_labels > 0.5).astype(np.int32)
        
        # Calculate metrics
        accuracy = accuracy_score(binary_labels, binary_preds)
        precision = precision_score(binary_labels, binary_preds, zero_division=1)
        recall = recall_score(binary_labels, binary_preds, zero_division=1)
        f1 = f1_score(binary_labels, binary_preds, zero_division=1)
        cm = confusion_matrix(binary_labels, binary_preds, normalize='true')
        
        # Calculate IoU for each image
        ious = []
        for i in range(len(test_images)):
            iou = calculate_iou(test_labels[i], predictions[i])
            ious.append(iou)
        mean_iou = np.mean(ious)
        
        # Save results to a text file
        with open(os.path.join(results_dir, "metrics.txt"), "w") as f:
            f.write(f"Model Evaluation Results\n")
            f.write(f"======================\n\n")
            f.write(f"Number of test images: {len(test_images)}\n")
            f.write(f"Average inference time: {inference_time:.2f} ms per image\n\n")
            f.write(f"Pixel-wise Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"Mean IoU: {mean_iou:.4f}\n\n")
            f.write(f"Confusion Matrix (normalized):\n")
            f.write(f"[ {cm[0, 0]:.4f}, {cm[0, 1]:.4f} ]\n")
            f.write(f"[ {cm[1, 0]:.4f}, {cm[1, 1]:.4f} ]\n")
        
        print(f"Results saved to {os.path.join(results_dir, 'metrics.txt')}")
        
        # Visualize predictions for a few test images
        print("Generating visualization of predictions...")
        
        # Select a few random images for visualization
        indices = np.random.choice(len(test_images), min(5, len(test_images)), replace=False)
        
        for idx, i in enumerate(indices):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(test_images[i])
            axes[0].set_title("Original Image")
            axes[0].axis("off")
            
            # Ground truth
            axes[1].imshow(test_labels[i, :, :, 0], cmap='gray')
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")
            
            # Prediction
            axes[2].imshow(predictions[i, :, :, 0], cmap='gray')
            axes[2].set_title(f"Prediction (IoU: {ious[i]:.2f})")
            axes[2].axis("off")
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"prediction_{idx}.png"), dpi=200)
            plt.close()
        
        print(f"Visualization saved to {results_dir} directory")
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": mean_iou,
            "confusion_matrix": cm,
            "inference_time": inference_time
        }
        
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return None

if __name__ == "__main__":
    evaluate_model() 