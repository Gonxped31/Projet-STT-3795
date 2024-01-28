import torch as T
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random


class Datasets: 
    def __init__(self):
        pass 
        
    def loadCifar10Training(): 
        with open('./src/data/cifar-10-batches-py/train/data_batch_1', 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
    
        data = T.tensor(batch['data'], dtype=T.float32).view(-1, 3, 32, 32)
        labels = T.tensor(batch['labels'], dtype=T.long)
        
        return {'data': data, 'labels': labels}

    # Params: 
    #   int: k. 
    # Returns: 
    #   k images from the CIFAR-10 training dataset 
    def getImages(self, k): 
        dataset = Datasets.loadCifar10Training()  # Load the dataset
        data = dataset['data']
        labels = dataset['labels']

        # Select k random indices
        indices = random.sample(range(len(data)), k)

        # Create a figure with subplots
        fig, axes = plt.subplots(1, k, figsize=(k * 2, 2))

        # Save the image paths 
        imgPaths = []
        for i, idx in enumerate(indices):
            image = data[idx].numpy()  # Convert tensor to numpy array
            image = np.transpose(image, (1, 2, 0))  # Rearrange color channels
            axes[i].imshow(image / 255)  # Normalize and display image
            axes[i].set_title(f"Label: {labels[idx].item()}")
            axes[i].axis('off')
            plt.show()

            # Save each image as a JPEG file
            plt.imsave(f'image_{idx}.jpg', image / 255)

       

# Usage
datasets = Datasets()
datasets.getImages(5)  # Display and save 5 random images