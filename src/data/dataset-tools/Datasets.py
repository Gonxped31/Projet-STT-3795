import torch as T
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import os

class Datasets: 
    def __init__(self):
        pass 
    
    #Load the CIFAR-10 dataset
    def loadCifar10Training(): 
        with open('./src/data/cifar-10-batches-py/train/data_batch_1', 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
    
        data = T.tensor(batch['data'], dtype=T.float32).view(-1, 3, 32, 32)
        labels = T.tensor(batch['labels'], dtype=T.long)
        
        return {'data': data, 'labels': labels}

    # Params: 
    #   int: k. 
    # Returns: 
    #   an array of k images from the CIFAR-10 training dataset 
    def getCifar10Images(self, k): 
        print("Current Working Directory:", os.getcwd()) 
        dataset = Datasets.loadCifar10Training()  # Load the dataset
        data = dataset['data']
        labels = dataset['labels']

        # Select k random indices
        indices = random.sample(range(len(data)), k)

        # Save the image paths 
        imgPaths = []

        save_dir = './src/data/dataset-tools'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i, idx in enumerate(indices):
            image = data[idx].numpy()  # Convert tensor to numpy array
            image = np.transpose(image, (1, 2, 0))  # Rearrange color channels

            # Save each image as a JPEG file
            img_path = os.path.join(save_dir, f'image_{idx}.jpg')

            # Save each image as a JPEG file
            imgPaths.append(img_path)
            plt.imsave(img_path, image / 255)
        return imgPaths