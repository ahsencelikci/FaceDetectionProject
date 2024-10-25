import matplotlib.pyplot as plt
import numpy as np
from preprocess import load_data

def visualize_samples():
    X_train, X_test, y_train, y_test = load_data()
    

    height, width = 92, 112  

    
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_test[i].reshape(height, width), cmap='gray')  
        plt.title(f'Label: {y_test[i]}')
        plt.axis('off')

    plt.show()  

if __name__ == "__main__":
    visualize_samples()
