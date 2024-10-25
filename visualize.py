import matplotlib.pyplot as plt
import numpy as np
from preprocess import load_data

def visualize_samples():
    X_train, X_test, y_train, y_test = load_data()
    
    # Görüntülerin boyutlarını ayarlamak için
    height, width = 92, 112  # AT&T Face Database için standart boyutlar

    # İlk 5 örneği göster
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X_test[i].reshape(height, width), cmap='gray')  # 1D diziyi 2D'ye çevir
        plt.title(f'Label: {y_test[i]}')
        plt.axis('off')

    plt.show()  # Bu satırın olduğundan emin olun


if __name__ == "__main__":
    visualize_samples()
