import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from constant import DATA_DIR, IMAGE_HEIGHT, IMAGE_WIDTH

def load_data():
    images = []
    labels = []

    for i in range(1, 41):  
        person_dir = os.path.join(DATA_DIR, f's{i}')
        for img_name in os.listdir(person_dir):
            if img_name.endswith('.pgm'):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Gri tonlamada oku
                img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))  # Boyutlandır
                images.append(img.flatten())  # Vektör hale getir
                labels.append(i)  # Etiketleri ayarla

    X = np.array(images)
    y = np.array(labels)
    return train_test_split(X, y, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print("Veri Yükleme Tamamlandı.")
