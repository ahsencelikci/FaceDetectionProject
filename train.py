# train.py
from preprocess import load_data
from model import FaceRecognitionModel

def main():
    X_train, X_test, y_train, y_test = load_data()
    
    # Modeli oluştur ve eğit
    model = FaceRecognitionModel()
    model.train(X_train, y_train)

    # Test yap ve sonuçları yazdır
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()  # Doğruluğu hesapla
    print(f"Model Doğruluğu: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
