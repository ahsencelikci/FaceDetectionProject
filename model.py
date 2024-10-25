from sklearn.svm import SVC

class FaceRecognitionModel:
    def __init__(self):
        self.model = SVC(kernel='linear', C=1)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("Model Eğitildi.")

    def predict(self, X_test):
        return self.model.predict(X_test)
