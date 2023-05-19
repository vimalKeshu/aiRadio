import os 
import pickle

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier

class ApproximatorML(object):

    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            self.__load()
        else:
            self.model = RandomForestClassifier()

    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
        self.model.fit(X_train,y_train)
        y_pred=self.model.predict(X_test)
        print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    def predict(self, X):
        return self.model.predict(X)

    def __load(self) -> RandomForestClassifier:
        self.model = pickle.load(open(self.path,'rb'))

    def save(self):
        pickle.dump(self.model, open(self.path, 'wb'))

    def analyze_model_behavior(self):
        import pandas as pd
        from app_constant import cols
        feature_imp = pd.Series(self.model.feature_importances_,index=cols[1:]).sort_values(ascending=False)
        print(feature_imp)
