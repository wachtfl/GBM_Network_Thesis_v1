from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class ModelFactory:
    @staticmethod
    def get(model_name):
        if model_name == "logistic":
            return LogisticRegression()
        if model_name == "dtree":
            return DecisionTreeClassifier()
        if model_name == "rf":
            return RandomForestClassifier()
        if model_name == "svm":
            return SVC()
        raise ValueError(f"Unknown model: {model_name}")
