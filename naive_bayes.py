'''
equation is p(w|x)=p(x|w)*p(w)/p(x)
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class NaiveBayes:
    def __init__(self):
        self.__x_train = None
        self.__y_train = None
        self.__Wprob = None
        self.__Xprob = None

    def fit(self, data, target):
        self.__x_train = data
        self.__y_train = target
        self.__Wprob = self.__aprior()
        self.__Xprob = self.__marginalization()

    def __aprior(self):
        return {label: np.mean(self.__y_train == label) for label in set(self.__y_train)}

    def __marginalization(self):
        Xprob = {}
        for label in set(self.__y_train):
            label_mask = (self.__y_train == label)
            Xprob[label] = {
                feature: {
                    'mean': np.mean(self.__x_train[label_mask, feature]),
                    'std': np.std(self.__x_train[label_mask, feature]) + 1e-8
                }
                for feature in range(self.__x_train.shape[1])
            }
        return Xprob

    def __likelihood(self, X, Y):
        sqrt_2pi = np.sqrt(2 * np.pi)
        means = np.array([self.__Xprob[Y][feature]['mean'] for feature in range(X.shape[0])])
        stds = np.array([self.__Xprob[Y][feature]['std'] for feature in range(X.shape[0])])
        exponents = np.exp(-(np.power(X - means, 2) / (2 * np.power(stds, 2))))
        return np.prod(exponents / (sqrt_2pi * stds))

    
    def predict(self, T):
        predList = []
        log_Wprob = {label: np.log(self.__Wprob[label]) for label in self.__Wprob}
        for sample in T:
            likelihoods = {
                label: np.sum(np.log(self.__likelihood(sample, label)))
                for label in self.__Wprob
            }
            # Calculate posterior probabilities
            posteriors = {
                label: likelihoods[label] + log_Wprob[label]
                for label in self.__Wprob
            }
            # Predict the class with the highest posterior probability
            predicted_class = max(posteriors, key=posteriors.get)
            predList.append(predicted_class)
        return predList


if __name__=="__main__":
    LE = LabelEncoder()
    tennis_data = pd.read_csv("PlayTennis.csv")
    
    # Convert categorical data to numerical values
    # Assuming 'Outlook', 'Temperature', 'Humidity', 'Wind' are the feature columns
    # and 'target' is the target column
    features_list=['Outlook','Temperature','Wind','Humidity','target']
    for f in features_list:
        tennis_data[f]=LE.fit_transform(tennis_data[f])
    
    print(tennis_data.head(14))
    features = tennis_data.drop('target', axis=1)
    target = tennis_data['target']
    
    # Fit the model
    model = NaiveBayes()
    model.fit(features.values, target)
    
    test = np.array([[2, 0, 0, 0]])  # This should match the encoded feature columns
    result = 'Yes Play' if model.predict(test) == 0 else 'No Play'
    print(result)



