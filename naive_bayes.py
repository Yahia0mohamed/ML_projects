'''
equation is p(w1|x)=p(x|w1)*p(w1)
equation is p(w2|x)=p(x|w2)*p(w2)
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class NaiveBayes:
    def __init__(self):
        self.__x_train = None
        self.__y_train = None
        self.__Wprob = None
        self.__XWprob = None
        self.__features_list = ['Outlook', 'Temperature', 'Wind', 'Humidity']


    def fit(self, data, target):
        self.__x_train = data
        self.__y_train = target
        self.__Wprob = self.__aprior()
        self.__XWprob=self.__likelihood()


    def __aprior(self):
        return {label: np.mean(self.__y_train == label) for label in set(self.__y_train)}

    def __likelihood(self):
        likelihood = {
            label: {
                feature_val: {
                    target_val: sum(1 for x, y in zip(self.__x_train[label], self.__y_train) if x == feature_val and y == target_val) / self.__y_train.value_counts()[target_val]
                    for target_val in set(self.__y_train)
                }
                for feature_val in set(self.__x_train[label])
            }
            for label in self.__features_list
        }
        return likelihood
    
    
    def predict(self, T):
        predList = []
        for sample in T:
            likelihoods = {
                label: np.prod([self.__XWprob[feature_name][feature_val][label] for feature_name,feature_val in zip(self.__features_list,sample)])
                for label in self.__Wprob
            }
            # Calculate posterior probabilities
            posteriors = {
                label: self.__Wprob[label] * likelihoods[label]
                for label in self.__Wprob
            }
            # Predict the class with the highest posterior probability
            predicted_class = max(posteriors, key=posteriors.get)
            predList.append(predicted_class)
        return predList




if __name__=="__main__":
    LE = LabelEncoder()
    tennis_data = pd.read_csv("PlayTennis.csv")

    features_list=['Outlook','Temperature','Wind','Humidity','target']
    for f in features_list:
        tennis_data[f]=LE.fit_transform(tennis_data[f])

    features = tennis_data.drop('target', axis=1)
    target = tennis_data['target']
    
    # Fit the model
    model = NaiveBayes()
    model.fit(features, target)
    
    test = np.array([[2, 0, 0, 0]])  # This should match the encoded feature columns
    result = 'Yes Play' if model.predict(test) == 0 else 'No Play'
    print(result)



