import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keras.utils import normalize
from keras.datasets import mnist
import ssl


#PCA + logistic regression

ssl._create_default_https_context = ssl._create_unverified_context

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = normalize(x_train.reshape(x_train.shape[0], -1))
x_test = normalize(x_test.reshape(x_test.shape[0], -1))


#logistic regression

x_train1 = x_train[(y_train >= 5)]
y_train1 = y_train[(y_train >= 5)]

x_train2 = x_train[(y_train < 5)]
y_train2 = y_train[(y_train < 5)]


log_reg_model1 = LogisticRegression()
log_reg_model1.fit(x_train1, y_train1)
log_reg_model2 = LogisticRegression()
log_reg_model2.fit(x_train2, y_train2)


y_pred_model1 = log_reg_model1.predict(x_test)
y_pred_model2 = log_reg_model2.predict(x_test)
accuracy_model1 = accuracy_score(y_test >= 5, y_pred_model1)
accuracy_model2 = accuracy_score(y_test < 5, y_pred_model2)

#PCA

pca = PCA(n_components=0.95)
pca.fit(x_train)
x_train3 = pca.transform(x_train)
x_test3 = pca.transform(x_test)
log_reg_model3 = LogisticRegression()
log_reg_model3.fit(x_train3, y_train)
y_pred_model3 = log_reg_model3.predict(x_test3)
accuracy_pca = accuracy_score(y_test, y_pred_model3)

print(f"Accuracy of logistic regression model:\n"
      f" model1 for y >= 5: {accuracy_model1}\n"
      f"model2 for y<5: {accuracy_model2} \n"
      f"model3 with PCA: {accuracy_pca} ")

#PCA дозволило побудувати більш точну модель. Моделі 1 та 2 показують дуже низькі
# результати, фільтрація датасету за якімось параметром, несе за собою втрату
# можлівисть моделі побудувати правильні закономірності і знижує точність моделі




