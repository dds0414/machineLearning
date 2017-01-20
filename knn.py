# -*- coding: utf-8 -*-
from sklearn import neighbors
from sklearn import datasets

knn = neighbors.KNeighborsClassifier()
iris = datasets.load_iris()

knn.fit(iris.data, iris.target)

print iris.target_names[knn.predict([[0.1, 0.2, 0.3, 0.4]])[0]]
