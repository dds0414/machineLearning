# -*- coding: utf-8 -*-
from sklearn.feature_extraction import DictVectorizer
import csv
from sklearn import preprocessing
from sklearn import tree

Data = open(r'tree.csv')
reader = csv.reader(Data)
headers = reader.next()


featrueList = []
lableList = []
for row in reader:
    lableList.append(row[len(row)-1])
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]

    featrueList.append(rowDict)

vec = DictVectorizer()
dummyX = vec.fit_transform(featrueList).toarray()

lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(lableList)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(dummyX, dummyY)

oneRowX = dummyX[0, :]
newRowX = [[]]
newRowX[0] = list(oneRowX)
newRowX[0][0] = 1
newRowX[0][2] = 0

predictedY = clf.predict(newRowX)

print predictedY


