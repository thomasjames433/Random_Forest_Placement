import pandas as pd
import numpy as np
df=pd.read_csv("placement.csv")

df["PlacementStatus"]=(df["PlacementStatus"]=="Placed").astype(int)
df["ExtracurricularActivities"]=(df["ExtracurricularActivities"]=="Yes").astype(int)
df["PlacementTraining"]=(df["PlacementTraining"]=="Yes").astype(int)

X=df[[
    "CGPA",
    "Internships",
    "Projects",
    "Workshops/Certifications",
    "AptitudeTestScore",
    "SoftSkillsRating",
    "ExtracurricularActivities",
    "PlacementTraining",
    "SSC_Marks",
    "HSC_Marks"
]].values

y=df["PlacementStatus"].values

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=scaler.fit_transform(X)

from descision_tree import DescisionTree
from random_forest import RandomForest

clf=RandomForest() # to test descision tree clf=DescisionTree()
clf.fit(X,y)
predictions=clf.predict(X)


def accuracy(y_test,y_pred):
    return np.sum(y_test==y_pred)/len(y_test)

acc= accuracy(y, predictions)
print(acc)