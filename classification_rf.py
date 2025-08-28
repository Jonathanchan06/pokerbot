from statistics import median

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

test = pd.read_csv("C:\\Users\\Asus\\Desktop\\poker_data\\poker+hand\\poker-hand-testing.data", header=None)
train = pd.read_csv("C:\\Users\\Asus\\Desktop\\poker_data\\poker+hand\\poker-hand-training-true.data", header=None)

X_train = train.iloc[:, :-1]   #5 cards
y_train = train.iloc[:, -1]    #class

X_test = test.iloc[:, :-1]   # all columns except last
y_test = test.iloc[:, -1]    # last column

train_count = Counter(y_train)
c0_count = train_count.get(0, 0) #counter of c0 classes

minority_count = [c_count for c, c_count in train_count.items() if c != 0 ] #list of frequency of each class except 0
mean_minority = int(pd.Series(minority_count).mean()) #take the median of minority_count
target_c_major = 8 * mean_minority


#Undersampling
undersample_map = {0: target_c_major}

#Oversampling
q80 = int(pd.Series(minority_count).quantile(0.80)) if minority_count else 500
target_minor = max(q80, 1800) #2000 max selected as similar to c0 freq but not too high
oversample_map = {c: target_minor for c in train_count.keys() if c != 0 and c != 1}


pipe = Pipeline(steps=[
    ("under", RandomUnderSampler(sampling_strategy=undersample_map, random_state=42)),
    ("over",  RandomOverSampler(sampling_strategy=oversample_map, random_state=42)),
    ("rf",    RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        max_samples=0.9,           # subsample each tree for speed/robustness
        class_weight=None,         # usually OFF when using over/under-sampling
        n_jobs=-1,
        random_state=42
    ))
])

pipe.fit(X_train, y_train)

#evaluation
y_pred = pipe.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
