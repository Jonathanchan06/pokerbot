import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


test = pd.read_csv("C:\\Users\\Asus\\Desktop\\poker_data\\poker+hand\\poker-hand-testing.data", header=None)
train = pd.read_csv("C:\\Users\\Asus\\Desktop\\poker_data\\poker+hand\\poker-hand-training-true.data", header=None)

X_train = train.iloc[:, :-1]   #5 cards
y_train = train.iloc[:, -1]    #class

X_test = test.iloc[:, :-1]   # all columns except last
y_test = test.iloc[:, -1]    # last column

rf = RandomForestClassifier(
    n_estimators=300,
    class_weight={0:1, 1:5, 2:5, 3:5, 4:10, 5:10, 6:15, 7:30, 8:50, 9:100},  # handle imbalance
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# class_counts = y_train.value_counts().sort_index()            ###Visualization of hands
#
# plt.figure(figsize=(10,5))
# sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
#
# plt.title("Poker Hand Class Distribution")
# plt.xlabel("Hand Class (0=Nothing ... 9=Royal Flush)")
# plt.ylabel("Number of Hands")
# plt.show()

# ranks = X_train.iloc[:, 1::2].values.flatten()  ###Visualization of ranks
# plt.figure(figsize=(10,5))
# sns.histplot(ranks, bins=13, discrete=True, kde=False)
# plt.title("Distribution of Card Ranks")
# plt.xlabel("Rank (1=Ace, 2..13=King)")
# plt.ylabel("Count")
# plt.show()
