import main
from main import y_train,X_train, y_test
import matplotlib.pyplot as plt
import seaborn as sns

class_counts = y_test.value_counts().sort_index()            ###Visualization of hands

plt.figure(figsize=(10,5))
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")

plt.title("Poker Hand Class Distribution")
plt.xlabel("Hand Class (0=Nothing ... 9=Royal Flush)")
plt.ylabel("Number of Hands")
plt.show()

# ranks = X_train.iloc[:, 1::2].values.flatten()  ###Visualization of ranks
# plt.figure(figsize=(10,5))
# sns.histplot(ranks, bins=13, discrete=True, kde=False)
# plt.title("Distribution of Card Ranks")
# plt.xlabel("Rank (1=Ace, 2..13=King)")
# plt.ylabel("Count")
# plt.show()
