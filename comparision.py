import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


j = {
    "CNN": [99.4,98.1,99.92],
    "SVM": [98.8,98.9,84.3]
}

print(j)

df = pd.DataFrame(j)
print(df.head())
index = np.arange(3)
bar_width = 0.35

fig, ax = plt.subplots()
cnn = ax.bar(index, df["CNN"].values, bar_width,
             label="CNN (VGG-16)")

svm = ax.bar(index + bar_width, df["SVM"].values,
             bar_width, label="SVM + ELM")

# ax.set_xlabel('Category')
ax.set_ylabel('Accuracies')
ax.set_title('Training Accuracy Comparision')
ax.set_xticks(index + bar_width / 2)
# ax.set_yticks([0,0.5,0.9,0.91,0.92,0.93,0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1,1.5])

ax.set_yticks([0,25,50,75,99,150])
ax.set_xticklabels(["CASIA-I", "CASIA-II", "DDSM"])
ax.legend()

for rect in cnn:
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
            '{:0.2f}'.format(height) % height,
            ha='center', va='bottom')


for rect in svm\
        :
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
            '{:0.2f}'.format(height) % height,
            ha='center', va='bottom')


plt.show()
