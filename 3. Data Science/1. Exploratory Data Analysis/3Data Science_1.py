from audioop import avg
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("example.csv")
#Get the numerical data from "example.csv" file
print(dataset.describe())
print(dataset.info())
#Get the graphical data from "example.csv" file
sns.heatmap(dataset.corr(), annot=True, square=True)
plt.style.use("bmh")
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1, squeeze=True)

ax1.plot(dataset["x"], dataset["y1"], color="blue")
ax1.set_title("Y1 Graph")
ax1.set_xlabel("X")
ax1.set_ylabel("Y1")

ax2.plot(dataset["x"], dataset["y2"], color="red")
ax2.set_title("Y2 Graph")
ax2.set_xlabel("X")
ax2.set_ylabel("Y2")

ax3.plot(dataset["x"], dataset["z1"], color="green")
ax3.set_title("Z1 Graph")
ax3.set_xlabel("X")
ax3.set_ylabel("Z1")

ax4.plot(dataset["x"], dataset["z2"], color="orange")
ax4.set_title("Z2 Graph")
ax4.set_xlabel("X")
ax4.set_ylabel("Z2")

ax5.plot(dataset["x"], dataset["z3"], color="black")
ax5.set_title("Z3 Graph")
ax5.set_xlabel("X")
ax5.set_ylabel("Z3")

fig.set_size_inches(10,25)

fig.show()
input()