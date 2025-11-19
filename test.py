import pandas as pd

data = pd.read_csv("output/summary.csv")


print(data["class"].value_counts())


# classes = ["AU", "TP"]
# classcounts = {cls: 0 for cls in classes}

# with open(data, "r") as f:
#     next(f)  # skip header
#     for line in f:
#         label = line.strip().split(",")[-1]
#         if label in classcounts:
#             classcounts[label] += 1

# print(classcounts)
