from ta import *
import pandas as pd

data = pd.read_csv("./ta_all.csv")
data = data.iloc[:len(data) // 8,:]
# print("Loaded CSV...")
# # #Add all the TA features
# #data = add_all_ta_features(data, "Open", "High", "Low", "Close","Volume", fillna=True)
# print("Added TA Features...")

data.to_csv("./ta_all_mini.csv", index=False)