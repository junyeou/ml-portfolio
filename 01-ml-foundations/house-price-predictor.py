import numpy as np

sqft = np.array([850, 1200, 1500, 2000, 2400])

print(sqft)
print(sqft.shape)      # how many items, and in what layout
print(sqft.mean())     # average
print(sqft * 2)        # math applies to EVERY element at once
print(sqft[sqft > 1300])  # filtering with a condition