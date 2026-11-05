# Session 1: Machine Learning Foundations, Part 1

## Key concepts

**What is machine learning?**
- Regular programming: rules + data → answers.
- Machine learning: data + answers → rules (the **model**).
- Spam example: hand-written rules break on edge cases; a model given enough labeled examples picks up the patterns itself.

**Features and labels**
- **Features** = the measurable characteristics of each example (rooms, bathrooms, square footage, HOA fee, ...).
- **Label / target** = the answer we want to predict (house price).
- Numeric features can be used directly; categorical ones (house type) must be converted to numbers first.

**Types of learning**
- **Supervised:** data has labels (spam filter, house prices).
- **Unsupervised:** no labels; find hidden structure (grouping viewers by watch history).
- **Reinforcement:** an agent takes actions in an environment and learns from rewards.

**Regression vs. classification** (both are supervised)
- Regression predicts a **number** (price, temperature).
- Classification predicts a **category** (spam / not spam, fraud / not fraud).
- Predicting a price *range* is classification; predicting the actual price is regression.

**ML workflow**
data → features → model → training → evaluation → prediction.
Evaluate on data the model has never seen, otherwise the score is falsely good (like memorizing practice problems).

**EDA lessons from the California housing data**
- A correlation near 0 only means no *straight-line* relationship; it does not rule a feature out (location matters a lot, non-linearly).
- Correlation strength = the size of the number, ignoring sign (-0.14 is stronger than 0.11).
- Correlation is not causation.
- Outliers in "Ave..." columns can come from tiny denominators (very few households).
- Capped (censored) target values: the model can't learn true prices above the cap.
- Two features can overlap heavily (multicollinearity): rooms and bedrooms, 0.85.

## Useful code snippets

```python
# NumPy
import numpy as np
arr = np.array([850, 1200, 1500])
arr * 2                      # vectorized math on every element
arr[arr > 1000]              # boolean indexing
grid[1, 2]                   # [row, column], counting from 0
grid[:, 0]                   # all rows, column 0

# pandas
import pandas as pd
df.head(); df.info(); df.describe(); df.shape
df[["a", "b"]]                                   # several columns (double brackets)
df[(df["a"] >= 3) & (df["b"] < 400000)]          # & = and, | = or, parentheses needed
df.isna().sum()                                  # missing values per column
df.duplicated().sum()                            # duplicate rows
df.groupby("bedrooms")["price"].mean()
df.corr(numeric_only=True)

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.histplot(df["MedHouseVal"], bins=50)
sns.scatterplot(data=df, x="MedInc", y="MedHouseVal", alpha=0.2)
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.savefig("location-vs-price.png", dpi=150, bbox_inches="tight")  # BEFORE plt.show()
plt.show()

# Loading the project data
from sklearn.datasets import fetch_california_housing
df = fetch_california_housing(as_frame=True).frame
```

## Common mistakes to remember
- `>` vs `>=` when a condition says "at least".
- Use `&` / `|`, not `and` / `or`, and wrap each condition in parentheses.
- Most pandas methods return a new table; save the result (`df = df.dropna()`).
- Run notebook cells top to bottom; `NameError: name 'df' is not defined` usually means the loading cell hasn't run.
- Skew is named by the direction of the long tail (house prices are right-skewed).

## Open questions
- How should outliers and the 5.0 price cap be handled before modeling?
- Which features really matter once they are combined in one model?
- How do I handle rows that look extreme (e.g., average rooms above 20)?
- How will I know whether a model's predictions can be trusted? (Session 2)
