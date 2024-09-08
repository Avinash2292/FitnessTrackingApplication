import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor
# Load the DataFrame from the pickle file
df = pd.read_pickle("/home/avinash/Downloads/data-science-template-main/data/interim/01_data_processed.pkl")
outlier_columns = list(df.columns[:6])
# Set Matplotlib style and figure options
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
# Create a boxplot for the "acc_x" column
df[["acc_x","label"]].boxplot(by="label",figsize = (20,10))
df[outlier_columns[:3] + ["label"]].boxplot(by="label" , figsize=(20,10),layout=(1,3))
df[outlier_columns[3:] + ["label"]].boxplot(by="label" , figsize=(20,10),layout=(1,3))
def plot_binary_outliers(dataset, col, outlier_col, reset_index):
 
    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")
    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()
    plt.xlabel("samples")
    plt.ylabel("value") 
    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "+",
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "r+",
    ) 
    plt.legend(
        ["outlier " + col, "no outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()
# Print the DataFrame 
def mark_outliers_iqr(dataset, col):
    
    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset 
col = "acc_x"
dataset = mark_outliers_iqr(df,col)
plot_binary_outliers(dataset=dataset,col=col,outlier_col=col+"_outlier",reset_index=False)
plot_binary_outliers(dataset=dataset,col=col,outlier_col=col+"_outlier",reset_index=True)
for col in outlier_columns:
    dataset = mark_outliers_iqr(df,col)
    plot_binary_outliers(dataset=dataset,col=col,outlier_col=col+"_outlier",reset_index=True)
# Chauvenents Criteron (Distribution method)
#Check for normal distribution
df[outlier_columns[:3] + ["label"]].plot.hist(by="label" , figsize=(20,20),layout=(13,3))
df[outlier_columns[3:] + ["label"]].plot.hist(by="label" , figsize=(20,20),layout=(3,3))
def mark_outliers_chauvenet(dataset, col, C=2):
    
    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

dataset = mark_outliers_chauvenet(df,col)
for col in outlier_columns:
    plot_binary_outliers(dataset=dataset,col=col,outlier_col=col+"_outlier",reset_index=True)
def mark_outliers_lof(dataset, columns, n=20):
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    dataset["outlier_lof"] = outliers == -1
    return dataset, outliers, X_scores
dataset, outliers, X_scores = mark_outliers_lof(df,col)
for col in outlier_columns:
    plot_binary_outliers(dataset=dataset,col=col,outlier_col="outlier_lof",reset_index=True)



  
