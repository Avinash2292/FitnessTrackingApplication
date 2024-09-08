import pandas as pd
from glob import glob
files = glob("data/raw/MetaMotion/*.csv")

data_path = "../../data/raw/MetaMotion"
f = files[0]
participant = f.split("-")[0].replace(data_path,"")
label = f.split("-")[1]
category = f.split("-")[2].rstrip("2")
df = pd.read_csv(f)
df["particiapant"] = participant
df["label"] = label
df["category"] = category

# Read all files
files = glob("data/raw/MetaMotion/*.csv")

def read_data_from_files(files):
    
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()
    acc_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split("-")[0].replace(data_path,"")
        label = f.split("-")[1]
        category = f.split("-")[2].rstrip("2").rstrip("_MetaWear_2019")
        
        df = pd.read_csv(f)
        df["particiapant"] = participant
        df["label"] = label
        df["category"] = category
        
        if "Accelerometer" in f:
            df["set"] = acc_set
            acc_set+=1
            acc_df = pd.concat([acc_df, df])
        
        if "Gyroscope" in f:
            df["set"] = gyr_set
            gyr_set+=1
            gyr_df = pd.concat([gyr_df, df])

    
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")

    acc_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)
    gyr_df.drop(columns=["epoch (ms)", "time (01:00)", "elapsed (s)"], inplace=True)

    #print(acc_df)
    #print(gyr_df)
    
    return acc_df,gyr_df

acc_df , gyr_df = read_data_from_files(files)
print(acc_df)


# Merging the datasets

data_merged = pd.concat([acc_df.iloc[:,:3],gyr_df],axis=1)

#print(data_merged.head(4))

data_merged.columns = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "participant",
    "label",
    "category",
    "set",
]
print(data_merged)


# Resample data (frequency conversion)

sampling = {
    "acc_x" : "mean",
    "acc_y" : "mean",
    "acc_z" : "mean",
    "gyr_x" : "mean",
    "gyr_y" : "mean",
    "gyr_z" : "mean",
    "participant" : "last",
    "label" : "last",
    "category" : "last",
    "set" : "last",
    
}

 
print(data_merged[:1000].resample(rule="200ms").apply(sampling))

days = [g for n,g in data_merged.groupby(pd.Grouper(freq="D"))]
#print(days[-1])

data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])
print(data_resampled)

print(data_resampled.info())
data_resampled["set"] = data_resampled["set"].astype("int")
print(data_resampled)



# Export Dataset

# Specify the path where you want to save the pickle file
file_path = "/home/avinash/Downloads/data-science-template-main/data/interim/01_data_processed.pkl"

# Save the DataFrame to a pickle file
data_resampled.to_pickle(file_path)
print(f"DataFrame saved to {file_path}")
 


