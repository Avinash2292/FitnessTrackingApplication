import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import display
import seaborn as sns
# Load data

df = pd.read_pickle("/home/avinash/Downloads/data-science-template-main/data/interim/01_data_processed.pkl")
print(df)

# Plot single column
set_df = df[df["set"] == 1]
plt.plot(set_df["acc_y"])
#plt.plot(set_df["acc_y"].reset_index(drop=True))
#plt.show()
# Plot all exercise

for label in df["label"].unique():
    subset = df[df["label"] == label]
    #print(display(subset.head(2)))
    fig, ax = plt.subplots()
    plt.plot(subset["acc_y"].reset_index(drop=True),label = label)
    plt.legend()
    plt.show()
    
for label in df["label"].unique():
    subset = df[df["label"] == label]
    #print(display(subset.head(2)))
    fig, ax = plt.subplots()
    plt.plot(subset[:100]["acc_y"].reset_index(drop=True),label = label)
    plt.legend()
    plt.show()
    
    
# Adjust plot settings

#mpl.style.use("seaborn-v0_8-deep")
mpl.rcParams["figure.figsize"] = (20,5)
mpl.rcParams["figure.dpi"] = 100



# Compare medium vs Heavy sets

category_df = df.query("label == 'squat'").reset_index()
#print(category_df)
fig,ax = plt.subplots()
category_df.groupby(["category"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()
plt.show()

# Compare Participants

participant_df = df.query("label == 'bench'").sort_values("participant").reset_index()
#print(participant_df)
fig,ax = plt.subplots()
participant_df.groupby(["participant"])["acc_y"].plot()
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.title("Compare Participants")
plt.legend()
plt.show()

# Plot multiple axis

label = "squat"
participant = "A"
all_axis_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()

fig,ax = plt.subplots()
all_axis_df[["acc_y","acc_y","acc_z"]].plot(ax =ax)
ax.set_ylabel("acc_y")
ax.set_xlabel("samples")
plt.legend()

# Create a loop to print all combinations per sensors
import matplotlib.pyplot as plt

# Assuming 'df' is the DataFrame and it has columns 'label', 'participant', 'acc_x', 'acc_y', 'acc_z'

labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        # Filter the DataFrame for the current label and participant
        all_axis_df = df.query(f"label == '{label}' and participant == '{participant}'").reset_index()
        
        if len(all_axis_df)>0:
            fig, ax = plt.subplots()
            
            # Plot acc_x, acc_y, and acc_z on the same axes
            all_axis_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax)
            
            ax.set_ylabel("Acceleration")
            ax.set_xlabel("Samples")
            
            plt.title(f"Label: {label}, Participant: {participant}")
            plt.legend(["acc_x", "acc_y", "acc_z"])
            
            # Show the figure
            plt.show()
            
            # Close the figure to free up memory
            plt.close(fig)


for label in labels:
    for participant in participants:
        # Filter the DataFrame for the current label and participant
        all_axis_df = df.query(f"label == '{label}' and participant == '{participant}'").reset_index()
        
        if len(all_axis_df)>0:
            fig, ax = plt.subplots()
            
            # Plot acc_x, acc_y, and acc_z on the same axes
            all_axis_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax)
            
            ax.set_ylabel("Gyroscope")
            ax.set_xlabel("Samples")
            
            plt.title(f"Label: {label}, Participant: {participant}")
            plt.legend(["gyr_x", "gyr_y", "gyr_z"])
            
            # Show the figure
            plt.show()
            
            # Close the figure to free up memory
            plt.close(fig)


import matplotlib.pyplot as plt
import os
labels = df["label"].unique()
participants = df["participant"].unique()

for label in labels:
    for participant in participants:
        combined_plot_df = df.query(f"label == '{label}' and participant == '{participant}'").reset_index(drop=True)
        
        if not combined_plot_df.empty:
            # Create subplots with shared x-axis
            fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(20, 15))

            # Plot accelerometer data
            combined_plot_df.plot(y=["acc_x", "acc_y", "acc_z"], ax=axs[0, 0])
            axs[0, 0].set_title(f"Accelerometer - Label: {label}, Participant: {participant}")
            axs[0, 0].set_ylabel("Acceleration")
            axs[0, 0].legend(["acc_x", "acc_y", "acc_z"], loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)

            combined_plot_df.plot(y=["acc_x", "acc_y", "acc_z"], ax=axs[1, 0])
            axs[1, 0].set_ylabel("Acceleration")
            axs[1, 0].legend(["acc_x", "acc_y", "acc_z"], loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            axs[1, 0].set_xlabel("Samples")

            # Plot gyroscope data
            combined_plot_df.plot(y=["gyr_x", "gyr_y", "gyr_z"], ax=axs[0, 1])
            axs[0, 1].set_title(f"gyrscope - Label: {label}, Participant: {participant}")
            axs[0, 1].set_ylabel("gyrscope")
            axs[0, 1].legend(["gyr_x", "gyr_y", "gyr_z"], loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)

            combined_plot_df.plot(y=["gyr_x", "gyr_y", "gyr_z"], ax=axs[1, 1])
            axs[1, 1].set_ylabel("gyrscope")
            axs[1, 1].legend(["gyr_x", "gyr_y", "gyr_z"], loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            axs[1, 1].set_xlabel("Samples")
            
            

            # Display the plot
            plt.tight_layout()
            plt.show()
