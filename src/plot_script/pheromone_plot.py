import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pheromap_name = 'tcds_exp3'
data = np.load('../../tmp/{}.npy'.format(pheromap_name))

#print("data: {}".format(data))

df = pd.DataFrame(data=data[0:,0:],
                  index=range(np.shape(data)[0]),
                  columns=range(np.shape(data)[0]))

#print("shape: {}".format(np.shape(data)))
print("df: {}".format(df))

df_heatmap = sns.heatmap(data, annot=True, linewidths=.1, cmap="BuPu")
df_heatmap.set_title("Pheromone map", fontsize="15")
df_heatmap.set_xlabel("x", fontsize="15")
df_heatmap.set_ylabel("y", fontsize="15")
#sr.invert_yaxis()

folder_path = "/home/swn/catkin_ws/src/Turtlebot3_Pheromone/tmp"
df_heatmap.get_figure().savefig("{}/{}".format(folder_path, pheromap_name))