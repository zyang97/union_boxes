import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Chair
path_aux = 'C:\\Users\\zyang\\Downloads\\Jan28_13-31-50_DESKTOP-DKPHVDU.csv' # aux loss
path_coverage = 'C:\\Users\\zyang\\Downloads\\Jan28_13-31-50_DESKTOP-DKPHVDU (2).csv'
path_consistency = 'C:\\Users\\zyang\\Downloads\\Jan28_13-31-50_DESKTOP-DKPHVDU (1).csv'

# Aero
# path_aux = 'C:\\Users\\zyang\\Downloads\\Feb13_18-24-14_DESKTOP-DKPHVDU.csv' # aux loss
# path_coverage = 'C:\\Users\\zyang\\Downloads\\Feb13_18-24-14_DESKTOP-DKPHVDU (2).csv'
# path_consistency = 'C:\\Users\\zyang\\Downloads\\Feb13_18-24-14_DESKTOP-DKPHVDU (1).csv'

df_aux = pd.read_csv(path_aux)
df_coverage = pd.read_csv(path_coverage)
df_consistency = pd.read_csv(path_consistency)

plt.plot(df_aux['Step'], df_aux['Value'], label= 'aux_loss')
plt.plot(df_coverage['Step'], df_coverage['Value'], label= 'coverage_loss')
plt.plot(df_consistency['Step'], df_consistency['Value'], label= 'consistency_loss')
plt.title('Training Loss: Chair')
plt.legend()
