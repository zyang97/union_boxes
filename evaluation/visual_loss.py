import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

path = 'C:\\Users\\Administrator.DESKTOP-DKPHVDU\\Downloads\\Feb25_13-34-54_DESKTOP-DKPHVDUchair_multi_view_p10_n10 (1).csv'
path_no_aux = 'C:\\Users\\Administrator.DESKTOP-DKPHVDU\\Downloads\\Mar01_02-07-31_DESKTOP-DKPHVDUchair_multi_view_p10_n10_sep_no_aux (1).csv'

# path_consistency = 'C:\\Users\\Administrator.DESKTOP-DKPHVDU\\Downloads\\Feb28_00-55-04_DESKTOP-DKPHVDUbench_multi_view_p10_n10_sep (1).csv'
# path_consistency_no_aux = 'C:\\Users\\Administrator.DESKTOP-DKPHVDU\\Downloads\\Feb28_13-58-13_DESKTOP-DKPHVDUbench_multi_view_p10_n10_sep_no_aux (1).csv'

df_consistency1 = pd.read_csv(path)
df_consistency2 = pd.read_csv(path_no_aux)

plt.plot(df_consistency1['Step'], df_consistency1['Value'], label= 'consistency_loss')
plt.plot(df_consistency2['Step'], df_consistency2['Value'], label= 'consistency_loss_no_aux')
plt.title('Consistency Loss: Chair')
plt.legend()
