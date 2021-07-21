import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from helper import find_csv
grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)
ax1 = plt.subplot(grid[0,0])
ax2= plt.subplot(grid[0,1])
ax3= plt.subplot(grid[1,:])
for i in find_csv():
    df = pd.read_csv(i,header=None)
    df_forward = df[:int(len(df)/2)]
    forward_peak = df_forward[0].iloc[df_forward[1].idxmin()]
    df_backward = df[int(len(df)/2):]
    df_backward = df_backward.reset_index(drop=True)   #Use drop to discard the old index 
    backward_peak = df_backward[0].iloc[df_backward[1].idxmax()]
    #ax1.axvline(x=forward_peak,alpha=0.7)
    #ax2.axvline(x=backward_peak,alpha=0.7)

    df_forward.plot(x=0,y=1, ax=ax1)
    df_backward.plot(x=0,y=1,ax=ax2)
    df.plot(x=0,y=1,ax=ax3)

ax1.legend(find_csv(),loc=0)
ax2.legend(find_csv(),loc=0)
ax3.legend(find_csv(),loc=0)
ax1.set_title('Forward scan')
ax2.set_title('Reverse Scan')
ax3.set_title('CV')
plt.show()
