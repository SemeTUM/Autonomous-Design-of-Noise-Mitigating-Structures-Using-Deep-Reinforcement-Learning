import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import matplotlib.font_manager as fm


#Figure 7a plotting
#52, 90, 106, 324, 328
#load DDQNCNNstates.npy as tl1000stat
ddqnstat100_image = np.rot90(np.reshape(tl1000stat[328], (20, 20)), k=1)

# Create a custom colormap for -2 and 2
cmap = plt.cm.colors.ListedColormap(['white', 'black'])

# Plot the image
plt.imshow(ddqnstat100_image, cmap=cmap, interpolation='nearest')
plt.show()
#______________________________________________________________________________________
#plotting Figure 7b

freq= np.linspace(300, 3000,28)# Frequency 
fig = plt.figure(figsize=(50,30), dpi=300)
ax = fig.add_subplot(111)
plt.plot(freq, tl1000tl[52], linewidth=20, label='#52')# 
plt.plot(freq, tl1000tl[90], linewidth=20, label='#90')#
plt.plot(freq, tl1000tl[106], linewidth=20,label='#106')#
plt.plot(freq, tl1000tl[324], linewidth=20,label='#324')#
plt.plot(freq, tl1000tl[328], linewidth=20,label='#328')#

font_properties = {'family': 'serif', 'weight': 'bold', 'size': 90}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])
plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Frequency (Hz)", fontproperties=font)
plt.ylabel("TL (dB)", fontproperties=font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

# Show the plot
legend_font = {'family': 'serif', 'weight': 'bold', 'size': 80}
plt.legend(prop=fm.FontProperties(**legend_font), loc='upper left')