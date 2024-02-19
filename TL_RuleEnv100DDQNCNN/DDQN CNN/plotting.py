import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import matplotlib.font_manager as fm

#Figure 6a
#tlsum= np.load('TL_RuleEnv100DDQNCNN/Results/DDQNCNNTLsum.npy')
#tl1000= np.load('TL_RuleEnv1000DDQNCNN/Results/DDQNCNNTLsum.npy')
fig = plt.figure(figsize=(50,20), dpi=300)
ax = fig.add_subplot(111)
plt.plot(np.arange(len(tlsum)), tlsum, linewidth=6,color='red', alpha=0.6,label='TL(epsilon decay=100)')# 
plt.plot(pd.Series(tlsum).rolling(100).mean(), linewidth=16,color='red')
plt.plot(np.arange(len(tl1000)), tl1000, linewidth=6,color='lightcoral',alpha=0.3, label='TL(epsilon decay=1000)')
plt.plot(pd.Series(tl1000).rolling(100).mean(), linewidth=16, color='lightcoral')
font_properties = {'family': 'serif', 'weight': 'bold', 'size': 90}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])
plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Episodes", fontproperties=font)
plt.ylabel("TL (dB)", fontproperties=font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 80}
plt.legend(prop=fm.FontProperties(**legend_font), loc='upper left')
plt.show()

#Figure 6b
#ddqnTLactrule1000=np.load('Src/SourceCode/TL_RuleEnv1000DDQNCNN/DDQN CNN/TL_RuleEnv1000DDQNCNN2/DDQNCNNactions.npy')
#ddqnTLactrule100=np.load('Src/SourceCode/TL_RuleEnv100DDQNCNN/DDQN CNN/TL_RuleEnv100DDQNCNN2/DDQNCNNactions.npy')

fig = plt.figure(figsize=(50,20), dpi=300)
ax = fig.add_subplot(111)
plt.hist(ddqnTLactrule100, bins=np.arange(min(ddqnTLactrule100), max(ddqnTLactrule100) + 1), color='red', edgecolor='red',alpha=0.5, linewidth=14, label='TL(epsilon decay=100)')#linewidth 1.5
plt.hist(ddqnTLactrule1000, bins=np.arange(min(ddqnTLactrule1000), max(ddqnTLactrule1000) + 1), color='green', edgecolor='green', linewidth=14,alpha=0.8, label='TL(epsilon decay=1000)')#linewidth 1.5
font_properties = {'family': 'serif', 'weight': 'bold', 'size': 90}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])
plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("actions", fontproperties=font)
plt.ylabel("Frequency of selection", fontproperties=font)
ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

# Show the plot
legend_font = {'family': 'serif', 'weight': 'bold', 'size': 80}
plt.legend(prop=fm.FontProperties(**legend_font), loc='upper left')
plt.show()

#Figure 6c
#73,511,552, 1169
#tl100stat= np.load('TL_RuleEnv100DDQNCNN/Results/DDQNCNNstates.npy')
ddqnstat100_image = np.rot90(np.reshape(tl100stat[73], (20, 20)), k=1)
# Create a custom colormap for -2 and 2
cmap = plt.cm.colors.ListedColormap(['white', 'black'])
# Plot the image
plt.imshow(ddqnstat100_image, cmap=cmap, interpolation='nearest')
plt.show()





#Figure 6d
#ddqnTLstatrule100= np.load('TL_RuleEnv100DDQNCNN/Results/DDQNCNNTL.npy')
fig = plt.figure(figsize=(50,30), dpi=300)
ax = fig.add_subplot(111)
plt.plot(freq, ddqnTLstatrule100[73], linewidth=20, label='#73')# 
plt.plot(freq, ddqnTLstatrule100[511], linewidth=20, label='#511')#
plt.plot(freq, ddqnTLstatrule100[552], linewidth=20,label='#552')#
plt.plot(freq, ddqnTLstatrule100[1169], linewidth=20,label='#1169')#

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
plt.show()