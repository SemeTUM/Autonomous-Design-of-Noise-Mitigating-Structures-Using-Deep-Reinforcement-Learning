import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import matplotlib.font_manager as fm
#Figure 5a
#ddqncnnabs= np.load('RefProb_RuleEnvDDQNCNNeps100 PAPER3/Results/DDQNCNNabsum.npy')
#ddqncnnrule1000= np.load('RefProb_RuleEnvDDQNCNNeps1000 PAPER3/Results/DDQNCNNabsum.npy')
fig = plt.figure(figsize=(50,20), dpi=300)
ax = fig.add_subplot(111)
plt.plot(np.arange(len(ddqncnnabs)), ddqncnnabs, linewidth=6,color='red', alpha=0.6,label='DDQN CNN(epsilon decay=100)')# 
#plt.plot(pd.Series(absum).rolling(100).mean(), linewidth=12, label='EPS DECAY')
plt.plot(pd.Series(ddqncnnabs).rolling(100).mean(), linewidth=16,color='red')
plt.plot(np.arange(len(ddqncnnrule1000)), ddqncnnrule1000, linewidth=6,color='lightcoral',alpha=0.3, label='DDQN CNN(epsilon decay=1000)')
plt.plot(pd.Series(ddqncnnrule1000).rolling(100).mean(), linewidth=16, color='lightcoral')
font_properties = {'family': 'serif', 'weight': 'bold', 'size': 90}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])
plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Episodes", fontproperties=font)
plt.ylabel("Absorption Sum", fontproperties=font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 80}
plt.legend(prop=fm.FontProperties(**legend_font), loc='lower right')
plt.show()

#Figure 5b
EPS_START = 0.99       # starting value of epsilon
EPS_END = 0.01         # Ending value of epsilon
EPS_DECAY = [100,1000]#100

epsilon_by_epsiode = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY[0])
epsilon_by_epsiode1 = lambda frame_idx: EPS_END + (EPS_START - EPS_END) * math.exp(-1. * frame_idx /EPS_DECAY[1])

epsilon_values100 = []
epsilon_values1000 = []
for frame_idx in range (2000):
    epsilon = epsilon_by_epsiode(frame_idx)
    epsilon1 = epsilon_by_epsiode1(frame_idx)
    epsilon_values100.append(epsilon)
    epsilon_values1000.append(epsilon1)

fig = plt.figure(figsize=(50,20), dpi=300)
ax = fig.add_subplot(111)
plt.plot(np.arange(len(epsilon_values)), epsilon_values, linewidth=20, label='epsilon decay=100')# 
plt.plot(np.arange(len(epsilon_values1)), epsilon_values1, linewidth=20, label='epsilon decay=1000')#

font_properties = {'family': 'serif', 'weight': 'bold', 'size': 90}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])
plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Episodes", fontproperties=font)
plt.ylabel(r"$\epsilon$", fontproperties=font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 80}
plt.legend(prop=fm.FontProperties(**legend_font), loc='upper right')
plt.show()

#Figure 5c
#24, 180, 225, 351
#load DDQNCNNstates.npy as tl1000stat
ddqnstat100_image = np.rot90(np.reshape(tl1000stat[328], (20, 20)), k=1)

# Plot the image
plt.imshow(ddqnstat100_image, cmap='gray', interpolation='nearest')
plt.show()


#Figure 5d
#ddqncnnruleabs1000=np.load('Results/DDQNCNNstates.npy')
fig = plt.figure(figsize=(50,30), dpi=300)
ax = fig.add_subplot(111)
plt.plot(freq, ddqncnnruleabs1000[24], linewidth=20, label='#24')# 
plt.plot(freq, ddqncnnruleabs1000[180], linewidth=20, label='#180')#
plt.plot(freq, ddqncnnruleabs1000[225], linewidth=20,label='#225')#
plt.plot(freq, ddqncnnruleabs1000[351], linewidth=20,label='#351')#

font_properties = {'family': 'serif', 'weight': 'bold', 'size': 90}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])
plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Frequency (Hz)", fontproperties=font)
plt.ylabel(r"$\alpha$", fontproperties=font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

# Show the plot
legend_font = {'family': 'serif', 'weight': 'bold', 'size': 80}
plt.legend(prop=fm.FontProperties(**legend_font), loc='upper left')
plt.show()



