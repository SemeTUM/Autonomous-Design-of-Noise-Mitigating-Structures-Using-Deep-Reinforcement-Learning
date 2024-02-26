import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import matplotlib.font_manager as fm

#Figure 4a
#ddqncnnabs>>random environment>>DDQNCnn>>DDQNCNNabs100.npy
#ddqncnnabs>>rule environment>>DDQNCnn>>DDQNCNNabsum100.npy
fig = plt.figure(figsize=(50,40), dpi=300)
ax = fig.add_subplot(111)
plt.plot(np.arange(len(ddqncnnabs)), ddqncnnabs, linewidth=6,color='red', alpha=0.6,label='Random selection')# 
plt.plot(pd.Series(ddqncnnabs).rolling(100).mean(), linewidth=16,color='red')
plt.plot(np.arange(len(ddqncnnabsrule)), ddqncnnabsrule, linewidth=6,color='green',alpha=0.3, label='Rule-based selection')
plt.plot(pd.Series(ddqncnnabsrule).rolling(100).mean(), linewidth=16, color='green')
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

#Figure 4b
fig = plt.figure(figsize=(50,30), dpi=300)
ax = fig.add_subplot(111)
plt.plot(freq, ddqncnnrule[3], linewidth=20, label='#3')# 
plt.plot(freq, ddqncnnrule[391], linewidth=20, label='#391')#
plt.plot(freq, ddqncnnrule[403], linewidth=20,label='#403')#
plt.plot(freq, ddqncnnrule[1725], linewidth=20,label='#1725')#

font_properties = {'family': 'serif', 'weight': 'bold', 'size': 90}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])
plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Frequency", fontproperties=font)
plt.ylabel(r"$\alpha$", fontproperties=font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 80}
plt.legend(prop=fm.FontProperties(**legend_font), loc='lower right')
plt.show()


#Figure 4c
fig = plt.figure(figsize=(50,30), dpi=300)
ax = fig.add_subplot(111)
plt.plot(freq, ddqncnnrule[3], linewidth=12, label='#3')# 
plt.plot(freq, ddqncnnrule[391], linewidth=12, label='#391')#
plt.plot(freq, ddqncnnrule[403], linewidth=12,label='#403')#
plt.plot(freq, ddqncnnrule[1725], linewidth=12,label='#1725')#

font_properties = {'family': 'serif', 'weight': 'bold', 'size': 90}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])
plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Frequency", fontproperties=font)
plt.ylabel(r"$\alpha$", fontproperties=font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 80}
plt.legend(prop=fm.FontProperties(**legend_font), loc='lower right')
plt.show()

#Figure 4d
#Genetic algorithm/computed via COMSOL Matlab Livelink