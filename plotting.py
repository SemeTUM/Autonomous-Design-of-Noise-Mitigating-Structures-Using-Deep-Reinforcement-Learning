import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pandas as pd
import matplotlib.font_manager as fm


#Figure 7a plotting
#Figures for the transmission problem
ddqnTLstat1000= np.load('C:/Users/admin/Downloads/Src/SourceCode/TL_RuleEnv1000DDQNCNN/DDQN CNN/TL_RuleEnv1000DDQNCNN_Rew400/DDQNCNNstates.npy')
ddqnTLstat100= np.load('C:/Users/admin/Downloads/Src/SourceCode/TL_RuleEnv100DDQNCNN/DDQN CNN/TL_RuleEnv100DDQNCNN1/DDQNCNNTlsum.npy')

ddqnstat100_image = np.rot90(np.reshape(ddqnTLstat100[98], (20, 20)), k=1)

# Create a custom colormap for -2 and 2
cmap = plt.cm.colors.ListedColormap(['white', 'black'])

# Plot the image
plt.imshow(ddqnstat100_image, cmap=cmap, interpolation='nearest')
plt.show()
################
#Figure B2
DDQNbatch=np.load('C:/Users/admin/Downloads/Src/SourceCode/RefProb_RuleEnvDDQNCNN100/DDQN CNN/RefProb_RuleEnvDDQNCNN1BatchNorm/DDQNCNNabsum.npy')
DDQNWoutbatch=np.load('C:/Users/admin/Downloads/Src/SourceCode/RefProb_RuleEnvDDQNCNN100/DDQN CNN/RefProb_RuleEnvDDQNCNN1WoutBatchNorm/DDQNCNNabsum.npy')
fig = plt.figure(figsize=(50,40), dpi=300)
ax = fig.add_subplot(111)
#plt.plot(np.arange(len(DDQNWoutbatch)), DDQNWoutbatch, linewidth=6,color='red', alpha=0.6,label='Reward 1')# 
plt.plot(pd.Series(DDQNWoutbatch).rolling(100).mean(), linewidth=16,color='red', label='Without batch norm')
#plt.plot(np.arange(len(PERddqnabsSumR2)), PERddqnabsSumR2, linewidth=6,color='green', alpha=0.6,label='Reward 2')# 
plt.plot(pd.Series(DDQNbatch).rolling(100).mean(), linewidth=16,color='green', label='With batch norm')
font_properties = {'family': 'serif', 'weight': 'bold', 'size': 100}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])

# Adjust font size for labels only
label_font_properties = {'family': 'serif', 'weight': 'bold', 'size': 120}  # Adjust the size as needed
label_font = fm.FontProperties(family=label_font_properties['family'], weight=label_font_properties['weight'], size=label_font_properties['size'])

plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Episodes", fontproperties=label_font)
plt.ylabel("Absorption sum", fontproperties=label_font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 90}
plt.legend(prop=fm.FontProperties(**legend_font), loc='lower right')
plt.show()

##########Figure B2
ddqncnnruleOPTIm=np.load('C:/Users/admin/Downloads/Src/SourceCode/RefProb_RandEnvDDQNCNN100/DDQN CNN/RefProb_RandEnvDDQNCNN32646464640.005_100_10064Optim/DDQNCNNabsum.npy')
ddqncnnrand100=np.load('C:/Users/admin/Downloads/Src/SourceCode/RefProb_RandEnvDDQNCNN100/DDQN CNN/RefProb_RandEnvDDQNCNN100/DDQNCNNabsum.npy')
fig = plt.figure(figsize=(50,40), dpi=300)
ax = fig.add_subplot(111)
#plt.plot(np.arange(len(DDQNWoutbatch)), DDQNWoutbatch, linewidth=6,color='red', alpha=0.6,label='Reward 1')# 
plt.plot(pd.Series(ddqncnnruleOPTIm).rolling(100).mean(), linewidth=16,color='red', label='Optimum hyperparameters')
#plt.plot(np.arange(len(PERddqnabsSumR2)), PERddqnabsSumR2, linewidth=6,color='green', alpha=0.6,label='Reward 2')# 
plt.plot(pd.Series(ddqncnnrand100).rolling(100).mean(), linewidth=16,color='green', label='Final hyperparameters')
font_properties = {'family': 'serif', 'weight': 'bold', 'size': 100}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])

# Adjust font size 
label_font_properties = {'family': 'serif', 'weight': 'bold', 'size': 120}  # Adjust the size as needed
label_font = fm.FontProperties(family=label_font_properties['family'], weight=label_font_properties['weight'], size=label_font_properties['size'])

plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Episodes", fontproperties=label_font)
plt.ylabel("Absorption sum", fontproperties=label_font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 90}
plt.legend(prop=fm.FontProperties(**legend_font), loc='lower right')
plt.show()
###Figure 8b
ddqnTLact100= np.load('C:/Users/admin/Downloads/Src/SourceCode/TL_RuleEnv100DDQNCNN/DDQN CNN/TL_RuleEnv100DDQNCNN1/DDQNCNNactions.npy')
ddqnTLact1000= np.load('C:/Users/admin/Downloads/Src/SourceCode/TL_RuleEnv1000DDQNCNN/DDQN CNN/TL_RuleEnv1000DDQNCNN_Rew400/DDQNCNNactions.npy')
fig = plt.figure(figsize=(70,30), dpi=300)
ax = fig.add_subplot(111)
plt.hist(ddqnTLact100, bins=np.arange(min(ddqnTLact100), max(ddqnTLact100) + 1), color='red', edgecolor='red',alpha=0.8, linewidth=40, label='TL(epsilon decay=100)')#linewidth 1.5
plt.hist(ddqnTLact1000, bins=np.arange(min(ddqnTLact1000), max(ddqnTLact1000) + 1), color='green', edgecolor='green', linewidth=40,alpha=0.5, label='TL(epsilon decay=1000)')#linewidth 1.5
font_properties = {'family': 'serif', 'weight': 'bold', 'size': 100}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])

# Adjust font size for labels only
label_font_properties = {'family': 'serif', 'weight': 'bold', 'size': 120}  # Adjust the size as needed
label_font = fm.FontProperties(family=label_font_properties['family'], weight=label_font_properties['weight'], size=label_font_properties['size'])

plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("actions", fontproperties=label_font)
plt.ylabel("Frequency of selection", fontproperties=label_font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 90}
plt.legend(prop=fm.FontProperties(**legend_font), loc='upper left')
plt.show()
####Figure 8b
ddqnTLsum100= np.load('C:/Users/admin/Downloads/Src/SourceCode/TL_RuleEnv100DDQNCNN/DDQN CNN/TL_RuleEnv100DDQNCNN1/DDQNCNNTlsum.npy')
ddqnTLsum1000= np.load('C:/Users/admin/Downloads/Src/SourceCode/TL_RuleEnv1000DDQNCNN/DDQN CNN/TL_RuleEnv1000DDQNCNN_Rew400/DDQNCNNabsum.npy')
ddqnTLact100= np.load('C:/Users/admin/Downloads/Src/SourceCode/TL_RuleEnv100DDQNCNN/DDQN CNN/TL_RuleEnv100DDQNCNN1/DDQNCNNactions.npy')
ddqnTLact1000= np.load('C:/Users/admin/Downloads/Src/SourceCode/TL_RuleEnv1000DDQNCNN/DDQN CNN/TL_RuleEnv1000DDQNCNN_Rew400/DDQNCNNactions.npy')

fig=plt.figure(figsize=(70,30), dpi=300)
scatter1=plt.scatter(np.arange(len(ddqnTLsum100)), ddqnTLact100, c=ddqnTLsum100,cmap='PuRd', s=700, alpha=0.9, label='epsilon decay(100)',  edgecolors='none')
scatter2=plt.scatter(np.arange(len(ddqnTLsum1000)), ddqnTLact1000, c=ddqnTLsum1000,cmap='Greens', s=700, alpha=0.9, label='epsilon decay(1000)',  edgecolors='none')

cbar1 = plt.colorbar(scatter1, orientation='vertical', label='epsilon decay(100)', fraction=0.04)
cbar2 = plt.colorbar(scatter2, orientation='vertical', label='epsilon decay(1000)',fraction=0.04)
cbar1.ax.set_ylabel('Magnitude z1', fontsize=60, labelpad=20)
cbar2.ax.set_ylabel('Magnitude z2', fontsize=60, labelpad=20)

# Set the colorbar font size
cbar1.ax.tick_params(labelsize=60)
cbar2.ax.tick_params(labelsize=60)

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
plt.legend(prop=fm.FontProperties(**legend_font), loc='lower right')

plt.show()

###Figure 7d
freq= np.linspace(300,3000,28)
ddqnTL100= np.load('C:/Users/admin/Downloads/Src/SourceCode/TL_RuleEnv100DDQNCNN/DDQN CNN/TL_RuleEnv100DDQNCNN1/DDQNCNNTL.npy')
ddqnTL1000= np.load('C:/Users/admin/Downloads/Src/SourceCode/TL_RuleEnv1000DDQNCNN/DDQN CNN/TL_RuleEnv1000DDQNCNN_Rew400/DDQNCNNabsorption.npy')

fig = plt.figure(figsize=(50,30), dpi=300)
ax = fig.add_subplot(111)
plt.plot(freq,ddqnTL100[73], linewidth=20, label='#73')# 
plt.plot(freq, ddqnTL100[136], linewidth=20, label='#136')#
plt.plot(freq, ddqnTL100[238], linewidth=20,label='#238')#
plt.plot(freq, ddqnTL100[900], linewidth=20,label='#900')#

font_properties = {'family': 'serif', 'weight': 'bold', 'size': 100}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])

label_font_properties = {'family': 'serif', 'weight': 'bold', 'size': 120}  # Adjust the size as needed
label_font = fm.FontProperties(family=label_font_properties['family'], weight=label_font_properties['weight'], size=label_font_properties['size'])

plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Frequency (Hz)", fontproperties=label_font)
plt.ylabel("TL (dB)", fontproperties=label_font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 90}
plt.legend(prop=fm.FontProperties(**legend_font), loc='upper left')
plt.show()
####Figure 9b (close up)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import matplotlib.font_manager as fm
ddqnTL1000= np.load('C:/Users/admin/Downloads/Src/SourceCode/TL_RuleEnv1000DDQNCNN/DDQN CNN/TL_RuleEnv1000DDQNCNN_Rew400/DDQNCNNabsorption.npy')
fig = plt.figure(figsize=(50,30), dpi=300)
ax = fig.add_subplot(111)
plt.plot(np.linspace(300,2000,17), ddqnTL1000[106][:17], linewidth=20, label='#106')# 
plt.plot(np.linspace(300,2000,17), ddqnTL1000[521][:17], linewidth=20, label='#521')#
plt.plot(np.linspace(300,2000,17), ddqnTL1000[1003][:17], linewidth=20,label='#1003')#
plt.plot(np.linspace(300,2000,17), ddqnTL1000[1119][:17], linewidth=20,label='#1119')#

font_properties = {'family': 'serif', 'weight': 'bold', 'size': 100}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])

label_font_properties = {'family': 'serif', 'weight': 'bold', 'size': 120}  # Adjust the size as needed
label_font = fm.FontProperties(family=label_font_properties['family'], weight=label_font_properties['weight'], size=label_font_properties['size'])

plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Frequency (Hz)", fontproperties=label_font)
plt.ylabel("TL (dB)", fontproperties=label_font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 90}
plt.legend(prop=fm.FontProperties(**legend_font), loc='lower right')
plt.show()

###Figure 7b
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import matplotlib.font_manager as fm
fig = plt.figure(figsize=(50,30), dpi=300)
ax = fig.add_subplot(111)
plt.plot(pd.Series(ddqnTLsum1000).rolling(100).mean(), linewidth=16,color='red',label='epsilon decay=1000')
plt.plot(pd.Series(ddqnTLsum100).rolling(100).mean(), linewidth=16, color='green',label='epsilon decay=100')

font_properties = {'family': 'serif', 'weight': 'bold', 'size': 100}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])

label_font_properties = {'family': 'serif', 'weight': 'bold', 'size': 120}  # Adjust the size as needed
label_font = fm.FontProperties(family=label_font_properties['family'], weight=label_font_properties['weight'], size=label_font_properties['size'])

plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Episodes", fontproperties=label_font)
plt.ylabel("TL sum", fontproperties=label_font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 90}
plt.legend(prop=fm.FontProperties(**legend_font), loc='lower right')
plt.show()

###Figure 7a
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import matplotlib.font_manager as fm
fig = plt.figure(figsize=(50,30), dpi=300)
ax = fig.add_subplot(111)
plt.plot(np.arange(len(ddqnTLsum1000)), ddqnTLsum1000, linewidth=6,color='red', alpha=0.4,label='epsilon decay=1000')# 
plt.plot(pd.Series(ddqnTLsum1000).rolling(100).mean(), linewidth=16,color='red')
plt.plot(np.arange(len(ddqnTLsum100)), ddqnTLsum100, linewidth=6,color='green',alpha=0.4, label='epsilon decay=100')
plt.plot(pd.Series(ddqnTLsum100).rolling(100).mean(), linewidth=16, color='green')

font_properties = {'family': 'serif', 'weight': 'bold', 'size': 100}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])

label_font_properties = {'family': 'serif', 'weight': 'bold', 'size': 120}  # Adjust the size as needed
label_font = fm.FontProperties(family=label_font_properties['family'], weight=label_font_properties['weight'], size=label_font_properties['size'])

plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Episodes", fontproperties=label_font)
plt.ylabel("TL sum", fontproperties=label_font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 90}
plt.legend(prop=fm.FontProperties(**legend_font), loc='upper left')
plt.show()

###Figure 7d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import matplotlib.font_manager as fm
freq= np.linspace(300,3000,28)
ddqncnnruleabsorp1000= np.load('C:/Users/admin/Downloads/Src/SourceCode/RefProb_RuleEnvDDQNCNNeps1000/DDQN CNN REF/RefProb_RuleEnv2DDQNCNNeps10001/DDQNCNNabsorption.npy')
fig = plt.figure(figsize=(50,30), dpi=300)
ax = fig.add_subplot(111)
plt.plot(freq, ddqncnnruleabsorp1000[198], linewidth=20, label='#198')# 
plt.plot(freq, ddqncnnruleabsorp1000[1095], linewidth=20, label='#1095')#
plt.plot(freq, ddqncnnruleabsorp1000[1211], linewidth=20,label='#1211')#
plt.plot(freq, ddqncnnruleabsorp1000[1430], linewidth=20,label='#1430')#

font_properties = {'family': 'serif', 'weight': 'bold', 'size': 100}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])

label_font_properties = {'family': 'serif', 'weight': 'bold', 'size': 120}  # Adjust the size as needed
label_font = fm.FontProperties(family=label_font_properties['family'], weight=label_font_properties['weight'], size=label_font_properties['size'])

plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Frequency (Hz)", fontproperties=label_font)
plt.ylabel(r"$\alpha$", fontproperties=label_font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 90}
plt.legend(prop=fm.FontProperties(**legend_font), loc='lower right')
plt.show()

####Figure 7a
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import matplotlib.font_manager as fm
ddqncnnruleabs1000= np.load('C:/Users/admin/Downloads/Src/SourceCode/RefProb_RuleEnvDDQNCNNeps1000/DDQN CNN REF/RefProb_RuleEnv2DDQNCNNeps10001/DDQNCNNabsum.npy')
ddqncnnruleabs100=np.load('C:/Users/admin/Downloads/Src/SourceCode/RefProb_RuleEnvDDQNCNN100/DDQN CNN/RefProb_RuleEnvDDQNCNN1/DDQNCNNabsum.npy')
fig = plt.figure(figsize=(50,30), dpi=300)
ax = fig.add_subplot(111)
plt.plot(np.arange(len(ddqncnnruleabs1000)), ddqncnnruleabs1000, linewidth=6,color='red', alpha=0.4,label='Epsilon decay=1000')# 
plt.plot(pd.Series(ddqncnnruleabs1000).rolling(100).mean(), linewidth=16,color='red')
plt.plot(np.arange(len(ddqncnnruleabs100)), ddqncnnruleabs100, linewidth=6,color='green',alpha=0.4, label='Epsilon decay=100')
plt.plot(pd.Series(ddqncnnruleabs100).rolling(100).mean(), linewidth=16, color='green')

font_properties = {'family': 'serif', 'weight': 'bold', 'size': 100}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])

label_font_properties = {'family': 'serif', 'weight': 'bold', 'size': 120}  # Adjust the size as needed
label_font = fm.FontProperties(family=label_font_properties['family'], weight=label_font_properties['weight'], size=label_font_properties['size'])

plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Episodes", fontproperties=label_font)
plt.ylabel("Absorption sum", fontproperties=label_font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 90}
plt.legend(prop=fm.FontProperties(**legend_font), loc='lower right')
plt.show()

###Figure 4b
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import matplotlib.font_manager as fm
abssumddqnnn=np.load('C:/Users/admin/Downloads/Src/SourceCode/RandEnvEPS100DDQNNN/DDQN NN/RandEnvEPS100DDQNNN1/DDQNNNabsum.npy')
abssumdqnnn=np.load('C:/Users/admin/Downloads/Src/SourceCode/RandEnvEPS100DQNNN/DQN NN/RandEnvEPS100DQNNN1/DQNNNabsum.npy')
fig = plt.figure(figsize=(50,40), dpi=300)
ax = fig.add_subplot(111)
plt.plot(np.arange(len(abssumddqnnn)), abssumddqnnn, linewidth=6,color='red', alpha=0.4,label='DDQN-FCN')# 
plt.plot(pd.Series(abssumddqnnn).rolling(100).mean(), linewidth=16,color='red')
plt.plot(np.arange(len(abssumdqnnn)), abssumdqnnn, linewidth=6,color='green',alpha=0.4, label='DQN-FCN')
plt.plot(pd.Series(abssumdqnnn).rolling(100).mean(), linewidth=16, color='green')
font_properties = {'family': 'serif', 'weight': 'bold', 'size': 120}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])
plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Episodes", fontproperties=font)
plt.ylabel("Absorption sum", fontproperties=font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 90}
plt.legend(prop=fm.FontProperties(**legend_font), loc='lower right')
plt.show()

## Figure 4c
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import matplotlib.font_manager as fm
import scipy.io as sio
abssumddqncnn=np.load('C:/Users/admin/Downloads/Src/SourceCode/RefProb_RandEnvDDQNCNN100/DDQN CNN/RefProb_RandEnvDDQNCNN100/DDQNCNNabsum.npy')
abssumdqncnn=np.load('C:/Users/admin/Downloads/Src/SourceCode/RefProb_RandEnvDQNCNN100/DQN CNN/RefProb_RandEnvDQNCNN1/DQNCNNabsorption.npy')
res= sio.loadmat('C:/Users/admin/Downloads/Src/SourceCode/Ga_res.mat')
bestres=res['bestres']
bestsum= np.sum(bestres, axis=1)
fig = plt.figure(figsize=(50,40), dpi=300)
ax = fig.add_subplot(111)
plt.plot(np.arange(len(abssumddqncnn[:500])), abssumddqncnn[:500], linewidth=8,color='red', alpha=0.6,label='DDQN-CNN')# 
#plt.plot(pd.Series(abssumddqncnn[:500]).rolling(100).mean(), linewidth=16,color='red',label='DDQN-CNN')
plt.plot(np.arange(len(bestsum)),bestsum, linewidth=16,color='gray', label='GA')
#plt.plot(pd.Series(abssumdqncnn).rolling(100).mean(), linewidth=16, color='green')
font_properties = {'family': 'serif', 'weight': 'bold', 'size': 120}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])
plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Episodes", fontproperties=font)
plt.ylabel("Absorption sum", fontproperties=font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 90}
plt.legend(prop=fm.FontProperties(**legend_font), loc='lower right')
plt.show()

#Figure 4a
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import matplotlib.font_manager as fm
abssumddqncnn=np.load('C:/Users/admin/Downloads/Src/SourceCode/RefProb_RandEnvDDQNCNN100/DDQN CNN/RefProb_RandEnvDDQNCNN100/DDQNCNNabsum.npy')
abssumdqncnn=np.load('C:/Users/admin/Downloads/Src/SourceCode/RefProb_RandEnvDQNCNN100/DQN CNN/RefProb_RandEnvDQNCNN1/DQNCNNabsorption.npy')
fig = plt.figure(figsize=(50,40), dpi=300)
ax = fig.add_subplot(111)
plt.plot(np.arange(len(abssumddqncnn)), abssumddqncnn, linewidth=6,color='red', alpha=0.4,label='DDQN-CNN')# 
plt.plot(pd.Series(abssumddqncnn).rolling(100).mean(), linewidth=16,color='red')
plt.plot(np.arange(len(abssumdqncnn)), abssumdqncnn, linewidth=6,color='green',alpha=0.4, label='DQN-CNN')
plt.plot(pd.Series(abssumdqncnn).rolling(100).mean(), linewidth=16, color='green')
font_properties = {'family': 'serif', 'weight': 'bold', 'size': 100}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])
plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Episodes", fontproperties=font)
plt.ylabel("Absorption sum", fontproperties=font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 90}
plt.legend(prop=fm.FontProperties(**legend_font), loc='lower right')
plt.show()

##Figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import matplotlib.font_manager as fm
#abssumddqncnn=np.load('C:/Users/admin/Downloads/Src/SourceCode/RefProb_RandEnvDDQNCNN100/DDQN CNN/RefProb_RandEnvDDQNCNN100/DDQNCNNabsum.npy')
#abssumdqncnn=np.load('C:/Users/admin/Downloads/Src/SourceCode/RefProb_RandEnvDQNCNN100/DQN CNN/RefProb_RandEnvDQNCNN1/DQNCNNabsorption.npy')
fig = plt.figure(figsize=(50,30), dpi=300)
ax = fig.add_subplot(111)
plt.plot(np.arange(len(abssumddqncnn)), abssumddqncnn, linewidth=6,color='red', alpha=0.4,label='Random selection')# 
plt.plot(pd.Series(abssumddqncnn).rolling(100).mean(), linewidth=16,color='red')
plt.plot(np.arange(len(ddqncnnruleabs100)), ddqncnnruleabs100, linewidth=6,color='green',alpha=0.4, label='Rule-based selection')
plt.plot(pd.Series(ddqncnnruleabs100).rolling(100).mean(), linewidth=16, color='green')

font_properties = {'family': 'serif', 'weight': 'bold', 'size': 100}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])

# Adjust font size for labels only
label_font_properties = {'family': 'serif', 'weight': 'bold', 'size': 120}  # Adjust the size as needed
label_font = fm.FontProperties(family=label_font_properties['family'], weight=label_font_properties['weight'], size=label_font_properties['size'])

plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Episodes", fontproperties=label_font)
plt.ylabel("Absorption sum", fontproperties=label_font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 90}
plt.legend(prop=fm.FontProperties(**legend_font), loc='lower right')
plt.show()

##Figure Five
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import matplotlib.font_manager as fm
freq= np.linspace(300,3000,28)
ddqncnnruleabsorp100= np.load('C:/Users/admin/Downloads/Src/SourceCode/RefProb_RuleEnvDDQNCNN100/DDQN CNN/RefProb_RuleEnvDDQNCNN1/DDQNCNNabsorption.npy')
fig = plt.figure(figsize=(50,30), dpi=300)
ax = fig.add_subplot(111)
plt.plot(freq, ddqncnnruleabsorp100[98], linewidth=20, label='#98')# 
plt.plot(freq, ddqncnnruleabsorp100[113], linewidth=20, label='#113')#
plt.plot(freq, ddqncnnruleabsorp100[258], linewidth=20,label='#258')#
plt.plot(freq, ddqncnnruleabsorp100[725], linewidth=20,label='#725')#

font_properties = {'family': 'serif', 'weight': 'bold', 'size': 100}
font = fm.FontProperties(family=font_properties['family'], weight=font_properties['weight'], size=font_properties['size'])

# Adjust font size for labels only
label_font_properties = {'family': 'serif', 'weight': 'bold', 'size': 120}  # Adjust the size as needed
label_font = fm.FontProperties(family=label_font_properties['family'], weight=label_font_properties['weight'], size=label_font_properties['size'])

plt.xticks(fontproperties=font, weight='bold')
plt.yticks(fontproperties=font, weight='bold')
plt.xlabel("Frequency (Hz)", fontproperties=label_font)
plt.ylabel("/alpha", fontproperties=label_font)

ax.spines['top'].set_linewidth(4)
ax.spines['right'].set_linewidth(4)
ax.spines['bottom'].set_linewidth(4)
ax.spines['left'].set_linewidth(4)

legend_font = {'family': 'serif', 'weight': 'bold', 'size': 90}
plt.legend(prop=fm.FontProperties(**legend_font), loc='lower right')
plt.show()

