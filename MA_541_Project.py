
# Part -1
import pandas as pd
import numpy as np
from numpy.random import seed
from numpy.random import randn
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import chisquare
from scipy.stats import kstest
import seaborn as sns
from itertools import combinations
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_excel('/Users/vinodhkumar/Downloads/ma541.xlsx')
df.head()

# Finding Mean and Standard deviation for all column
df.describe()
df.isnull().sum()


# the sample correlations among each pair of the four random variables (columns) of the data.
corr = df.corr()
corr_mat = df.corr().abs()
corr_mat

# Heatmap
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)


# Part-2

# 1. A histogram for each column (hint: four histograms total)
fig, axes = plt.subplots(len(df.columns)//3, 3, figsize=(20, 6))
for col, axis in zip(df.columns, axes):
    df.hist(column = col, bins = 100, ax=axis)


# 2. A time series plot for each column (hint: use the series “1, 2, 3, ..., 1000” as the
# horizontal axis; four plots total)

xaxis = np.arange(1,1001)
plt.plot(xaxis, np.array(df['Close_ETF']))
plt.show()

y=np.array(df['Close_ETF'])

#3. A time series plot for all four columns (hint: one plot including four “curves” and each “curve” describes one column)
sns.pairplot(df)


#4. Three scatter plots to describe the relationships between the ETF column and the OIL
# column; between the ETF column and the GOLD column; between the ETF column and the JPM column, respectively

# Scatter Plot between ETF column and the OIL column
plt.scatter(df['Close_ETF'], df['oil'])
plt.xlabel('Close_ETF')
plt.ylabel('oil')

# Scatter Plot between ETF column and the GOLD column
plt.scatter(df['Close_ETF'], df['gold'])
plt.xlabel('Close_ETF')
plt.ylabel('gold')

# Scatter Plot between ETF column and the JPM column
plt.scatter(df['Close_ETF'], df['JPM'])
plt.xlabel('Close_ETF')
plt.ylabel('JPM')

# Part-3
# Gaussian distribution checking for Close_ETF column (univariate observations)

seed(1)
data = df['Close_ETF']

# normality test
stat, p = shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05

# Shapiro-Wilk Test
if p > alpha:
	print('Acc to Shapiro test Sample looks Gaussian (fail to reject H0)')
else:
	print('Acc to Shapiro test Sample does not look Gaussian (reject H0)')

# Kolmogorov Smirnov test
stat, p = kstest(data,'norm')
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Acc to Kolmogorov-Smirnov test Sample looks Gaussian (fail to reject H0)')
else:
	print('Acc to Kolmogorov-Smirnov test Sample does not look Gaussian (reject H0)')

# chisquare
stat, p = chisquare(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Acc to chisquare test Sample looks Gaussian (fail to reject H0)')
else:
	print('Acc to chisquare test Sample does not look Gaussian (reject H0)')

# Normal Test
stat, p = normaltest(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Acc to Normal test Sample looks Gaussian (fail to reject H0)')
else:
	print('Acc to Normal test Sample does not look Gaussian (reject H0)')

# Gaussian distribution checking for Oil column (univariate observations)

seed(1)
data = df['oil']

# normality test
stat, p = shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05

# Shapiro-Wilk Test
if p > alpha:
	print('Acc to Shapiro test Sample looks Gaussian (fail to reject H0)')
else:
	print('Acc to Shapiro test Sample does not look Gaussian (reject H0)')

# Kolmogorov Smirnov test
stat, p = kstest(data,'norm')
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Acc to Kolmogorov-Smirnov test Sample looks Gaussian (fail to reject H0)')
else:
	print('Acc to Kolmogorov-Smirnov test Sample does not look Gaussian (reject H0)')

# chisquare
stat, p = chisquare(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Acc to chisquare test Sample looks Gaussian (fail to reject H0)')
else:
	print('Acc to chisquare test Sample does not look Gaussian (reject H0)')

# Normal Test
stat, p = normaltest(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Acc to Normal test Sample looks Gaussian (fail to reject H0)')
else:
	print('Acc to Normal test Sample does not look Gaussian (reject H0)')


# Gaussian distribution checking for Gold column (univariate observations)

seed(1)
data = df['gold']

# normality test
stat, p = shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05

# Shapiro-Wilk Test
if p > alpha:
	print('Acc to Shapiro test Sample looks Gaussian (fail to reject H0)')
else:
	print('Acc to Shapiro test Sample does not look Gaussian (reject H0)')

# Kolmogorov Smirnov test
stat, p = kstest(data,'norm')
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Acc to Kolmogorov-Smirnov test Sample looks Gaussian (fail to reject H0)')
else:
	print('Acc to Kolmogorov-Smirnov test Sample does not look Gaussian (reject H0)')

# chisquare
stat, p = chisquare(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Acc to chisquare test Sample looks Gaussian (fail to reject H0)')
else:
	print('Acc to chisquare test Sample does not look Gaussian (reject H0)')

# Normal Test
stat, p = normaltest(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Acc to Normal test Sample looks Gaussian (fail to reject H0)')
else:
	print('Acc to Normal test Sample does not look Gaussian (reject H0)')


# Gaussian distribution checking for Oil column (univariate observations)

seed(1)
data = df['JPM']

# normality test
stat, p = shapiro(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05

# Shapiro-Wilk Test
if p > alpha:
	print('Acc to Shapiro test Sample looks Gaussian (fail to reject H0)')
else:
	print('Acc to Shapiro test Sample does not look Gaussian (reject H0)')

# Kolmogorov Smirnov test
stat, p = kstest(data,'norm')
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Acc to Kolmogorov-Smirnov test Sample looks Gaussian (fail to reject H0)')
else:
	print('Acc to Kolmogorov-Smirnov test Sample does not look Gaussian (reject H0)')

# chisquare
stat, p = chisquare(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Acc to chisquare test Sample looks Gaussian (fail to reject H0)')
else:
	print('Acc to chisquare test Sample does not look Gaussian (reject H0)')

# Normal Test
stat, p = normaltest(data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > alpha:
	print('Acc to Normal test Sample looks Gaussian (fail to reject H0)')
else:
	print('Acc to Normal test Sample does not look Gaussian (reject H0)')
