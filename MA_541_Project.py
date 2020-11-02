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
#%matplotlib inline
from itertools import combinations
import matplotlib.pyplot as plt

df = pd.read_excel('finalProjectData.xlsx')
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


# Part 4: Break your data into small groups and let them discuss the importance of the Central Limit Theorem

# Consider the ETF column (1000 values) as the population (x), and do the follows. Any software may be used.
# Mean for ETF Column
mean = df.mean(axis = 0, skipna = True) 
ETF_mean = mean[0]
print(f'Mean for ETF Column {ETF_mean}')

# Standard deviation for ETF Column
std = df.std(axis = 0, skipna = True) 
ETF_std = std[0]
print(f'Standard deviation for ETF Column {ETF_std}')


# # Break the population into 50 groups sequentially and each group includes 20 values.
seq_sample_means =[]
for split in np.split(df['Close_ETF'], 50):
    seq_sample_means.append(split.mean())
print(seq_sample_means)


# Calculate the sample mean (𝑥) of each group. Draw a histogram of all the sample means. Comment on the distribution of these sample means, 
# i.e., use the histogram to assess the normality of the data consisting of these sample means.
plt.hist(seq_sample_means, density=True, bins=50)  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('Sequntial data split Sample Mean');


#Calculate the mean (𝜇𝑥) and the standard deviation (𝜎𝑥) of the data including these sample means. Make a comparison between 𝜇𝑥 and 𝜇𝑥 , between 𝜎𝑥 and 𝜎𝑥 . Here, 𝑛 is
# √𝑛 the number of sample means calculated from Item 

# To be done

seq_sample_means_10 =[]
for split in np.split(df['Close_ETF'], 10):
    seq_sample_means_10.append(split.mean())
print(seq_sample_means_10)
plt.hist(seq_sample_means_10, density=True, bins=50)  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('Sequntial data split Sample Mean');

#Calculate the mean (𝜇𝑥) and the standard deviation (𝜎𝑥) of the data including these sample means. Make a comparison between 𝜇𝑥 and 𝜇𝑥 , between 𝜎𝑥 and 𝜎𝑥 . Here, 𝑛 is
# √𝑛 the number of sample means calculated from Item 

# To be done

# Generate 50 simple random samples or groups (with replacement) from the population.
# The size of each sample is 20, i.e., each group includes 20 values

sample_means =[] 
for _ in range(50):
    sample = df['Close_ETF'].sample(20, replace = True)
    mean = sample.mean()
    sample_means.append(mean)
print(sample_means)

# Calculate the sample mean (𝑥) of each group. Draw a histogram of all the sample means. Comment on the distribution of these sample means, i.e., 
# use the histogram to assess the normality of the data consisting of these sample means.

plt.hist(sample_means, density=True, bins=50)  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('Sample Mean');

#Calculate the mean (𝜇𝑥) and the standard deviation (𝜎𝑥) of the data including these sample means. Make a comparison between 𝜇𝑥 and 𝜇𝑥 , between 𝜎𝑥 and 𝜎𝑥 . Here, 𝑛 is
# √𝑛 the number of sample means calculated from Item 

# To be done


# Generate 10 simple random samples or groups (with replacement) from the population.
# The size of each sample is 100, i.e., each group includes 100 values

sample_means_10 =[] 
for _ in range(10):
    sample = df['Close_ETF'].sample(100, replace = True)
    mean = sample.mean()
    sample_means_10.append(mean)
print(sample_means_10)

# # Calculate the sample mean (𝑥) of each group. Draw a histogram of all the sample means. Comment on the distribution of these sample means, i.e., 
# use the histogram to assess the normality of the data consisting of these sample means.

plt.hist(sample_means_10, density=True, bins=50)  # `density=False` would make counts
plt.ylabel('Probability')
plt.xlabel('Sample Mean');

#Calculate the mean (𝜇𝑥) and the standard deviation (𝜎𝑥) of the data including these sample means. Make a comparison between 𝜇𝑥 and 𝜇𝑥 , between 𝜎𝑥 and 𝜎𝑥 . Here, 𝑛 is
# √𝑛 the number of sample means calculated from Item 

# To be done

# In Part 3 of the project, you have figured out the distribution of the population (the entire
# ETF column). Does this information have any impact on the distribution of the sample mean(s)? Explain your answer.
