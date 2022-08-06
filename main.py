# Initial commit for main.py
from IForest import *

##### Generating Test Dataset #####
mean = [0, 0]
cov = [[1, 0], [0, 1]]  # diagonal covariance
Nobjs = 2000

x, y = np.random.multivariate_normal(mean, cov, Nobjs).T
x[0], y[0] = 3.3, 3.3       #Add manual outlier

df=np.array([x,y]).T
df = pd.DataFrame(df,columns=['feat1','feat2'])

plt.figure(figsize=(7,7))
plt.plot(x,y,'bo')
plt.show()


iforest_pred(df=df)
