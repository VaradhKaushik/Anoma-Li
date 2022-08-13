from IForest import *

##### Generating Test Dataset #####
mean = [0, 0]
cov = [[1, 0], [0, 1]]  # diagonal covariance
Nobjs = 2000

x, y = np.random.multivariate_normal(mean, cov, Nobjs).T
x[0], y[0] = 3.3, 3.3       #Add manual outlier

df=np.array([x,y]).T
df = pd.DataFrame(df,columns=['feat1','feat2'])

plt.figure("Generated Dataset")
plt.plot(x,y, 'bo')

iforest_pred(df=df, subspace=256, cntm=0.01)

color = ['r' if val==1 else 'g' for val in df["IF_anomaly"]]
df.plot.scatter("feat1", "feat2", color=color)

plt.plot(df['feat1'], color=color)
plt.xlabel("feat1")

plt.plot(df['feat2'], color=color)
plt.xlabel("feat2")

plt.show()
