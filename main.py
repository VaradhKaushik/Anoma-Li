from iforest import *

np.random.seed(14)

##### Generating Test Dataset #####         - Cited in references
def generate_data(n=2000):
    mean = [0, 0]
    cov = [[1, 0], [0, 1]]  # diagonal covariance

    x, y = np.random.multivariate_normal(mean, cov, n).T
    x[0], y[0] = 3.3, 3.3       #Add manual outlier
    x[100], y[100] = 4.5, 3

    df=np.array([x,y]).T
    df = pd.DataFrame(df,columns=['feat1','feat2'])

    plt.figure("Generated Dataset")
    plt.title(f"Random Generated Dataset with {n} points")
    plt.xlabel("Feature 1 of df")
    plt.ylabel("Feature 2 of df")
    plt.scatter(x,y,c='turquoise', marker='o', alpha=0.7)

    return df

df = generate_data()

##### Calling the algorithms #####
iforest_pred(df=df, subspace=64, cntm=0.01, seed=14)

##### Visualization of the anomalies #####
try:
    color = ['r' if val==1 else 'g' for val in df["IF_anomaly"]]
    
    fig1 = plt.figure("Isolation Forest")
    plt.title("Isolation Forest Anomalies")
    
    plt.scatter(df['feat1'], df['feat2'], c=color, alpha=0.7)
    plt.xlabel("Feature 1 of df")
    plt.ylabel("Feature 2 of df")

except KeyError:
    print("Isolation Forest anomaly data not found")

# def visualise():

plt.show()
