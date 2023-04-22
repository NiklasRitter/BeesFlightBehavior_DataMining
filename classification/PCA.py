import numpy as np

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def PCA(filenumber):
    maDf = pd.read_csv('ma_nearest_neighbor_transform_all_median_' + str(filenumber) + '.csv', sep=',')
    maDf = maDf.drop(['date', 'activity'], axis=1)

    # Create features and target datasets
    features = ['temperature', 'tavg', 'tmax', 'tmin', 'prcp', 'tsun', 'humidity', 'weight']
    X = maDf[features].values

    # Standardize the features
    X = StandardScaler().fit_transform(X)

    # Preview X
    pd.DataFrame(data=X, columns=features).head()


    # Instantiate PCA
    pca = PCA(n_components=4)


    # Fit PCA to features
    principalComponents = pca.fit_transform(X)

    # write pca data into file
    pd.DataFrame(principalComponents).to_csv("pca_data.csv", index=False)

    # look at the PC´s
    for a in pca.components_:
        print(list(map(lambda x: round(x, 3), list(a))))

    print(pca.explained_variance_ratio_)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    plt.show()


    # Calculate the variance explained by principle components
    plt.scatter(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
    plt.xlabel('component')
    plt.ylabel('explained variance ratio');
    plt.show()

