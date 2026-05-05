# 5)
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

np.random.seed(42)
x = np.random.rand(100)
tr_x, ts_x = x[:50], x[50:]
tr_y = ["Class1" if v <= 0.5 else "Class2" for v in tr_x]

for k in [1, 2, 3, 4, 5, 20, 30]:
    preds = []
    for p in ts_x:
        idx = np.argsort(np.abs(tr_x - p))[:k]
        lab = [tr_y[i] for i in idx]
        preds.append(Counter(lab).most_common(1)[0][0])

    print("k =", k, "Class1:", preds.count("Class1"), "Class2:", preds.count("Class2"))

    plt.figure()
    plt.scatter(tr_x, [0] * 50, c=['b' if l == 'Class1' else 'r' for l in tr_y])
    plt.scatter(ts_x, [1] * 50, c=['b' if p == 'Class1' else 'r' for p in preds], marker='x')
    plt.title(f"KNN Classification (k={k})")
    plt.show()


# 6)
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 10, 50)
y = np.sin(X) + np.random.normal(0, 0.1, 50)
X_mat = np.c_[np.ones(50), X]

def lwr(x, X, y, tau):
    w = np.exp(-np.sum((X - x) ** 2, axis=1) / (2 * tau ** 2))
    W = np.diag(w)
    theta = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
    return x @ theta

tau = 0.5
x_test = np.linspace(0, 10, 100)
pred = [lwr([1, v], X_mat, y, tau) for v in x_test]

plt.scatter(X, y, c='red')
plt.plot(x_test, pred, c='blue')
plt.title("Locally Weighted Regression")
plt.show()


# 7)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import fetch_california_housing

df = fetch_california_housing(as_frame=True).frame
X, y = df[['AveRooms']], df['MedHouseVal']

model = LinearRegression().fit(X, y)

plt.scatter(X, y, s=10, c='b')
plt.plot(X, model.predict(X), c='r')
plt.title("Linear Regression")
plt.show()

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
cols = ["mpg", "c", "disp", "hp", "wt", "acc", "yr", "orig"]

df2 = pd.read_csv(url, sep='\s+', names=cols, na_values="?").dropna()
X2, y2 = df2[['disp']], df2['mpg']

X_poly = PolynomialFeatures(2).fit_transform(X2)
model2 = LinearRegression().fit(X_poly, y2)

plt.scatter(X2, y2, s=10, c='b')
plt.scatter(X2, model2.predict(X_poly), s=10, c='r')
plt.title("Polynomial Regression")
plt.show()


# 8)
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree

data = load_breast_cancer()
X, y = data.data, data.target

clf = DecisionTreeClassifier().fit(X, y)
pred = clf.predict(X[:1])

print("Prediction:", data.target_names[pred[0]])

plt.figure(figsize=(12, 8))
plot_tree(
    clf,
    feature_names=data.feature_names,
    class_names=data.target_names,
    filled=True
)
plt.show()


# 9)
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

X, y = fetch_olivetti_faces(
    shuffle=True,
    random_state=42,
    return_X_y=True
)

X_tr, X_ts, y_tr, y_ts = train_test_split(
    X, y, test_size=0.2, random_state=42
)

m = GaussianNB().fit(X_tr, y_tr)
pred = m.predict(X_ts)

print(f"Accuracy: {m.score(X_ts, y_ts) * 100:.2f}%")
print("Classification Report:\n", classification_report(y_ts, pred))
print("Confusion Matrix:\n", confusion_matrix(y_ts, pred))
print(f"Cross-validation accuracy: {cross_val_score(m, X, y, cv=5).mean() * 100:.2f}%")

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_ts[i].reshape(64, 64), cmap='gray')
    plt.title(f"T:{y_ts[i]} P:{pred[i]}")
    plt.axis('off')

plt.show()


# 10)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report

X, y = load_breast_cancer(return_X_y=True)

X_sc = StandardScaler().fit_transform(X)

km = KMeans(n_clusters=2, random_state=42).fit(X_sc)
y_km = km.predict(X_sc)

print(
    "Confusion Matrix:\n",
    confusion_matrix(y, y_km),
    "\nReport:\n",
    classification_report(y, y_km)
)

pca = PCA(n_components=2)
X_p = pca.fit_transform(X_sc)
ctr = pca.transform(km.cluster_centers_)

df = pd.DataFrame(X_p, columns=['PC1', 'PC2'])
df[['Cluster', 'True Label']] = list(zip(y_km, y))

for h, p, t in [
    ('Cluster', 'Set1', 'K-Means'),
    ('True Label', 'coolwarm', 'True Labels')
]:
    sns.scatterplot(
        data=df,
        x='PC1',
        y='PC2',
        hue=h,
        palette=p,
        s=100,
        edgecolor='black',
        alpha=0.7
    )
    plt.title(t)
    plt.legend(title=h)
    plt.show()

plt.scatter(X_p[:, 0], X_p[:, 1], c=y_km, cmap='Set1', s=100, alpha=0.7)
plt.scatter(ctr[:, 0], ctr[:, 1], s=200, c='red', marker='X', label='Centroids')
plt.title('K-Means with Centroids')
plt.legend()
plt.show()