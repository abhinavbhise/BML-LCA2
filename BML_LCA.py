!pip install ucimlrepo -q
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, silhouette_score
from ucimlrepo import fetch_ucirepo

bank_marketing = fetch_ucirepo(id=222)
X_raw = bank_marketing.data.features.copy()
y_raw = bank_marketing.data.targets.copy().iloc[:, 0]
df = X_raw.copy()
df["y"] = y_raw.values
df = df.dropna().reset_index(drop=True)
target_col = "y"
y = df[target_col].map({"yes": 1, "no": 0}).values
X = df.drop(columns=[target_col])
cat_cols = X.select_dtypes(include=["object"]).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
k_range = range(2, 11)
for k_test in k_range:
    km_test = KMeans(n_clusters=k_test, init="k-means++", n_init=10, random_state=42)
    km_test.fit(X_scaled)
    inertias.append(km_test.inertia_)
plt.figure(figsize=(7, 4))
plt.plot(list(k_range), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.show()

k = 8
kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
cluster_to_label = {c: pd.Series(y[clusters == c]).mode()[0] for c in np.unique(clusters)}
y_pred = np.array([cluster_to_label[c] for c in clusters])
accuracy = accuracy_score(y, y_pred)
sil_score = silhouette_score(X_scaled, clusters)
print(f"Cluster sizes: {pd.Series(clusters).value_counts().sort_index().to_dict()}")
print(f"Cluster -> Label mapping: {cluster_to_label}")
print(f"Silhouette Score: {sil_score:.4f}")
print(f"Accuracy of K-Means clustering: {accuracy * 100:.2f}%")
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred, target_names=["no", "yes"]))

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="tab10", s=5, alpha=0.6)
axes[0].set_title("K-Means Clusters (PCA-reduced)")
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
plt.colorbar(scatter1, ax=axes[0], label="Cluster")
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="coolwarm", s=5, alpha=0.6)
axes[1].set_title("True Labels (PCA-reduced)")
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
plt.colorbar(scatter2, ax=axes[1], label="0=no, 1=yes")
plt.tight_layout()
plt.show()