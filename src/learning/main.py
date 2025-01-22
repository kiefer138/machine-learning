import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap

from learning.classifiers import Classifier, Perceptron, AdalineGD, AdalineSD


def get_data_frame(data_site: str) -> pd.DataFrame:
    print("From URL:", data_site)
    return pd.read_csv(data_site, header=None, encoding="utf-8")


def plot_convergence(classifier: Classifier):
    plt.plot(
        range(len(classifier.errors)), classifier.errors, color="magenta", marker="o"
    )
    plt.xlabel("Iterations")
    plt.ylabel("Number of updates")
    plt.savefig(f"convergence_{str(classifier)}.png")
    plt.clf()


def plot_decision_domains(
    X: np.ndarray,
    y: np.ndarray,
    classifier: Classifier,
    resolution: float = 0.02,
    threshold: float = 0.0,
):
    markers = ("o", "s", "^", "v", "<")
    colors = ("magenta", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # Plot decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )

    lab = classifier.predict(
        np.array([xx1.ravel(), xx2.ravel()]).T, threshold=threshold
    )
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=f"Class {cl}",
            edgecolor="black",
        )
    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Petal length (cm)")
    plt.legend(loc="upper left")
    plt.savefig(f"decision_regions_{str(classifier)}.png")
    plt.clf()


def scale_features(X: np.ndarray) -> np.ndarray:
    X_std = np.copy(X)
    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    return X_std


def main():
    data_site = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    )

    df = get_data_frame(data_site)

    # Select setosa and versicolor
    y = df.iloc[0:100, 4].values

    # Set binary target values
    y = np.where(y == "Iris-setosa", 0, 1)

    # Extract features data
    X = df.iloc[0:100, [0, 2]].values

    plt.scatter(X[:50, 0], X[:50, 1], color="magenta", marker="o", label="Setosa")
    plt.scatter(
        X[50:100, 0], X[50:100, 1], color="blue", marker="o", label="Versicolor"
    )
    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Petal length (cm)")
    plt.legend(loc="upper left")
    plt.savefig("raw_data.png")
    plt.clf()

    perceptron = Perceptron(learning_rate=0.1, n_iter=10)
    perceptron.fit(X, y)
    plot_convergence(perceptron)
    plot_decision_domains(X, y, perceptron)

    # Featur scaling
    X_std = scale_features(X)

    adalinegd = AdalineGD(learning_rate=0.5, n_iter=20)
    adalinegd.fit(X_std, y)
    plot_decision_domains(X_std, y, adalinegd, threshold=0.5)
    plot_convergence(adalinegd)

    adalinesd = AdalineSD(learning_rate=0.01, n_iter=15, random_state=1)
    adalinesd.fit(X_std, y)
    plot_decision_domains(X_std, y, adalinesd, threshold=0.5)
    plot_convergence(adalinesd)


if __name__ == "__main__":
    main()
