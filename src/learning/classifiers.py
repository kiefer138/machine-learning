from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from typing import Any, List, Tuple


class Classifier(ABC):
    def __init__(
        self, learning_rate: float = 0.01, n_iter: int = 50, random_state: int = 1
    ):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        self.w_: np.ndarray
        self.b_: np.float64

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """Net input function"""
        return np.dot(X, self.w_) + self.b_

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> Classifier:
        """Updates weights and biases based on training data
        and target values.

        Args:
            X (np.ndarray): Training vectors, shape = [n_samples, m_features]
            y (np.ndarray): Target values, shape = [n_samples]

        Returns:
            Perceptron: returns updated self
        """
        pass

    @property
    @abstractmethod
    def errors(self):
        pass

    def predict(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        return np.where(self.net_input(X) >= threshold, 1, 0)

    def activation(self, X: np.ndarray) -> np.ndarray:
        return X


class Perceptron(Classifier):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Perceptron:
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    @property
    def errors(self) -> List[int]:
        return self.errors_

    def __str__(self):
        return "perceptron"


class AdalineGD(Classifier):
    def fit(self, X: np.ndarray, y: np.ndarray) -> AdalineGD:
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.losses_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.learning_rate * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.learning_rate * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    @property
    def errors(self) -> List[float]:
        return self.losses_

    def __str__(self):
        return "adalinegd"


class AdalineSD(Classifier):
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iter: int = 10,
        shuffle: bool = True,
        random_state=None,
    ):
        super().__init__(learning_rate, n_iter, random_state)
        self.w_initialized: bool = False
        self.shuffle = shuffle

    def fit(self, X: np.ndarray, y: np.ndarray) -> AdalineSD:
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> AdalineSD:
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _initialize_weights(self, m: int):
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float_(0.0)

        self.w_initialized = True

    def _update_weights(self, xi: np.ndarray, target: np.ndarray) -> np.ndarray:
        output = self.activation(self.net_input(xi))
        error = target - output
        self.w_ += self.learning_rate * 2.0 * xi * error
        self.b_ += self.learning_rate * 2.0 * error
        loss = error**2
        return loss

    def _shuffle(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    @property
    def errors(self) -> List[float]:
        return self.losses_

    def __str__(self):
        return "adalinesd"
