import jax.numpy as jnp
import numpy as np
import scipy
from coinem.dataset import Dataset
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple

from abc import abstractproperty

import os

dir, filename = os.path.split(__file__)


class UCI(Dataset):
    @property
    @abstractproperty
    def name(self) -> str:
        raise NotImplementedError

    def preprocess(
        self, test_size: float = 0.2, random_state: int = 0, scale_X: bool = True
    ) -> Tuple[Dataset, Dataset]:
        """Preprocess the dataset."""

        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        if scale_X:
            scaler_X = StandardScaler()
            X_train = scaler_X.fit_transform(X_train)
            X_test = scaler_X.transform(X_test)

        return Dataset(jnp.array(X_train), jnp.array(y_train)), Dataset(
            jnp.array(X_test), jnp.array(y_test)
        )


@dataclass
class Boston(UCI):
    """Boston Housing Dataset."""

    def __post_init__(self):
        path = os.path.join(dir, "boston.csv")
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        self.X = jnp.array(data[:, :-1])
        self.y = jnp.array(data[:, -1])

    @property
    def name(self) -> str:
        return "Boston"


@dataclass
class Concrete(UCI):
    """Concrete Compressive Strength Dataset."""

    def __post_init__(self):
        path = os.path.join(dir, "concrete.csv")
        data = np.loadtxt(path, delimiter=",", skiprows=1)

        self.X = jnp.array(data[:, :-1])
        self.y = jnp.array(data[:, -1])

    @property
    def name(self) -> str:
        return "Concrete"


@dataclass
class Energy(UCI):
    """Energy Efficiency Dataset."""

    def __post_init__(self):
        path = os.path.join(dir, "energy.csv")
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        self.X = jnp.array(data[:, :-2])
        self.y = jnp.array(data[:, -1])  # Cooling Load

    @property
    def name(self) -> str:
        return "Energy"


@dataclass
class Kin8nm(UCI):
    """Kin8nm Dataset."""

    def __post_init__(self):
        path = os.path.join(dir, "kin8nm.csv")
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        self.X = jnp.array(data[:, :-1])
        self.y = jnp.array(data[:, -1])

    @property
    def name(self) -> str:
        return "Kin8nm"


@dataclass
class Naval(UCI):
    """Naval Propulsion Dataset."""

    def __post_init__(self):
        path = os.path.join(dir, "naval-propulsion.csv")
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        self.X = jnp.array(data[:, :-2])
        self.y = jnp.array(data[:, -1])  # GT Turbine decay state coefficient

    @property
    def name(self) -> str:
        return "Naval"


@dataclass
class Power(UCI):
    """Power Plant Dataset."""

    def __post_init__(self):
        path = os.path.join(dir, "power-plant.csv")
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        self.X = jnp.array(data[:, :-1])
        self.y = jnp.array(data[:, -1])

    @property
    def name(self) -> str:
        return "Power"


@dataclass
class Protein(UCI):
    """Protein Structure Dataset."""

    def __post_init__(self):
        path = os.path.join(dir, "protein-structure.csv")
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        self.X = jnp.array(data[:, :-1])
        self.y = jnp.array(data[:, -1])

    @property
    def name(self) -> str:
        return "Protein"


@dataclass
class Wine(UCI):
    """Wine Quality Dataset."""

    def __post_init__(self):
        path = os.path.join(dir, "wine.csv")
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        self.X = jnp.array(data[:, :-1])
        self.y = jnp.array(data[:, -1])

    @property
    def name(self) -> str:
        return "Wine"


@dataclass
class Yacht(UCI):
    """Yacht Hydrodynamics Dataset."""

    def __post_init__(self):
        path = os.path.join(dir, "yacht.csv")
        data = np.loadtxt(path, delimiter=",", skiprows=1)
        self.X = jnp.array(data[:, :-1])
        self.y = jnp.array(data[:, -1])

    @property
    def name(self) -> str:
        return "Yacht"


# @dataclass
# class Year(UCI):
#     """Year Prediction MSD Dataset."""

#     def __post_init__(self):
#         data = jnp.loadtxt('year.csv' , delimiter = ',' , skiprows = 1)
#         self.X = data[:, :-1]
#         self.y = data[:, -1]

#     @property
#     def name(self) -> str:
#         return 'Year'


@dataclass
class Covertype(UCI):
    """Covertype Dataset."""

    def __post_init__(self):
        path = os.path.join(dir, "covertype.mat")
        data = scipy.io.loadmat(path)
        X_input = data["covtype"][:, 1:]
        y_input = data["covtype"][:, 0]
        y_input[y_input == 2] = 0
        self.X = jnp.array(X_input)
        self.y = jnp.array(y_input)

    @property
    def name(self) -> str:
        return "Covertype"


@dataclass
class Wisconsin(UCI):
    def __post_init__(self):
        path = os.path.join(dir, "breast-cancer-wisconsin.data")

        # Load dataset:
        dataset = np.loadtxt(path, dtype=str, delimiter=",")

        # Remove datapoints with missing attributes and change dtype to float:
        dataset = dataset[~(dataset == "?").any(axis=1), :].astype(float)

        # Extract features and labels:
        features = np.array(dataset[:, 1:10])
        labels = np.array([(dataset[:, 10] - 2) / 2]).transpose()

        self.X = jnp.array(features)
        self.y = jnp.array(labels)

    @property
    def name(self) -> str:
        return "Wisconsin"


@dataclass
class Banknote(UCI):
    def __post_init__(self):
        path = os.path.join(dir, "data_banknote_authentication.txt")
        dataset = np.loadtxt(path, delimiter=",")
        self.X = jnp.array(dataset[:, :-1])
        self.y = jnp.array(dataset[:, -1])

    @property
    def name(self) -> str:
        return "Banknote"


@dataclass
class Cleveland(UCI):
    def __post_init__(self):
        path = os.path.join(dir, "heart_cleveland_upload.csv")
        dataset = np.loadtxt(path, delimiter=",", skiprows=1)
        self.X = jnp.array(dataset[:, :-1])
        self.y = jnp.array(dataset[:, -1])

    @property
    def name(self) -> str:
        return "Cleveland"


@dataclass
class Haberman(UCI):
    def __post_init__(self):
        path = os.path.join(dir, "haberman.data")
        dataset = np.loadtxt(path, delimiter=",")
        self.X = jnp.array(dataset[:, :-1])
        self.y = jnp.array(dataset[:, -1]) - 1.0

    @property
    def name(self) -> str:
        return "Haberman"
