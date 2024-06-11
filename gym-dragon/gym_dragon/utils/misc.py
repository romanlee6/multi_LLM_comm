import math
import numpy as np

from numpy.typing import ArrayLike
from typing import Any, Iterable, Union



def cycle_encoding(i: int, cycle_length: int) -> np.ndarray[float]:
    """
    Encode the i-th position in a cycle.

    Parameters
    ----------
    i : int
        Cycle index
    cycle_length : int
        Total length of cycle
    """
    return np.array([
        math.sin(2 * math.pi * i / cycle_length),
        math.cos(2 * math.pi * i / cycle_length),
    ])

def positional_encoding(i: int, embedding_size=2):
    """
    Transformer-style positional encoding.

    Parameters
    ----------
    i : int
        Position in sequence
    embedding_size : int
        Dimension of embedding
    """
    K = -np.log(10000) / embedding_size
    frequencies = np.exp(K * np.arange(0, embedding_size, 2))
    encoding = np.zeros(embedding_size)
    encoding[0::2] = np.sin(i * frequencies)
    encoding[1::2] = np.cos(i * frequencies)
    return encoding

def np_replace(x: np.ndarray, mapping: dict) -> np.ndarray:
    """
    Replace values in an numpy array using a lookup table.

    Parameters
    ----------
    x : np.ndarray
        Numpy array
    mapping : dict
        Dictionary indicating which values should replace occurences of each key
    """
    return np.vectorize(lambda key: mapping.get(key, key))(x)

def one_hot(x: ArrayLike, d: Union[int, Iterable]):
    """
    Convert integer features to one-hot representation.

    Parameters
    ----------
    x : ArrayLike[int]
        Integer feature(s) to convert
    d : int or Iterable
        Dimension of one-hot representation
    """
    d = len(d) if isinstance(d, Iterable) else d
    return np.eye(d)[np.array(x).astype(int)]

def get_item(x: Iterable) -> Any:
    """
    Get an arbitrary item from a collection.

    Parameters
    ----------
    x : Iterable
        Collection of items
    """
    try:
        return next(iter(x))
    except:
        return None

def random_allocation(
    budget: float, costs: ArrayLike, exclude: Iterable[int], random: np.random.Generator):
    """
    Return a random allocation of items that exhaust a fixed budget.

    Parameters
    ----------
    budget : float
        Budget to be allocated
    costs : ArrayLike[float]
        Cost of each item
    exclude : Iterable[int]
        Indices of items to exclude
    random : np.random.Generator
        Random number generator
    """
    costs = np.array(costs, dtype=float)
    costs[list(exclude)] = float('inf')
    allocation = np.zeros_like(costs, dtype=int)
    affordable = (costs <= budget)
    while budget and any(affordable):
        selection = random.choice(np.where(affordable)[0])
        allocation[selection] += 1
        budget -= costs[selection]
        affordable = (costs <= budget)

    return allocation
