from collections import deque
from typing import Sequence, TypeVar, Generic
import numpy as np
from numpy.typing import ArrayLike

def euclidean_distance(a:ArrayLike, b:ArrayLike) -> ArrayLike:
    '''
    Euclidean distance.

    Parameters
    ==========
    a : (m,n) ArrayLike
        First array.
    b : (m,n) ArrayLike | (n,1) ArrayLike
        Second array or vector
    
    Returns
    =======
    d : (m,1) ArrayLike
        Vector of distances or pairwise distances.
    '''
    return np.sqrt(np.sum((a-b)**2, axis=1))


T = TypeVar('T') # Generic type

class SimpleMovingAverage(Generic[T]):
    def __init__(self, n:int, elements:Sequence[T]=[]):
        '''
        Simple Moving Average filter.
    
        Parameters
        ==========
        n : int
            Size of averaging window.
        element : Sequence[T]
            Initial sequence of elements.
        '''
        self._deque = deque(elements, maxlen=n)
        self._value = None

    def update(self, p:T) -> T:
        '''
        Update filter with new reading.

        Parameters
        ==========
        p : T
            New value.

        Returns
        =======
        value : T
            Filtered value.
        '''
        n = len(self._deque)

        if n < self._deque.maxlen:
            # not full
            self._value = (self._value*n + p)/(n+1) if self._value is not None else p
        else:
            self._value += (p - self._deque.popleft())/self._deque.maxlen

        self._deque.append(p)
        
        return self._value

    def get_value(self) -> T:
        '''
        Get filtered value.

        Returns
        =======
        value: T
            Filtered value.
        '''
        return self._value

    def clear(self) -> None:
        '''
        Clear filter.
        '''
        self._deque.clear()

    def __len__(self):
        return len(self._deque)

    def is_full(self):
        '''
        Check if filter is fully populated.

        Returns
        =======
        is_full : bool
            True if full, False otherwise.
        '''
        return len(self._deque) == self._deque.maxlen