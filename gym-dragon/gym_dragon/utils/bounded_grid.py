"""
Module for representing grids with arbitrary integer coordinates (e.g. (x, y, z) location).
"""
import itertools
import numpy as np

from collections.abc import Iterable
from numbers import Real
from typing import Any, Union



### BoundedGrid

class BoundedGrid:
    """
    Wrapper around a numpy array that allows for integer indexing
    via absolute coordinates within arbitrary bounds.

    Attributes
    ----------
    bounds : list[tuple]
        List of (lower, upper) tuples specifying the bounds for each axis
    shape : tuple
        The dimensions of the grid
    view : BoundedGrid.View
        View of this BoundedGrid that references the same underlying data

    Methods
    -------
    __getitem__(loc)
        Get the grid value at an absolute location
    __setitem__(loc, val)
        Set the grid value at an absolute location

    Examples
    --------
    >>> grid = BoundedGrid((-10, 10), (20, 30))

    >>> grid[-5, 25] = 'label'
    >>> grid[-5, 24:27]
    array([None, 'label', None], dtype=object)

    >>> grid[-15, 25]
    IndexError: index -15 is out of range [-10, 10]
    """

    def __init__(self, *bounds: tuple[Real, Real], dtype=object, fill=None):
        """
        Parameters
        ----------
        bounds : tuple[Real, Real]
            Tuples of (lower, upper) specifying the bounds for each axis
        dtype : np.dtype
            Grid data type
        fill : object
            Initial value to fill the grid
        """
        self._bounds = list(bounds)
        self._grid = np.empty(self.shape, dtype=dtype)
        self._grid.fill(fill)
        self._fill = fill

    def __str__(self):
        """
        String representation of this object.
        """
        return f'{self.__class__.__name__}{tuple(self.bounds)}'

    def __iter__(self):
        """
        Iterator over all grid values.
        """
        return self._grid.__iter__()
    
    def __eq__(self, obj):
        """
        Check equality. Returns the result broadcasted as an numpy array.
        """
        return self._grid.__eq__(obj)

    def __getitem__(self, loc: tuple):
        """
        Get the grid value at a given location.

        Parameters
        ----------
        loc : tuple
            Location coordinates
        """
        if loc is ...:
            return self._grid.__getitem__(loc)
        else:
            loc = loc if isinstance(loc, tuple) else (loc,)
            relative_loc = _to_relative_coords(loc, self.bounds)
            return self._grid.__getitem__(relative_loc)

    def __setitem__(self, loc: tuple, value: Any):
        """
        Set the grid value at a given location.

        Parameters
        ----------
        loc : tuple
            Location coordinates
        value : Any
            The value to set
        """
        if isinstance(loc, (type(...), np.ndarray)):
            return self._grid.__setitem__(loc, value)
        else:
            loc = loc if isinstance(loc, tuple) else (loc,)
            relative_loc = _to_relative_coords(loc, self.bounds)
            return self._grid.__setitem__(relative_loc, value)

    @property
    def bounds(self) -> list[tuple[Real, Real]]:
        """
        List of (lower, upper) tuples specifying the grid bounds for each axis.
        """
        return list(self._bounds)

    @property
    def shape(self) -> tuple[int]:
        """
        The dimensions of the grid.
        """
        return tuple(upper - lower + 1 for lower, upper in self.bounds)

    @property
    def view(self) -> 'BoundedGrid.View':
        """
        View of this BoundedGrid that references the same underlying data.

        >>> grid.view[-10:10, 5]
        BoundedGrid((-10, 9), (5, 5))
        """
        return BoundedGrid.View(self)

    def is_in_bounds(self, loc: tuple) -> bool:
        """
        Return whether or not the given location is within bounds of the grid.

        Parameters
        ----------
        loc : tuple
            Location coordinates
        """
        for coord, (lower, upper) in zip(loc, self.bounds):
            if coord < lower or coord > upper:
                return False
        return True

    def locations(self) -> set[tuple]:
        """
        Return a set of all locations within the grid bounds.

        Particularly useful for iteration. For example:

        >>> for (x, y, z) in grid.locations():
        >>>     do_something(grid[x, y, z])
        """
        coords = (slice(lower, upper + 1) for lower, upper in self.bounds)
        return iter(_expand_coords(*coords))

    def neighborhood(self, loc: tuple, radius=1) -> set[tuple]:
        """
        Get all grid locations within a specified cell radius of a given location.

        Parameters
        ----------
        loc : tuple
            Location of center grid cell
        radius : int, default=1
            Cell radius
        """
        deltas = itertools.product(*[range(-radius, radius + 1)] * len(self.bounds))
        locations = set()
        for delta in deltas:
            _loc = tuple(x + dx for x, dx in zip(loc, delta))
            if self.is_in_bounds(_loc):
                locations.add(_loc)

        return locations

    def unique(self, exclude=set()) -> set:
        """
        Return a set of all unique values stored in the grid.

        Parameters
        ----------
        exclude : Iterable
            Values to exclude
        """
        return set(self._grid.flat) - set(exclude)

    def numpy(self) -> np.ndarray:
        """
        Return a numpy array with all the values within the absolute bounds
        of this grid.
        """
        return np.array(self._grid[...])

    def copy(self) -> 'BoundedGrid':
        """
        Return a shallow copy of this BoundedGrid.
        """
        new_grid = BoundedGrid(*self.bounds, dtype=self._grid.dtype, fill=self._fill)
        new_grid[...] = self[...]
        return new_grid

    def find(self, value: Any) -> set:
        """
        Find all grid locations with a particular value.

        Parameters
        ----------
        value : Any
            Grid value
        """
        return {loc for loc in self.locations() if self[loc] == value}

    def replace(self, value: Any, new_value: Any):
        """
        Replace occurences of a particular value.

        Parameters
        ----------
        value : Any
            Old grid value
        new_value : Any
            New grid value
        """
        self._grid[self._grid == value] = new_value

    def count(self, value: Any) -> int:
        """
        Number of occurences of a particular value.

        Parameters
        ----------
        value : Any
            Grid value
        """
        return (self._grid == value).sum()

    def _supergrid(self, *new_bounds: tuple[Real, Real]) -> 'BoundedGrid':
        """
        Enlarge the bounds of the grid.

        Parameters
        ----------
        new_bounds : tuple[Real, Real]
            Tuples of (lower, upper) specifying the bounds for each axis
        """
        supergrid = BoundedGrid(*new_bounds, dtype=self._grid.dtype, fill=self._fill)
        area = tuple(slice(lower, upper + 1) for lower, upper in self.bounds)
        supergrid[area] = self[area] # copy data from this BoundedGrid to supergrid
        self._grid = supergrid[area] # this BoundedGrid references data supergrid
        return supergrid

    def _subgrid(self, *new_bounds: tuple[Real, Real]) -> 'BoundedGrid':
        """
        Shrink the bounds of the grid.

        Parameters
        ----------
        new_bounds : tuple[Real, Real]
            Tuples of (lower, upper) specifying the bounds for each axis
        """
        subgrid = BoundedGrid(*new_bounds, dtype=self._grid.dtype, fill=self._fill)
        area = tuple(slice(lower, upper + 1) for lower, upper in subgrid.bounds)
        subgrid._grid = self[area] # subgrid references data from this BoundedGrid
        return subgrid


    class View:
        """
        Wrapper around a BoundedGrid that allows for adjusting bounds while
        maintaining reference to the same underlying data.

        Examples
        --------
        >>> grid = BoundedGrid((5, 7))

        >>> grid[6] = 'red'
        >>> grid[:]
        array([None, 'red', None], dtype=object)
        >>> grid[4] = 'green'
        IndexError: index 4 is out of range [5, 7]

        Create a view of our original grid.

        >>> new_grid = grid.view[4:8]
        >>> new_grid.bounds
        [(4, 7)]
        >>> new_grid[4] = 'green'
        >>> new_grid[:]
        array(['green', None, 'red', None], dtype=object)

        Update the new grid.

        >>> new_grid[6] = 'blue'
        >>> new_grid[:]
        array(['green', None, 'blue', None], dtype=object)

        The original grid should also be updated.

        >>> grid[:]
        array([None, 'blue', None], dtype=object)
        """

        def __init__(self, bounded_grid: 'BoundedGrid'):
            """
            Parameters
            ----------
            bounded_grid : BoundedGrid
                The `BoundedGrid` instance to be wrapped
            """
            self._bounded_grid = bounded_grid

        def __getitem__(self, area: tuple) -> 'BoundedGrid':
            """
            Return a BoundedGrid representing a view over the specified area.

            Parameters
            ----------
            area : tuple
                Location coordinates with each axis specified as a number or slice
            """
            area = area if isinstance(area, tuple) else (area,)

            # Determine bounds
            supergrid_bounds = list(self._bounded_grid.bounds)
            subgrid_bounds = list(self._bounded_grid.bounds)
            for axis, c in enumerate(area):
                lower, upper = self._bounded_grid.bounds[axis]
                if isinstance(c, slice):
                    start = lower if c.start is None else c.start
                    stop = upper + 1 if c.stop is None else c.stop
                    supergrid_bounds[axis] = min(lower, start), max(upper, stop - 1)
                    subgrid_bounds[axis] = start, stop - 1
                else:
                    supergrid_bounds[axis] = min(lower, c), max(upper, c)
                    subgrid_bounds[axis] = c, c

            # Create view
            supergrid = self._bounded_grid._supergrid(*supergrid_bounds)
            return supergrid._subgrid(*subgrid_bounds)



### Helper Functions

def _to_relative_coords(loc: tuple, bounds: Iterable[tuple]) -> tuple:
    """
    Convert an absolute location to a relative location
    (i.e. for internal grid indexing in a `BoundedGrid`).

    Parameters
    ----------
    loc : tuple
        Absolute location coordinates
    bounds : Iterable[tuple]
        Tuples of (lower, upper) specifying the bounds for each coordinate
    """
    return tuple(
        _to_relative_coord_1d(coord, *axis_bounds)
        for coord, axis_bounds in zip(loc, bounds)
    )


def _to_relative_coord_1d(
    c: Union[Real, slice], lower: Real, upper: Real) -> Union[Real, slice]:
    """
    Convert a single absolute coordinate to a relative coordinate
    (i.e. for internal grid indexing in a `BoundedGrid`).

    Parameters
    ----------
    c : numbers.Real or slice:
        The absolute coordinate to convert (an integer, float, or slice)
    lower : Real
        Absolute coordinate lower bound
    upper : Real
        Absolute coordinate upper bound
    """
    if isinstance(c, slice):
        # Check range
        start, stop = c.start, c.stop
        if start is not None and (start < lower or start >= upper + 1):
            raise IndexError(f'index {start} is out of range [{lower}, {upper}]')
        if stop is not None and (stop < lower or stop > upper + 1):
            raise IndexError(f'index {stop} is out of range [{lower}, {upper}]')

        # Adjust values
        if start is not None:
            start -= lower
        if stop is not None:
            stop -= lower

        return slice(start, stop, c.step)

    elif isinstance(c, Real):
        # Check range
        if not (c >= lower and c < upper + 1):
            raise IndexError(f'index {c} is out of range [{lower}, {upper}]')

        # Adjust value
        return int(c - lower)

    else:
        raise IndexError(f'Only integers, floats, and slices (`:`) are valid indices')


def _expand_coords(*coords: Union[Real, slice]):
    """
    Return the set of tuple locations specified by the given coordinates.

    Parameters
    ----------
    coords : numbers.Real or slice:
        Each coordinate given as an integer, float, or slice
    """
    def coord_to_list(c):
        if isinstance(c, slice):
            return np.arange(c.start, c.stop, c.step)
        elif isinstance(c, Real):
            return np.array([c])
        else:
            raise TypeError(c)

    coords = [coord_to_list(c) for c in coords]
    locs = zip(*[x.flat for x in np.meshgrid(*coords)])
    return set(tuple(loc) for loc in locs)
