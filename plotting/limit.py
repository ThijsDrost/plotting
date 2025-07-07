from collections.abc import Sequence


class Limit:
    """
    A class to represent and manage a range of values (limits) with optional inversion.

    Attributes
    ----------
    inverted : bool
        Indicates if the limits are inverted.
    _outer_limit : tuple of float
        The outer bounds of the limit. The limit is always within these bounds.
    _limit : tuple of float
        The current limit within the outer bounds.
    """

    def __init__(self, outer_limit, limit=None, inverted=False):
        """
        A class to represent and manage a range of values (limits) with optional inversion.

        Parameters
        ----------
        outer_limit : Sequence
            The outer bounds of the limit.
        limit : float, int, or Sequence, optional
            The initial limit. Defaults to None.
        inverted : bool, optional
            Whether the limits are inverted. Defaults to False.

        Raises
        ------
        ValueError
            If the bounds of `outer_limit` or `limit` are invalid.
        TypeError
            If `limit` is not a valid type.
        """
        self.inverted = inverted
        self._outer_limit = tuple(outer_limit)
        if inverted:
            self._outer_limit = (outer_limit[1], outer_limit[0])

        if limit is None:
            self._limit = self._outer_limit
        else:
            self._limit = self.parse_limit(limit, inverted)

        self.check_order(self._outer_limit, 'outer_limit')
        self.check_order(self._limit, 'limit')

    def parse_limit(self, limit, inverted) -> tuple[float, float]:
        """
        Parse and validate the given limit.

        Parameters
        ----------
        limit : float, int, Sequence[float | int]
            The limit to parse.

        Returns
        -------
        tuple of float
            The parsed limit as a tuple.

        Raises
        ------
        ValueError
            If the limit values are incorrect (same value or wrong order), or if the limit is an empty sequence.
        TypeError
            If the limit is not a float, integer, or Sequence[float | integer].
        """
        if isinstance(limit, (float, int)):
            return limit, self._outer_limit[1]
        elif isinstance(limit, Sequence):
            if len(limit) == 0:
                raise ValueError("`limit` cannot be an empty Sequence")
            if len(limit) == 1:
                return self.parse_limit(limit[0], inverted)

            if inverted:
                limit = (limit[1], limit[0])
            else:
                limit = (limit[0], limit[1])

            if limit[0] is None:
                limit = (self._outer_limit[0], limit[1])
            if limit[1] is None:
                limit = (limit[0], self._outer_limit[1])

            return limit
        else:
            raise TypeError("`limit` should be float, integer or Sequence")

    def _new_limit(self, limit, inverted=None) -> tuple[float, float]:
        """
        Calculate a new limit within the outer bounds.

        Parameters
        ----------
        limit : float, int, or Sequence
            The new limit to set.
        inverted : bool, optional
            Whether the limits are inverted. Defaults to None.

        Returns
        -------
        tuple of float
            The adjusted limit within the outer bounds.
        """
        inverted = inverted or self.inverted

        limit = self.parse_limit(limit, inverted)
        if limit[0] < self._outer_limit[0]:
            limit = (self._outer_limit[0], limit[1] + self._outer_limit[0] - limit[0])
            if limit[1] > self._outer_limit[1]:
                return self._outer_limit
            return limit
        if self._outer_limit[1] < limit[1]:
            limit = (limit[0] - (limit[1] - self._outer_limit[1]), self._outer_limit[1])
            if limit[0] < self._outer_limit[0]:
                return self._outer_limit
            return limit
        return limit

    def set_limit(self, limit, inverted=None):
        """
        Calculate a new limit within the outer bounds.

        Parameters
        ----------
        limit : float, int, Sequence[float | int]
            The new limit to set.
        inverted : bool, optional
            Whether the limits are inverted. Defaults to None.

        Returns
        -------
        tuple of float
            The adjusted limit within the outer bounds.
        """
        limit = self._new_limit(limit, inverted)
        self.check_order(limit, 'limit')
        self._limit = limit

    def width(self):
        """
        Calculate the width of the current limit.

        Returns
        -------
        float
            The width of the current limit.
        """
        return self._limit[1] - self._limit[0]

    def zoom(self, middle, factor):
        """
        Zoom into the limit by adjusting its width around a middle point.

        Parameters
        ----------
        middle : float
            The center point for zooming.
        factor : float
            The zoom factor to adjust the width.
        """
        new_width = factor*self.width()
        new_limits = (middle-0.5*new_width, middle+0.5*new_width)
        self.set_limit(new_limits, False)

    @property
    def limit(self):
        """
        Get the current limit, considering inversion.

        Returns
        -------
        tuple of float
            The current limit.
        """
        if self.inverted:
            return self._limit[::-1]
        else:
            return self._limit

    @property
    def outer_limit(self):
        """
        Get the outer bounds of the limit, considering inversion.

        Returns
        -------
        tuple of float
            The outer bounds of the limit.
        """
        if self.inverted:
            return self._outer_limit[::-1]
        else:
            return self._outer_limit

    @staticmethod
    def check_order(value, name):
        """
        Validate the order of the given bounds.

        Parameters
        ----------
        value : tuple of float
            The bounds to validate.
        name : str
            The name of the bounds for error messages.

        Raises
        ------
        ValueError
            If the bounds are invalid.
        """
        if value[0] == value[1]:
            raise ValueError(f"`{name}` bounds must be different")
        if value[1] < value[0]:
            raise ValueError(f"`{name}` bounds are in wrong order: {value}")

    def __repr__(self):
        """
        Return a string representation of the Limit instance.

        Returns
        -------
        str
            A string representation of the instance.
        """
        return f"Limit({self.limit=}, {self.outer_limit=})"
