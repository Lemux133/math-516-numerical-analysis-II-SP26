"""
Chebyshev polynomial basis module.

Implements the Chebyshev polynomials of the first kind on the reference
interval [-1, 1]. The basis is defined as T_n(x) = cos(n * arccos(x)).
"""

import warnings
import numpy as np
from numanalysislib.basis._abstract import PolynomialBasis


class ChebyshevBasis(PolynomialBasis):
    """
    Chebyshev polynomials of the first kind on the reference interval [-1, 1].

    The Chebyshev polynomials T_n(x) are defined as:
        T_n(x) = cos(n * arccos(x)), for x in [-1, 1].

    This basis is defined on the reference interval [-1, 1]. To use on a
    physical interval [a, b], wrap with AffinePolynomialBasis.

    Parameters
    ----------
    degree : int
        The maximum degree of the polynomial basis.
    """

    def __init__(self, degree: int) -> None:
        """
        Initialize the Chebyshev basis on the reference interval [-1, 1].

        Parameters
        ----------
        degree : int
            Maximum polynomial degree.
        """
        super().__init__(degree)

    def evaluate_basis(self, index: int, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the i-th Chebyshev polynomial T_i(x) at points x.

        Implements the definition: T_n(x) = cos(n * arccos(x)).

        Parameters
        ----------
        index : int
            The polynomial degree (0 <= index <= degree).
        x : np.ndarray
            Points at which to evaluate the basis.

        Returns
        -------
        np.ndarray
            Values of T_index(x) at each point, same shape as x.

        Raises
        ------
        ValueError
            If index is out of range for the basis degree.
        """
        if index < 0 or index > self.degree:
            raise ValueError(
                f"Basis index {index} out of range for degree {self.degree}"
            )

        x = np.asarray(x)

        # Warn for high-degree evaluation which may be unstable
        if index > 50:
            warnings.warn(
                f"Evaluating high-degree Chebyshev polynomial T_{index}(x) "
                f"using cos(n * arccos(x)) may be numerically unstable.",
                RuntimeWarning
            )

        return np.cos(index * np.arccos(x))

    def fit(self, x_nodes: np.ndarray, y_nodes: np.ndarray) -> np.ndarray:
        """
        Compute coefficients for Chebyshev interpolation.

        Solves the Vandermonde system V * c = y, where V_ij = T_j(x_i).

        Parameters
        ----------
        x_nodes : np.ndarray
            Interpolation node coordinates. Shape (n_dofs,).
        y_nodes : np.ndarray
            Function values at the nodes. Shape (n_dofs,).

        Returns
        -------
        np.ndarray
            Coefficients c such that sum(c_i * T_i(x)) interpolates at nodes.
            Shape (n_dofs,).

        Raises
        ------
        ValueError
            If number of nodes does not match n_dofs, or if matrix is singular.
        """
        x_nodes = np.asarray(x_nodes)
        y_nodes = np.asarray(y_nodes)

        if len(x_nodes) != self.n_dofs:
            raise ValueError(
                f"Expected {self.n_dofs} nodes for degree {self.degree}, "
                f"got {len(x_nodes)}"
            )

        # Build Vandermonde matrix using vectorized basis evaluation
        # Each column j corresponds to T_j evaluated at all x_nodes
        vander = np.column_stack([
            self.evaluate_basis(j, x_nodes) for j in range(self.n_dofs)
        ])

        # Check condition number
        cond_num = np.linalg.cond(vander)
        if cond_num > 1e12:
            warnings.warn(
                f"Chebyshev Vandermonde matrix is ill-conditioned "
                f"(cond={cond_num:.2e}). Results may be inaccurate.",
                RuntimeWarning
            )

        try:
            coefficients = np.linalg.solve(vander, y_nodes)
        except np.linalg.LinAlgError:
            raise ValueError(
                "Singular matrix encountered. Ensure nodes are distinct."
            )

        return coefficients

    def evaluate(self, coefficients: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the Chebyshev series p(x) = sum(c_i * T_i(x)).

        Uses vectorized operations: evaluates all basis functions at all
        points simultaneously, then sums with coefficients.

        Parameters
        ----------
        coefficients : np.ndarray
            Coefficients c_0, c_1, ..., c_n. Length must equal n_dofs.
        x : np.ndarray
            Points at which to evaluate the polynomial.

        Returns
        -------
        np.ndarray
            The evaluated polynomial values, same shape as x.

        Raises
        ------
        ValueError
            If number of coefficients does not match n_dofs.
        """
        if len(coefficients) != self.n_dofs:
            raise ValueError(
                f"Expected {self.n_dofs} coefficients, "
                f"got {len(coefficients)}"
            )

        x = np.asarray(x)

        # Build matrix where row i, column j = T_j(x_i)
        # Then multiply by coefficients and sum along columns
        basis_matrix = np.column_stack([
            self.evaluate_basis(j, x) for j in range(self.n_dofs)
        ])

        # Vectorized evaluation: sum over basis functions
        return basis_matrix @ coefficients

    def chebyshev_nodes(self, n: int, kind: str = "roots") -> np.ndarray:
        """
        Generate Chebyshev nodes on the reference interval [-1, 1].

        Chebyshev nodes are optimal for polynomial interpolation because they
        minimize the Runge phenomenon.

        Parameters
        ----------
        n : int
            Number of nodes to generate.
        kind : str, optional
            Type of Chebyshev nodes:
                - "roots": Chebyshev points of the first kind (roots of T_n)
                - "extrema": Chebyshev points of the second kind
            Default is "roots".

        Returns
        -------
        np.ndarray
            Array of n Chebyshev nodes in [-1, 1].

        Raises
        ------
        ValueError
            If n <= 0 or kind is invalid.
        """
        if n <= 0:
            raise ValueError(f"Number of nodes n must be positive, got {n}")

        if kind == "roots":
            # Roots of T_n: x_k = cos((2k-1)π/(2n)) for k = 1,...,n
            k = np.arange(1, n + 1)
            nodes = np.cos((2 * k - 1) * np.pi / (2 * n))
        elif kind == "extrema":
            # Extrema of T_n: x_k = cos(kπ/(n-1)) for k = 0,...,n-1
            if n == 1:
                nodes = np.array([0.0])
            else:
                k = np.arange(n)
                nodes = np.cos(k * np.pi / (n - 1))
        else:
            raise ValueError(
                f"kind must be 'roots' or 'extrema', got {kind}"
            )

        return nodes
