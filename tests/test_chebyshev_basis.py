# tests/test_chebyshev_basis.py

import numpy as np
import pytest
from numanalysislib.basis.chebyshev import ChebyshevBasis


class TestChebyshevBasis:
    
    def test_initialization(self):
        basis = ChebyshevBasis(degree=3)
        assert basis.degree == 3
        assert basis.n_dofs == 4
        assert basis.a == -1.0
        assert basis.b == 1.0
    
    def test_evaluate_basis(self):
        basis = ChebyshevBasis(degree=3)
        
        # T_0(x) = 1
        x = np.array([0.5])
        val = basis.evaluate_basis(0, x)
        assert val[0] == 1.0
        
        # T_1(x) = x
        val = basis.evaluate_basis(1, x)
        assert val[0] == 0.5
        
        # T_2(x) = 2x^2 - 1
        val = basis.evaluate_basis(2, x)
        np.testing.assert_allclose(val[0], 2*0.5**2 - 1)
    
    def test_evaluate_basis_vectorized(self):
        basis = ChebyshevBasis(degree=2)
        x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
        
        expected = 2 * x**2 - 1
        result = basis.evaluate_basis(2, x)
        np.testing.assert_allclose(result, expected, atol=1e-14)
    
    def test_fit_exact_reconstruction(self):
        degree = 3
        basis = ChebyshevBasis(degree=degree)
        
        true_coeffs = np.array([1.0, -2.0, 3.0, 1.0])
        x_nodes = basis.chebyshev_nodes()
        y_nodes = basis.evaluate(true_coeffs, x_nodes)
        
        fitted_coeffs = basis.fit(x_nodes, y_nodes)
        np.testing.assert_allclose(fitted_coeffs, true_coeffs, atol=1e-12)
    
    def test_chebyshev_nodes(self):
        basis = ChebyshevBasis(degree=3)
        
        nodes = basis.chebyshev_nodes(n=4, kind="roots")
        assert len(nodes) == 4
        assert np.all(nodes >= -1.0) and np.all(nodes <= 1.0)
        
        # Known values for n=4
        expected = np.cos(np.array([7, 5, 3, 1]) * np.pi / 8)
        np.testing.assert_allclose(nodes, np.sort(expected), atol=1e-14)
    
    def test_evaluate_clenshaw(self):
        basis = ChebyshevBasis(degree=3)
        coeffs = np.array([2.5, -1.25, 1.5, 0.25])
        
        x = np.linspace(-1, 1, 10)
        
        # Direct evaluation
        y_direct = np.zeros_like(x)
        for i, c in enumerate(coeffs):
            y_direct += c * basis.evaluate_basis(i, x)
        
        # Clenshaw evaluation
        y_clenshaw = basis.evaluate(coeffs, x)
        
        np.testing.assert_allclose(y_clenshaw, y_direct, atol=1e-14)
