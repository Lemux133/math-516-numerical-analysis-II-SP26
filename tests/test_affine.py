import pytest
import numpy as np
from numanalysislib.basis._abstract import PolynomialBasis
from numanalysislib.basis.affine import AffinePolynomialBasis
from numanalysislib.basis.power import PowerBasis


def test_vector_pull_back():
    """
    Test if the interval [-4, 5] is successfully mapped to [-1, 1]
    """

    # define endpoints
    a = -4
    b = 5

    # make instance
    basis = PowerBasis(10)
    Affine = AffinePolynomialBasis(basis, a = a, b = b)

    # verify pull back maps interval
    physical_int = np.linspace(a, b, 100)
    reference_int = np.linspace(0, 1, 100)

    mapped_physical_int = Affine.pull_back(physical_int)

    np.testing.assert_allclose(mapped_physical_int, reference_int)

def test_vector_push_forward():
    """
    Test if the interval [-1, 1] is successfully mapped to [-4, 5]
    """
    # define endpoints
    a = -4
    b = 5

    # make instance
    basis = PowerBasis(10)
    Affine = AffinePolynomialBasis(basis, a = a, b = b)

    # verify push forward maps interval
    physical_int = np.linspace(a, b, 100)
    reference_int = np.linspace(0, 1, 100)

    mapped_reference_int = Affine.push_forward(reference_int)

    np.testing.assert_allclose(mapped_reference_int, physical_int)

def test_failure():
    """
    Check for failure
    """
    a = -2
    b = -4

    with pytest.raises(ValueError, match = "a must be greater than b"):
        basis = PowerBasis(10)
        Affine = AffinePolynomialBasis(basis, a = a, b = b)


