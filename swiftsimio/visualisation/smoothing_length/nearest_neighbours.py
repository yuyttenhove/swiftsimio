from numpy import cbrt, concatenate
from unyt.exceptions import UnitConversionError

from swiftsimio import SWIFTDataset, cosmo_array
from swiftsimio.objects import _cbrt_cosmo_factor
from swiftsimio.visualisation.smoothing_length.sph import get_hsml as get_hsml_sph


def cbrt_cosmo_array(arr):
    # TODO remove this hack once np.cbrt is supported by unyt
    units = (hasattr(arr, "units"), getattr(arr, "units", None))
    comoving = getattr(arr, "comoving", None)
    cosmo_factor = (
        hasattr(arr, "cosmo_factor"),
        getattr(arr, "cosmo_factor", None),
    )
    if units[0]:
        units_cbrt = units[1] ** (1.0 / 3.0)
    else:
        units_cbrt = None
    return cosmo_array(
        cbrt(arr.value),
        units=units_cbrt,
        comoving=comoving,
        cosmo_factor=_cbrt_cosmo_factor(cosmo_factor),
    )

def concatenate_cosmo_arrays(arrs):
    """
    Concatenate cosmo_arrays into a single cosmo_array object.
    This will try to make the necessary unit conversions and will fail
    if the cosmo_factors do not match.

    Note: from version 3.0.0 onwards unyt supports concatenating by providing itss
    own implemetation for np.concatenate using the "array function protocol"
    (see NEP 18: https://numpy.org/neps/nep-0018-array-function-protocol.html).
    We might want to do something similar?
    """
    units = [(hasattr(arr, "units"), getattr(arr, "units", None)) for arr in arrs]
    if any(u[0] for u in units):
        if not all(u[0] for u in units):
            raise RuntimeError("Trying to concatenate arrays with and without units")
        try:
            arrs = [arr.to(units[0][1]) for arr in arrs]
        except UnitConversionError:
            raise RuntimeError("Trying to concatenate arrays with incompatible units")

    comoving = [getattr(arr, "comoving", None) for arr in arrs]
    if any(c != comoving[0] for c in comoving):
        raise RuntimeError("Trying to concatenate arrays with inconsistent comoving flags")

    cosmo_factors = [(hasattr(arr, "cosmo_factor"), getattr(arr, "cosmo_factor", None)) for arr in arrs]
    if any(c[0] for c in cosmo_factors):
        if not all(c[0] for c in cosmo_factors):
            raise RuntimeError("Trying to concatenate arrays with and without cosmo_factor")
        if not all(c[1] == cosmo_factors[0][1] for c in cosmo_factors):
            raise RuntimeError("Trying to concatenate arrays with inconsistent cosmo_factors")

    return cosmo_array(
        concatenate([arr.value for arr in arrs]),
        units=units[0][1],
        comoving=comoving[0],
        cosmo_factor=cosmo_factors[0][1],
    )

def get_hsml(data: SWIFTDataset) -> cosmo_array:
    """
    This actually returns a 2D array with containing both the smoothing length
    and the cube root of the volume of the particles

    Parameters
    ----------
    data : SWIFTDataset
        The Dataset from which slice will be extracted

    Returns
    -------
    The extracted "smoothing lengths".
    """
    try:
        volumes = data.gas.volumes
        radii = cbrt_cosmo_array(volumes)
    except AttributeError:
        try:
            # Try computing the volumes explicitly?
            masses = data.gas.masses
            densities = data.gas.densities
            radii = cbrt_cosmo_array(masses / densities)
        except AttributeError:
            # Fall back to SPH behavior if above didn't work...
            radii = get_hsml_sph(data)
    hsml = get_hsml_sph(data)

    return concatenate_cosmo_arrays([hsml, radii])
