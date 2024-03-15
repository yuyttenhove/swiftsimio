from numpy import cbrt, concatenate

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

    return concatenate([hsml, radii])
