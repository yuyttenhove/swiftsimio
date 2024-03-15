from typing import Tuple

import numpy as np
from numpy import (
    float64,
    float32,
    ndarray,
    int64,
    int32,
    arange,
    full,
    inf,
    zeros,
    empty,
    argmin,
    take_along_axis,
)

from swiftsimio.accelerated import jit, prange, NUM_THREADS


@jit(nopython=True, fastmath=True)
def nn_slice_scatter_core(
    x_coords: float64,
    y_coords: float64,
    z_coords: float64,
    z_slice: float,
    hsml: float64,
    xres: int,
    yres: int,
    box_x: float = 0.0,
    box_y: float = 0.0,
    box_z: float = 0.0,
    safety_factor: float = 1.5,
) -> Tuple[ndarray, ndarray]:
    """
    Creates two 2D numpy arrays with the same shape as the slice image,
    containing the indices of and distances to the nearest particle for each pixel.

    Parameters
    ----------
    x_coords : array of float64
        x-positions of the particles. Must be bounded by [0, 1].
    y_coords : array of float64
        y-positions of the particles. Must be bounded by [0, 1].
    z_coords : array of float64
        z-positions of the particles. Must be bounded by [0, 1].
    z_slice : float64
        the position at which we wish to create the slice
    xres : int
        the number of pixels in x direction.
    yres : int
        the number of pixels in the y direction.
    box_x: float
        box size in x, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).
    box_y: float
        box size in y, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).
    box_z: float
        box size in z, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).
    safety_factor: float
        Factor to multiply the safety radius with to check whether to consider particles

    Returns
    -------
    Tuple[ndarray, ndarray] of types (resp.) int64 and float32
        output arrays used to create the slice image
    """
    part_indices = arange(x_coords.shape[0])
    nn_idx = full((xres, yres), -1, dtype=int64)
    nn_d2 = full((xres, yres), inf, dtype=float32)

    max_idx_x = xres - 1
    max_idx_y = yres - 1

    # We need this for combining with the x_pos and y_pos variables.
    float_res = float64(max(xres, yres))
    pixel_width = 1.0 / float_res

    i_box_x = 1.0 / box_x if box_x > 0.0 else 0.0
    i_box_y = 1.0 / box_y if box_y > 0.0 else 0.0
    i_box_z = 1.0 / box_z if box_z > 0.0 else 0.0

    if box_x == 0.0:
        xshift_min = 0
        xshift_max = 1
    else:
        xshift_min = -1
        xshift_max = 2
    if box_y == 0.0:
        yshift_min = 0
        yshift_max = 1
    else:
        yshift_min = -1
        yshift_max = 2
    if box_z == 0.0:
        zshift_min = 0
        zshift_max = 1
    else:
        zshift_min = -1
        zshift_max = 2

    for p_idx, x_pos_original, y_pos_original, z_pos_original, h in zip(
        part_indices, x_coords, y_coords, z_coords, hsml
    ):
        # Compute kernel width in pixels
        safety_radius = h * 2.0
        kernel_width = int32(safety_radius / pixel_width) + 1

        # loop over periodic copies of the particle
        for xshift in range(xshift_min, xshift_max):
            for yshift in range(yshift_min, yshift_max):
                for zshift in range(zshift_min, zshift_max):
                    x_pos = x_pos_original + xshift * box_x
                    y_pos = y_pos_original + yshift * box_y
                    z_pos = z_pos_original + zshift * box_z
                    # dz_wrapped = dz - width_z * round(dz / width_z)
                    dz = z_pos - z_slice - box_z * round((z_pos - z_slice) * i_box_z)
                    if abs(dz) > safety_factor * safety_radius:
                        # No overlap in z, we can skip this particle
                        continue
                    dz2 = dz ** 2

                    # Calculate the pixel that this particle lives above; use 64 bits
                    # resolution as this is the same type as the positions
                    cell_x = int32(float_res * x_pos)
                    cell_y = int32(float_res * y_pos)

                    # No overlap in x, y  we can skip this particle
                    if (
                        cell_x < -kernel_width
                        or cell_x > max_idx_x + kernel_width
                        or cell_y < -kernel_width
                        or cell_y > max_idx_y + kernel_width
                    ):
                        continue

                    # Loop over pixels in the kernel
                    x_range = (
                        max(0, cell_x - kernel_width),
                        min(cell_x + kernel_width, max_idx_x + 1),
                    )
                    y_range = (
                        max(0, cell_y - kernel_width),
                        min(cell_y + kernel_width, max_idx_y + 1),
                    )
                    for px in range(*x_range):
                        for py in range(*y_range):
                            if nn_d2[px, py] < dz2:
                                # Neighbouring pixel is guaranteed to have smaller distance
                                continue
                            # Now we update the distance and idx of this pixel
                            dx = x_pos - (px + 0.5) * pixel_width
                            dx = dx - box_x * round(dx * i_box_x)
                            dy = y_pos - (py + 0.5) * pixel_width
                            dy = dy - box_y * round(dy * i_box_y)
                            distance_2 = float32(dx * dx + dy * dy + dz2)
                            if nn_d2[px, py] > distance_2:
                                nn_idx[px, py] = p_idx
                                nn_d2[px, py] = distance_2

    # return nn_idx, nn_d2

    # Now perform a flood fill to treat the remaining pixels
    # Create the ringbuffer containing the indices of the pixels to be checked
    buffer_size = 4 * xres * yres
    ring_buffer = zeros(buffer_size, dtype=int64)
    buffer_head = 0
    buffer_tail = 0

    # Fill buffer with initialized pixels whose neighbours are not all equal to itself
    for x in range(xres):
        for y in range(yres):
            i = nn_idx[x, y]
            if i == -1:
                continue
            for dpx in range(-1, 2):
                for dpy in range(-1, 2):
                    nx = (x + dpx) % xres
                    ny = (y + dpy) % yres
                    if nn_idx[nx, ny] != i:
                        ring_buffer[buffer_tail] = x * yres + y
                        buffer_tail += 1
                        break
                else:
                    continue  # only executed if the inner loop did NOT break
                break  # break the outer loop also if the inner loop was broken

    # Now we flood fill the remaining pixels
    while buffer_head != buffer_tail:
        p = ring_buffer[buffer_head % buffer_size]
        buffer_head += 1
        x = p // yres
        y = p % yres

        i = nn_idx[x, y]
        xi = x_coords[i]
        yi = y_coords[i]

        dz = z_coords[i] - z_slice
        # dz_wrapped = dz - width_z * round(dz / width_z)
        dz = dz - box_z * round(dz * i_box_z)
        dz2 = dz ** 2

        # Loop over neighbouring pixels
        for dnx in range(-1, 2):
            for dny in range(-1, 2):
                nx = (x + dnx) % xres
                ny = (y + dny) % yres
                if nn_idx[nx, ny] == i or nn_d2[nx, ny] < dz2:
                    # Neighbouring pixel has same value or is guaranteed to have smaller distance;
                    # nothing to be checked
                    continue
                dx = xi - (nx + 0.5) * pixel_width
                dy = yi - (ny + 0.5) * pixel_width
                # dx_wrapped = dx - width_x * round(dx / width_x)
                dx = dx - box_x * round(dx * i_box_x)
                dy = dy - box_y * round(dy * i_box_y)  # similarly
                d2 = float32(dx ** 2 + dy ** 2 + dz2)
                # Replace the values of the neighbouring pixel if the new distance is smaller
                if d2 < nn_d2[nx, ny]:
                    nn_idx[nx, ny] = i
                    nn_d2[nx, ny] = d2
                    ring_buffer[buffer_tail % buffer_size] = nx * yres + ny
                    buffer_tail += 1

    return nn_idx, nn_d2


@jit(nopython=True, fastmath=True)
def slice_scatter(
    x: float64,
    y: float64,
    z: float64,
    m: float32,
    h: float32,
    z_slice: float,
    xres: int,
    yres: int,
    box_x: float = 0.0,
    box_y: float = 0.0,
    box_z: float = 0.0,
) -> ndarray:
    """
    Parallel implementation of slice_scatter

    Creates a 2D numpy array (image) of the given quantities of all particles in
    a data slice including periodic boundary effects.

    Parameters
    ----------
    x : array of float64
        x-positions of the particles. Must be bounded by [0, 1].
    y : array of float64
        y-positions of the particles. Must be bounded by [0, 1].
    z : array of float64
        z-positions of the particles. Must be bounded by [0, 1].
    m : array of float32
        masses (or otherwise weights) of the particles
    h : array of float32
        smoothing lengths of the particles and cube roots of the volumes
        of the particles. I.e.: this array is twice as long as the m array.
    z_slice : float64
        the position at which we wish to create the slice
    xres : int
        the number of pixels in x direction.
    yres : int
        the number of pixels in the y direction.
    box_x: float
        box size in x, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).
    box_y: float
        box size in y, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).
    box_z: float
        box size in z, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).
    Returns
    -------
    ndarray of float32
        output array for the slice image

    See Also
    --------
    scatter : Create 3D scatter plot of SWIFT data
    scatter_parallel : Create 3D scatter plot of SWIFT data in parallel
    slice_scatter_parallel : Create scatter plot of a slice of data in parallel
    """

    numpart = x.shape[0]
    h, r = h[:numpart], h[numpart:]

    nn_idx, _ = nn_slice_scatter_core(
        x, y, z, z_slice, h, xres, yres, box_x, box_y, box_z
    )
    values = m / r ** 3

    return values[nn_idx.flatten()].reshape(xres, yres)


@jit(nopython=True, fastmath=True, parallel=True)
def slice_scatter_parallel(
    x: float64,
    y: float64,
    z: float64,
    m: float32,
    h: float32,
    z_slice: float,
    xres: int,
    yres: int,
    box_x: float = 0.0,
    box_y: float = 0.0,
    box_z: float = 0.0,
) -> ndarray:
    """
    Parallel implementation of slice_scatter

    Creates a 2D numpy array (image) of the given quantities of all particles in
    a data slice including periodic boundary effects.

    Parameters
    ----------
    x : array of float64
        x-positions of the particles. Must be bounded by [0, 1].
    y : array of float64
        y-positions of the particles. Must be bounded by [0, 1].
    z : array of float64
        z-positions of the particles. Must be bounded by [0, 1].
    m : array of float32
        masses (or otherwise weights) of the particles
    h : array of float32
        smoothing lengths of the particles and cube roots of the volumes
        of the particles. I.e.: this array is twice as long as the m array.
    z_slice : float64
        the position at which we wish to create the slice
    xres : int
        the number of pixels in x direction.
    yres : int
        the number of pixels in the y direction.
    box_x: float
        box size in x, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).
    box_y: float
        box size in y, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).
    box_z: float
        box size in z, in the same rescaled length units as x, y and z.
        Used for periodic wrapping (if not 0).

    Returns
    -------
    ndarray of float32
        output array for the slice image

    See Also
    --------
    scatter : Create 3D scatter plot of SWIFT data
    scatter_parallel : Create 3D scatter plot of SWIFT data in parallel
    slice_scatter_parallel : Create scatter plot of a slice of data in parallel
    """
    # Same as scatter, but executes in parallel! This is actually trivial,
    # we just make NUM_THREADS images and reduce them by picking the minimal distances.

    # Same as scatter, but executes in parallel! This is actually trivial,
    # we just make NUM_THREADS images and reduce them by picking the minimal distances.

    numpart = x.shape[0]
    h, r = h[:numpart], h[numpart:]

    number_of_particles = x.size
    core_particles = number_of_particles // NUM_THREADS

    output_idx = empty((xres * int(yres), NUM_THREADS), dtype=int64)
    output_dist_sqrd = empty((xres * int(yres), NUM_THREADS), dtype=float32)

    for thread in prange(NUM_THREADS):
        # Left edge is easy, just start at 0 and go to 'final'
        left_edge = thread * core_particles

        # Right edge is harder in case of left over particles...
        right_edge = thread + 1

        if right_edge == NUM_THREADS:
            right_edge = number_of_particles
        else:
            right_edge *= core_particles

        nn_idx, nn_dist_sqrd = nn_slice_scatter_core(
            x[left_edge:right_edge],
            y[left_edge:right_edge],
            z[left_edge:right_edge],
            z_slice=z_slice,
            hsml=h[left_edge:right_edge],
            xres=xres,
            yres=yres,
            box_x=box_x,
            box_y=box_y,
            box_z=box_z,
        )
        output_idx[:, thread] = nn_idx.flatten() + left_edge
        output_dist_sqrd[:, thread] = nn_dist_sqrd.flatten()

    amin = argmin(output_dist_sqrd, axis=1)[:, np.newaxis]
    nn_idx = take_along_axis(output_idx, amin, axis=1)[:, 0]

    values = m / r ** 3
    return values[nn_idx].reshape(xres, yres)
