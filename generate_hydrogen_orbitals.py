import numpy as np
import pyvista as pv
from pyvista import examples
from numpy.fft import fftn, ifftn
import math
import matplotlib.pyplot as plt
import scipy.special
from scipy.special import sph_harm
from itertools import product
import os

def norm_lin_range(values, min_val=0, max_val=1):
    """
    Normalize values to a specified linear range.
    
    Args:
        values: Array of values to normalize
        min_val: Minimum value in the target range
        max_val: Maximum value in the target range
        
    Returns:
        Normalized array with values scaled to the target range
    """
    return (values - values.min())/(values.max() - values.min()) * (max_val - min_val) + min_val

def convolve_nd(signal, kernel, fft_norm="ortho", range_norm="im"):
    """
    N-dimensional convolution using Fast Fourier Transform.
    
    Args:
        signal: N-dimensional array representing the signal
        kernel: N-dimensional array representing the convolution kernel
        fft_norm: Normalization method for FFT ("ortho", "forward", or "backward")
        range_norm: Method for normalizing the output range ("re", "im", or None)
        
    Returns:
        N-dimensional array containing the convolution result
    """
    assert signal.shape == kernel.shape, "Signal and kernel must have the same shape"
    axes = np.arange(len(signal.shape))
    
    # Compute FFTs across all axes for signal and kernel
    signal_fft = fftn(signal, axes=axes, norm=fft_norm)
    kernel_fft = fftn(kernel, axes=axes, norm=fft_norm)
    kernel_fft_flipped = np.flip(kernel_fft, axes)
    
    # Compute convolution in frequency domain
    conv_result = ifftn(signal_fft * kernel_fft_flipped, axes=axes[::-1], norm=fft_norm)
    
    # Center the result
    for axis in range(len(conv_result.shape)):
        conv_result = np.roll(conv_result, -conv_result.shape[axis]//2+1, axis=axis)
    
    # Normalize the output range if requested
    if range_norm == "re":
        conv_result = norm_lin_range(conv_result, np.abs(signal).min(), np.abs(signal).max())
    elif range_norm == "im":
        # No normalization needed for "im" mode
        pass
    
    return conv_result

def hydrogen_wf_range(n, l, m=0):
    """
    Calculate a characteristic range for the hydrogen wavefunction.
    Used for determining appropriate scales for visualization.
    
    Args:
        n: Principal quantum number
        l: Azimuthal quantum number
        m: Magnetic quantum number (default: 0)
        
    Returns:
        Characteristic decay length of the wavefunction
    """
    R = 1.0
    rho = 2.0 * R / n
    return np.exp(-rho/2.0)
    
def hydrogen_wf(n, l, m, X, Y, Z):
    """
    Calculate the hydrogen atom wavefunction at given coordinates.
    
    Args:
        n: Principal quantum number
        l: Azimuthal quantum number
        m: Magnetic quantum number
        X, Y, Z: 3D coordinate arrays
        
    Returns:
        Complex array containing wavefunction values
    """
    # Convert to spherical coordinates
    R = np.sqrt(X**2 + Y**2 + Z**2)
    Theta = np.arccos(Z/R)
    Phi = np.arctan2(Y, X)
    
    rho = 2.0 * R / n
    
    # Calculate spherical harmonics
    spherical_harmonic = sph_harm(m, l, Phi, Theta)
    
    # Calculate associated Laguerre polynomial
    laguerre_poly = scipy.special.genlaguerre(n-l-1, 2*l+1)(rho)
    
    # Calculate normalization prefactor
    prefactor = np.sqrt((2.0/n)**3 * math.factorial(n-l-1) / (2.0*n*math.factorial(n+l)))
    
    # Combine all parts of the wavefunction
    wavefunction = prefactor * np.exp(-rho/2.0) * rho**l * spherical_harmonic * laguerre_poly
    
    # Replace NaN values with zeros (occurs at origin for some orbitals)
    return np.nan_to_num(wavefunction)
  
def gaussian(x, mu, sigma): 
    """
    Calculate a 1D Gaussian function.
    
    Args:
        x: Input coordinate array
        mu: Mean (center) of the Gaussian
        sigma: Standard deviation of the Gaussian
        
    Returns:
        Gaussian function values
    """
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x-mu)/sigma)**2)

def gaussian_nd(coordinates, means, sigmas):
    """
    Calculate an N-dimensional Gaussian function as product of 1D Gaussians.
    
    Args:
        coordinates: List of coordinate arrays [X, Y, Z, ...]
        means: List of means for each dimension
        sigmas: List of standard deviations for each dimension
        
    Returns:
        N-dimensional Gaussian function values
    """
    result = 1
    for coord, mean, sigma in zip(coordinates, means, sigmas):
        result *= gaussian(coord, mean, sigma)
    return result


################################
## Orbital STL Generator Code ##
################################

# This script generates STL files from hydrogen orbital visualizations
# for 3D printing or other visualization purposes.

# Define orbitals to generate by quantum numbers
# n: principal quantum number
# l: azimuthal quantum number
# m: magnetic quantum number

# Generate all valid orbitals for n=1,2,3,4
orbitals_generate = []
for n in [1, 2, 3, 4]:
    for l in range(0, n, 1):
        for m in range(-l, l+1, 1):
            orbitals_generate.append([n, l, m])

# Or, generate specific orbitals from this list
orbitals_generate = [
    [1, 0, 0], [2, 0, 0], [2, 1, 0], [4, 3, 1], 
    [4, 3, 2], [4, 3, 0], [3, 1, 1], [3, 2, 0], [3, 2, 2]
]

# Configuration parameters
contour_gen_field = "wf_real_abs_conv"  # Field to use for contour generation
# contour_gen_field = "prob_conv"        # Alternative: probability density
contour_eval_frac = 0.2                  # Fraction of max value to use for contour level
blur = 1.5                               # Gaussian blur amount (0 = no blur)
do_pv_example = True                     # Use PyVista examples or calculate directly
show_orbital = False                     # Show orbital visualization
show_merge = False                       # Show merged visualization with base

# Create output directory
output_dir = f"./orbitals_{contour_gen_field}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each orbital
for n, l, m in orbitals_generate:
    print(f"Processing orbital: n={n}, l={l}, m={m}")
    nstr, lstr, mstr = str(n).replace("-", "n"), str(l).replace("-", "n"), str(m).replace("-", "n")
    
    if do_pv_example:
        # Use PyVista's built-in hydrogen orbital examples
        orbital_data = examples.load_hydrogen_orbital(n, l, m)
        
        # Extract wavefunction data
        orbital_data_real_wf = orbital_data["real_wf"].reshape(100, 100, 100)
        orbital_data_wf = orbital_data["wf"].reshape(100, 100, 100)
        
        # Extract coordinate information
        xs = orbital_data.points.reshape(100, 100, 100, 3)[0, 0, :, 0]
        ys = orbital_data.points.reshape(100, 100, 100, 3)[0, :, 0, 1]
        zs = orbital_data.points.reshape(100, 100, 100, 3)[:, 0, 0, 2]
        
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        dr = orbital_data.spacing[0]
        dr_conv = hydrogen_wf_range(n, l, m)
        
        # Create convolution kernel (identity or Gaussian)
        if blur == 0 or n == 1:
            # No blur for n=1 or if blur is disabled
            conv_kernel = np.zeros(X.shape)
            conv_kernel[50, 50, 50] = 1  # Identity kernel
        else:
            # Gaussian blur kernel
            conv_kernel = gaussian_nd([X, Y, Z], [0, 0, 0], 
                                     [blur*dr_conv, blur*dr_conv, blur*dr_conv])
        
        # Apply convolution to wavefunction and probability density
        orbital_data["wf_real_abs_conv"] = np.abs(
            convolve_nd(np.abs(orbital_data_real_wf), conv_kernel)
        ).flatten()
        
        orbital_data["prob"] = (np.abs(orbital_data_wf)**2).flatten()
        orbital_data["prob_conv"] = np.abs(
            convolve_nd(np.abs(orbital_data_wf)**2, conv_kernel)
        ).flatten()
    else:
        # Calculate wavefunction directly (more accurate but slower)
        dr = hydrogen_wf_range(n, l, m) / 2
        r_min = -dr * 100
        r_max = dr * 100
        dr_conv = dr
        
        # Create coordinate grid
        x = y = z = np.linspace(r_min, r_max, 201)
        dr = x[1] - x[0]  # True grid spacing
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        
        # Calculate wavefunction
        wave_amp = hydrogen_wf(n, l, m, X, Y, Z)
        wave_amp_real = np.real(wave_amp)
        wave_amp_real_abs = np.abs(wave_amp_real)
        prob_amp = abs(wave_amp)**2
        
        # Create convolution kernel
        if blur == 0 or n == 1:
            conv_kernel = np.zeros(X.shape)
            conv_kernel[100, 100, 100] = 1  # Identity kernel
        else:
            conv_kernel = gaussian_nd([X, Y, Z], [0, 0, 0], 
                                     [blur*dr_conv, blur*dr_conv, blur*dr_conv])
        
        # Apply convolution
        wave_amp_real_abs_conv = np.abs(convolve_nd(wave_amp_real_abs, conv_kernel))
        prob_conv = np.abs(convolve_nd(prob_amp, conv_kernel))

        # Create PyVista image data
        orbital_data = pv.ImageData(
            dimensions=wave_amp.shape, 
            spacing=(dr, dr, dr), 
            origin=(r_min, r_min, r_min)
        )
        
        # Add data fields
        orbital_data["wf"] = wave_amp.flatten()
        orbital_data["wf_real"] = wave_amp_real.flatten()
        orbital_data["wf_real_abs"] = wave_amp_real_abs.flatten()
        orbital_data["prob_amp"] = prob_amp.flatten()
        orbital_data["prob_conv"] = prob_conv.flatten()
        orbital_data["wf_real_abs_conv"] = wave_amp_real_abs_conv.flatten()
    
    # Generate contour surface
    # Special case for spherically symmetric orbitals with n>1
    if n != 1 and l == 0 and m == 0:
        eval_at = orbital_data[contour_gen_field].max() * contour_eval_frac
        # Only use half of the orbital for symmetric cases
        orbital_data = orbital_data.extract_points(orbital_data.z <= 0, include_cells=True)
        orbital_contours = orbital_data.contour(
            [eval_at],
            scalars=np.abs(orbital_data[contour_gen_field]),
            method='contour',
        )
    else:
        eval_at = orbital_data[contour_gen_field].max() * contour_eval_frac
        orbital_contours = orbital_data.contour(
            [eval_at],
            scalars=np.abs(orbital_data[contour_gen_field]),
            method='contour',
        )
    
    # Scale the contours for 3D printing
    contour_bounds = np.array(orbital_contours.bounds)
    contour_max_range = np.max(contour_bounds[1::2] - contour_bounds[::2])
    scale = 10 / contour_max_range  # Scale to 10mm max dimension
    orbital_contours_scaled = orbital_contours.scale(scale)
    
    # Position the orbital above the base
    z_min = np.array(orbital_contours_scaled.bounds)[4]
    orbital_contours_scaled = orbital_contours_scaled.translate((0, 0, -z_min - 0.2 - 5))
    orbital_contours_scaled_mesh = orbital_contours_scaled.extract_surface()
    
    # Create text label for the orbital
    text_depth = 0.12
    orbit_text = pv.Text3D(
        f"n,l,m = {n},{l},{m}", 
        center=(0, 6.75, -5.0 + text_depth/2), 
        depth=text_depth, 
        height=1.5
    )
    orbit_text_mesh = orbit_text.extract_surface()
    
    # Create scale indicator (Angstrom unit box)
    unit_angstrom = scale * 1.88973  # Convert to Angstroms
    factor = 1.0
    
    # Adjust scale factor to get reasonable size
    while unit_angstrom <= 0.1:
        factor *= 10
        unit_angstrom *= factor
    while unit_angstrom >= 2:
        factor /= 10
        unit_angstrom /= factor
    
    # Create unit box
    box_dims = np.array((-.5, .5, -.5, .5, 0, 0)) * unit_angstrom + np.array((-7.0, -7.0, -7.0, -7.0, -5, -5))
    box_dims[5] = box_dims[5] + text_depth
    unit_box = pv.Box(bounds=box_dims, level=0, quads=True)
    unit_box_mesh = unit_box.extract_surface()
    
    # Create text for scale indicator
    factor_text = f"{factor} A"
    unit_box_text = pv.Text3D(
        factor_text, 
        center=(-5 + len(factor_text)/5, -7.0, -5.0 + text_depth/2), 
        depth=text_depth, 
        height=1.0
    )
    unit_box_text_mesh = unit_box_text.extract_surface()
    print(f"Scale factors: {scale:.4f}, {unit_angstrom:.4f}, {factor}")
    
    # Create base for the model
    base_dims = np.array((-8, 8, -8, 8, -5.4, -5))
    base = pv.Box(bounds=base_dims, level=0, quads=True)
    base_mesh = base.extract_surface()
    
    # Merge all components
    merged_mesh = pv.merge([
        orbital_contours_scaled_mesh, 
        orbit_text_mesh, 
        unit_box_mesh, 
        unit_box_text_mesh, 
        base_mesh
    ])
    
    # Visualize if requested
    if show_orbital:
        orbital_contours_scaled_mesh.plot()
    if show_merge:
        merged_mesh.plot()

    # Save STL files
    # Save complete model with base
    orbital_base_surface = merged_mesh.extract_surface()
    stl_path = f"{output_dir}/{nstr}_{lstr}_{mstr}_base.stl"
    orbital_base_surface.save(stl_path)
    
    # Save just the orbital
    orbital_surface = orbital_contours_scaled_mesh.extract_surface()
    stl_path = f"{output_dir}/{nstr}_{lstr}_{mstr}.stl"
    orbital_surface.save(stl_path)
