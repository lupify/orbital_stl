# Hydrogen Orbital STL Generator

This Python script generates 3D printable STL files of hydrogen atomic orbitals. It calculates and visualizes quantum mechanical wavefunctions for different electron configurations, making abstract quantum concepts tangible through 3D printing.

## Features

- Generate STL files for hydrogen orbitals with any valid quantum numbers (n, l, m)
- Apply Gaussian blur for smoother visualization
- Automatic scaling for 3D printing
- Built-in labeling with quantum numbers
- Scale indicator in Angstroms
- Optional base generation for stable printing

## Requirements

```text
numpy
pyvista
matplotlib
scipy
```

Install requirements using:
```bash
pip install -r requirements.txt
```

## Usage

1. Configure the orbitals you want to generate by modifying the `orbitals_generate` list:
```python
orbitals_generate = [
    [1, 0, 0],  # 1s orbital
    [2, 0, 0],  # 2s orbital
    [2, 1, 0],  # 2p orbital
    # ... add more configurations as needed
]
```

2. Run the script to generate STL files in the output directory.

## Key Functions

### Wavefunction Calculations

- `hydrogen_wf(n, l, m, X, Y, Z)`: Calculates the hydrogen atom wavefunction for given quantum numbers and coordinates
  - Parameters: principal (n), azimuthal (l), and magnetic (m) quantum numbers, and 3D coordinate arrays
  - Returns: Complex array of wavefunction values

- `hydrogen_wf_range(n, l, m)`: Calculates characteristic range for the hydrogen wavefunction
  - Used for determining appropriate visualization scales
  - Returns decay length of the wavefunction

### Data Processing

- `norm_lin_range(values, min_val, max_val)`: Normalizes values to a specified linear range
  - Useful for scaling visualization data

- `convolve_nd(signal, kernel, fft_norm, range_norm)`: Performs N-dimensional convolution using FFT
  - Used for applying Gaussian blur to the orbital visualization

- `gaussian(x, mu, sigma)`: Calculates 1D Gaussian function
  - Used in blur kernel generation

- `gaussian_nd(coordinates, means, sigmas)`: Calculates N-dimensional Gaussian function
  - Creates blur kernels for smoothing orbital visualizations

## Output

The script generates two STL files for each orbital configuration:
1. `{n}_{l}_{m}.stl`: Just the orbital surface
2. `{n}_{l}_{m}_base.stl`: Orbital with attached base and labels

Files are saved in a directory named `orbitals_{contour_gen_field}`.

## Configuration Options

- `contour_gen_field`: Field used for contour generation ("wf_real_abs_conv" or "prob_conv")
- `contour_eval_frac`: Fraction of maximum value used for contour level (default: 0.2)
- `blur`: Gaussian blur amount (0 = no blur)
- `do_pv_example`: Use PyVista examples or calculate directly
- `show_orbital`: Show orbital visualization during generation
- `show_merge`: Show merged visualization with base

## Model Features

Each generated model includes:
- 3D visualization of the orbital
- Text label showing quantum numbers (n,l,m)
- Scale indicator in Angstroms
- Optional base for stable 3D printing
- Automatic scaling to 10mm maximum dimension

## Notes

- Special handling is implemented for spherically symmetric orbitals (l=0, m=0)
- Models are automatically scaled for 3D printing
- NaN values at the origin are automatically handled
- The script includes built-in error checking for valid quantum numbers

## License
None
