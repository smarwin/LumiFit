# Spectral Analysis Tool

This Python script provides a comprehensive solution for analyzing spectral data, particularly focusing on emission and excitation spectra. It includes functionalities for importing data, smoothing spectra, performing Gaussian fits, and visualizing results with a focus on spectral features like peak maxima, full width at half maximum (FWHM), and area under the curve.

## Features

- **Data Import**: Supports importing spectral data from text files.
- **Data Processing**: Includes smoothing of spectra using the Savitzky-Golay filter.
- **Spectral Analysis**: Performs Gaussian fitting on emission spectra to determine peak characteristics.
- **Visualization**: Generates plots of processed spectra and fits, including a visual representation of the FWHM and peak maxima.

## Installation

To run this script, ensure you have Python installed on your system along with the following packages:

- numpy
- scipy
- matplotlib

Install the required packages using the following command:

```
pip install -r requirements.txt
```

## Usage

1. Prepare your spectral data files in a plain text format. The script expects two types of data files:
   - Excitation data file
   - Emission data file
2. Adjust the script parameters to match your data and analysis needs. Important parameters include:
   - File paths for the excitation and emission data
   - Smoothing parameters
   - Fitting parameters and options
3. Run the script with Python:

```
python main.py
```

4. The script will process the data, perform any requested fits, and generate plots of the results. Plots will be saved in the specified directory.

## Customization

- **Data Files**: Modify `exc_path` and `emi_path` to point to your data files.
- **Plot Appearance**: Adjust plot parameters such as colors and linewidths within the `pp` dictionary.
- **Analysis Parameters**: Set `smooth`, `fit`, `initial_guess`, and other parameters according to your analysis requirements.

## Example

The provided example demonstrates a full workflow from importing data, processing it, fitting Gaussian curves, and plotting the results. To adapt the example to your data, modify the file paths and analysis parameters as needed.

## Contributing

Feel free to fork this project and submit pull requests with improvements or report any issues you encounter.
