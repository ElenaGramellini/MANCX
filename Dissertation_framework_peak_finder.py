"""
Dissertation: Axion Searches at ADMX and MANCX, Lukas Wystemp, 2024

This script is used to read in data from the MANCX test stand and plot the fourier transform. It automatically 
detects significant axion-like peaks in the data and calculates the signal to noise ratio. The script also 
calculates the signal to noise ratio. Artificial gaussian noise or coherent sinusoidal noise can be
added to the data and the accuracy of the script can be evaluated. 

Input data should be of the form 'RunFrequency.csv'. For example, 'A75.csv' will be read as run A with
resonant frequency of 75 MHz. If no run number is specified, the run number is set to 0. 

The script performs the following actions:
1. Import data from all valid files
2. Sort dataframe by run and frequency
3. Specify noise level which can be added to the data. Set to 0 by default
4. Calculate the average background. Freqeuncies need to be specified (in this case all except 75 MHz are used).
5. Plot the data for each resonant frequency as Voltage-time and Frequency-Amplitude
6. Calculate the signal to noise ratio for the 75 MHz peak. Requires manual input. Turned off by default
7. Find peak algorithm
    7.1 Split the data into 10 sections
    7.2 For each section, find peaks using the find_peaks function
    7.3 Calculate the noise floor and noise standard deviation
    7.4 Set thresholds for peak detection
    7.5 Calculate the p-value for each peak
    7.6 Estimate the signal to noise ratio for each peak
    7.7 Perform Benjamini-Hochberg FDR correction
8. Plot the flagged peaks for each resonant frequency

"""


# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from scipy.optimize import curve_fit
from scipy.signal import welch
from matplotlib.patches import Rectangle
from scipy.integrate import quad
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import rayleigh
from statsmodels.stats.multitest import multipletests

# Folder with axion data
folder_path = '/Users/lukaswystemp/Documents/University/Semester 5/Dissertation/Data_Axion'


def list_files_in_folder(folder_path):
    """
    Lists all CSV files in the specified folder.
    Args:
        folder_path (str): The path to the folder where the files are located.
    Returns:
        list: A list of filenames (str) that end with '.csv' in the specified folder.
    Raises:
        Exception: If an error occurs during file import, it prints the error message and exits the program.
    """

    try:
        return [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    except Exception as e:
        print(f"An error occurred during file import: {e}")
        sys.exit()


def import_data(file):
    """
    Imports data from a CSV file, processes it, and returns a pandas DataFrame.
    The function performs the following steps:
    1. Parses the filename to extract the run number and resonance frequency.
    2. Reads data from the specified CSV file, skipping the header and removing rows with NaN values.
    3. Adds columns for the run number and resonance frequency.
    4. Converts the data into a pandas DataFrame and adjusts the column names.
    5. Adjusts the 'time' column to start from zero.
    Parameters:
    file (str): The name of the CSV file to import.
    Returns:
    pd.DataFrame: A DataFrame containing the processed data with columns ['run', 'freq', 'time', 'voltage'].
    """

    run_number, res_freq = parse_filename(file)

    # Read data from CSV file
    df = np.genfromtxt(f'{folder_path}/{file}', delimiter=',', skip_header=1)
    df = df[~np.isnan(df).any(axis=1)]

    # Add run number and frequency columns
    run_column = np.full((df.shape[0], 1), run_number)
    freq_column = np.full((df.shape[0], 1), res_freq)
    
    df = np.column_stack((freq_column, df))
    df = np.column_stack((run_column, df))
    
    # Convert to DataFrame and adjust columns
    df = pd.DataFrame(df, columns=['run', 'freq', 'time', 'voltage'])
    df[['freq', 'time', 'voltage']] = df[['freq', 'time', 'voltage']].apply(pd.to_numeric)

    # Adjust time column to start from zero
    df['time'] = df['time'] - df['time'][0]

    return df


def calculate_run_map(files_list):
    """
    Generates a mapping of run numbers to their respective indices.
    Args:
        files_list (list of str): A list of file names.
    Returns:
        dict: A dictionary where keys are run numbers (as strings) and values are their respective indices.
    """

    run_numbers = set()
    for file in files_list:
        run_number, _ = parse_filename(file)
        run_numbers.add(str(run_number))
    run_numbers = sorted(run_numbers, key=lambda x: (0 if x == 0 else 1, x))
    run_map = {run: idx for idx, run in enumerate(run_numbers)}
    return run_map


def parse_filename(file):
    """
    Parses the given filename to extract the run number and resonance frequency.
    Args:
        file (str): The filename to parse.
    Returns:
        tuple: A tuple containing the run number (str) and the resonance frequency (int).
    """

    res_freq = ''.join(filter(str.isdigit, file))
    if file[0].isalpha():
        run_number = file[0]
    else:
        run_number = 0
    return str(run_number), int(res_freq)

def add_legend_background(ax, legend, color="white"):
    """
    Matplotlib has a bug where the legend background is not drawn when using seaborn-v0_8 style. 
    This function adds a white background to the legend. If the bug is fixed, this function can be removed.
    """

    bbox = legend.get_window_extent()
    bbox = bbox.transformed(ax.transAxes.inverted())
    
    ax.add_patch(
        Rectangle(
            (bbox.x0, bbox.y0),  
            bbox.width,          
            bbox.height,         
            transform=ax.transAxes, 
            color=color,         
            zorder=legend.get_zorder() - 1,        
            linewidth=0       
        )
    )


def plot_res_freqs(df):
    """
    Plots the resonant frequencies and identifies significant peaks in the data.
    Parameters:
    df (pandas.DataFrame): DataFrame containing the data with columns 'freq', 'run', 'time', and 'voltage'.
    The function performs the following steps:
    1. Calculates the average background using the `calculate_background` function.
    2. Initializes a DataFrame to store significant peaks.
    3. Iterates over each unique resonant frequency in the DataFrame.
    4. For each resonant frequency:
        a. Filters the DataFrame for the current frequency.
        b. Plots the voltage-time data for each run.
        c. Adds Gaussian noise to the voltage data.
        d. Calculates the Fourier transform of the voltage data.
        e. Plots the average Fourier transform.
        f. Identifies peaks in the Fourier transform data.
        g. Calculates p-values for the identified peaks.
        h. Performs Benjamini-Hochberg FDR correction to identify significant peaks.
        i. Stores significant peaks in the initialized DataFrame.
    5. Plots the section edges and flagged peaks.
    6. Prints the DataFrame containing significant peaks.
    Returns:
    None
    """

    # Calculate the average background
    background = calculate_background(df)

    # Initialise dataframe which will store the significnat peaks
    flag = pd.DataFrame(columns = ['res_freq', 'x_pos', 'y_pos', 'p_value', 'snr'])

    # Plot data for each resonant frequency
    for i in df["freq"].unique():
        df_f = df[df["freq"] == i]

        plt.style.use("seaborn-v0_8")
        fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns
        
        length = int(np.shape(df_f[df_f['run']=='A'])[0]/2 - 1)
        average_df = np.zeros((len(df_f["run"].unique()), length))

        num = 0
        for idx, j in enumerate(df_f["run"].unique()):
            # Plot all runs
            if True:
                df_r = df_f[df_f["run"] == j]

                # Add noise to the data, set to 0 by default
                df_r.loc[:, "voltage"] = add_gaussian_noise(df_r["voltage"])
                #df_r.loc[:, "voltage"] = add_sinusoidal_noise(df_r["time"], df_r["voltage"], 2*np.pi*0.8e9)

                # Plot voltage time data
                axs[0].scatter(df_r["time"], df_r["voltage"], label=f'{j}', s = 4)

                # Calculate the fourier transform
                sample_spacing = 1.562000e-10
                N = N = len(df_r['voltage'])
                
                y = np.array(df_r['voltage'])
                yf = fft(y)
                xf = fftfreq(N, sample_spacing)#[:N//2]

                yf = yf[xf > 0]
                freqs = xf[xf > 0]

                average_df[idx] = np.abs(yf)

                #average_df[idx] -= background


            elif num < 1:
                # Consider only the first run, This is now deprecated
                df_r_0 = df_f[df_f["run"] == '0']
                df_r_0.loc[:, "voltage"] = add_gaussian_noise(df_r_0["voltage"])
                
                axs[0].scatter(df_r_0["time"], df_r_0["voltage"], label=f'{j}', s = 3)


                sample_spacing = df_r_0['time'][1] - df_r_0['time'][0]
                N = N = len(df_r_0['voltage'])

                y = np.array(df_r_0['voltage'])
                yf = fft(y)
                xf = fftfreq(N, sample_spacing)[:N//2]

                average_df[idx] = np.abs(yf)
                
                axs[0].set_xlim(0, df_r_0["time"].max())
                num += 1


        # Set plot parameters
        axs[0].grid(True)
        axs[0].legend(loc='upper right', title='Run', facecolor = 'white')
        axs[0].set_xlim(0, np.max(df_r["time"]))
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Voltage (V)')
        legend1 = axs[0].legend(loc = 'upper right')
        add_legend_background(axs[0], legend1, color="white")
        
        # Plot the average fourier transform
        average = np.mean(average_df, axis = 0)
        average = np.abs(average)
        std = np.std(average_df, axis = 0)

        # Plot the average background
        axs[1].errorbar(freqs, average[:N//2], yerr = std[:N//2], label = 'Average', linestyle = '--', 
                        markersize = 2, fmt = 'o', capsize = 2, color = 'black', ecolor = 'darkviolet')
        axs[1].grid(True)
        #axs[1].set_xlim(0, 1e9)
        axs[1].set_ylim(0, 0.25)
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Amplitude')

        # Calculate the SNR for a known peak, otherwise SNR is automatically estimated
        #_, _ = SNR(freqs, average, axs)

        # Set plot parameters
        fig.suptitle(f'Resonance Frequency: {i} MHz', fontsize=16, fontweight='bold')
        legend2 = axs[1].legend(loc = 'upper right')
        add_legend_background(axs[1], legend2, color="white")


        # Find peaks
        n_sections = 10  # Number of sections, adjust as needed
        freq_min = np.min(freqs)
        freq_max = np.max(freqs)
        section_edges = np.linspace(freq_min, freq_max, n_sections + 1)

        # Initialise arrays to store flagged peaks
        flagged_local_frequencies = np.array([])
        flagged_local_amps = np.array([])
        flagged_p_values = np.array([])
        flagged_snr_values = np.array([])


        for k in range(n_sections):
            idx_section = (freqs >= section_edges[k]) & (freqs < section_edges[k+1])
            freqs_section = freqs[idx_section]
            average_section = average[idx_section]

            if len(average_section) == 0:
                break  

            # Noise floor and standard deviation
            noise_floor = np.median(average_section)
            noise_std = np.std(average_section)

            # Input parameters for peak finding algorithm. Adjust as needed
            threshold_factor = 3 
            height_threshold = noise_floor + threshold_factor * noise_std
            prominence_threshold = noise_std * threshold_factor 

            peaks, _ = find_peaks(average_section, height = height_threshold, prominence=prominence_threshold)

            # Calculate p-values
            peaks_pos = freqs_section[peaks]
            peaks_height = average_section[peaks]

            noise_indices = np.ones(len(average_section), dtype=bool)
            noise_indices[peaks] = False
            noise_data = average_section[noise_indices]

            sigma_hat = np.sqrt(np.sum(noise_data**2) / (2 * len(noise_data)))
            p_values = rayleigh.sf(peaks_height, scale=sigma_hat)

            # Store flagged peaks
            flagged_local_frequencies = np.append(flagged_local_frequencies, peaks_pos)
            flagged_local_amps = np.append(flagged_local_amps, peaks_height)
            flagged_p_values = np.append(flagged_p_values, p_values)

            # Roughly estimate the SNR
            if len(peaks) > 0:
                noise_power = np.sum(noise_floor**2)
                signal_power = np.sum(peaks_height**2)

                snr = signal_power / noise_power
                flagged_snr_values = np.append(flagged_snr_values, snr)

        # Plot Section edges and flagged peaks
        axs[1].scatter(section_edges, np.ones((len(section_edges)))*0.025, color = 'blue', 
                       label = 'Section Edges', s = 2)
        axs[1].scatter(flagged_local_frequencies, flagged_local_amps, color = 'grey', 
                       label = 'Flagged Peaks')

        # Perform Benjamini-Hochberg FDR correction
        if len(flagged_p_values) > 0:
            alpha = 0.01 #3.726653e-6 # Significance level

            rejected, pvals_corrected, _, _ = multipletests(flagged_p_values, alpha=alpha, method='fdr_bh')

            significant_freqs = flagged_local_frequencies[rejected]
            significant_amps = flagged_local_amps[rejected]
            significant_p_values = flagged_p_values[rejected]

            axs[1].scatter(significant_freqs,significant_amps,color='red',label='Significant Peaks')

            # Store significant peaks
            for freq, amp, p, s in zip(significant_freqs, significant_amps, significant_p_values, 
                                       flagged_snr_values):
                
                new_row = pd.DataFrame([{'res_freq': i, 'x_pos': freq, 'y_pos': amp, 'p_value': p, 'snr': s}])
                flag = pd.concat([flag, new_row], ignore_index=True)

        # Set plot parameters
        axs[1].legend(loc = 'upper right')
        plt.show()

    # Print significant peaks
    print(flag)


def calculate_background(df):
    """
    Calculate the background noise from a DataFrame containing frequency and voltage data.
    This function processes the input DataFrame to calculate the average background noise
    for each unique frequency (excluding the frequency 75). It adds sinusoidal noise to the
    voltage data, performs a Fast Fourier Transform (FFT) on the voltage data, and then
    averages the FFT results to compute the background noise. 

    WARNING: 
    This is not an automatic function. Needs to be specified manually which frequencies are considered
    background and which are not

    Returns:
        numpy.ndarray: An array representing the average background noise.
    """

    background = []
    for i in df["freq"].unique():
        if i != 75:
            df_f = df[df["freq"] == i]


            length = int(np.shape(df_f[df_f['run']=='A'])[0]/2 - 1)
            average_df = np.zeros((len(df_f["run"].unique()), length))

            for idx, j in enumerate(df_f["run"].unique()):
                #vt
                df_r = df_f[df_f["run"] == j]
                #df_r.loc[:, "voltage"] = add_noise(df_r["voltage"])
                df_r.loc[:, "voltage"] = add_sinusoidal_noise(df_r["time"], df_r["voltage"], 2*np.pi*0.8e9)
                sample_spacing = 1.562000e-10
                N =  len(df_r['voltage'])
                    
                y = np.array(df_r['voltage'])
                yf = fft(y)
                xf = fftfreq(N, sample_spacing)#[:N//2]

                yf = yf[xf > 0]

                average_df[idx] = np.abs(yf)

            average = np.mean(average_df, axis = 0)
            average = np.abs(average)

            background.append(average)
    
    background = np.array(background)
    background = np.mean(background, axis = 0)

    return background


def SNR(freqs, average, axs):
    """
    Calculate the Signal to Noise Ratio (SNR) for a specific known peak. Requires input of the peak
    frequency, amplitude, and standard deviation. Funtion will attempt a Gaussian fit

    WARNING:
    The function assumes a Gaussian signal centered at 75 MHz with a standard deviation of 1.1 MHz.
    This is because for the MANCX test set data this was the case. As such, the function is not 
    generalised and not used in the main script

        
    Parameters:
    freqs (numpy.ndarray): Array of frequency values.
    average (numpy.ndarray): Array of averaged FFT values corresponding to the frequencies.
    axs (matplotlib.axes.Axes): Axes object for plotting (not used in the function).
    Returns:
    tuple: A tuple containing the SNR value and its uncertainty.
    The function performs the following steps:
    1. Defines the left and right sections around a central frequency (75 MHz) to isolate the signal.
    2. Creates a mask to separate the signal and noise frequencies.
    3. Calculates the noise power by averaging the FFT values outside the signal region.
    4. Fits a Gaussian to the data to estimate the signal power.
    5. Computes the SNR as the ratio of signal power to noise power.
    6. Estimates the uncertainty in the SNR calculation.

    """
    # Signal to noise ratio
    left_section = 7.5e7 - 1.1e7 * 2
    right_section = 7.5e7 + 1.1e7 * 2
    gaussian_mask = (freqs > left_section) & (freqs < right_section)

    signal_freqs = freqs[gaussian_mask]
    signal_fft = average[gaussian_mask]
    noise_freqs = freqs[~gaussian_mask]
    noise_fft = average[~gaussian_mask]

    noise_height = np.mean(noise_fft)
    w = right_section - left_section

    noise_power = np.mean(noise_height) * w

    args = (0.19, 7.5e7, 1.1e7)
    popt, cov = do_fit(freqs, average, args)
    #axs[1].plot(freqs, gaussian(freqs, 0.19, 7.5e7, 1.1e7), color = 'red', label = 'Fit')
    signal = quad(gaussian, np.min(signal_freqs), np.max(signal_freqs), args = args)

    signal_power = signal[0] - (noise_power)
    snr = signal_power / noise_power

    amplitude_uncertainty = np.sqrt(cov[0,0])
    x_pos_uncertainty = np.sqrt(cov[1,1])
    std_uncertainty = np.sqrt(cov[2,2])

    snr_1 = (1/(noise_power)**2)*signal[1]**2
    delta_noise_height = np.std(noise_fft) / np.sqrt(len(noise_fft))
    snr_2 = (signal[0]/(noise_height**2 * w))**2 * delta_noise_height**2
    snr_3= (signal[0]/(noise_height * w**2))**2 *std_uncertainty**2
    snr_uncertainty = np.sqrt(snr_1 + snr_2 + snr_3)

    print(f'SNR for 75 MHz: {snr:.2f} ± {snr_uncertainty:.2f}')
    print(f'Noise Level: {NOISE_LEVEL}')
    return snr, snr_uncertainty


def do_fit(freqs, average, initial_guess):
    """
    Fits a Gaussian function to the provided frequency and average data.
    Parameters:
    freqs (array-like): The frequency data points.
    average (array-like): The average values corresponding to the frequency data points.
    initial_guess (tuple): Initial guess for the parameters of the Gaussian function.
    Returns:
    tuple: A tuple containing the optimal parameters (popt) and the covariance of the parameters (cov).
           If the fit fails, returns (None, None).
    """

    amplitude_init = np.max(average)
    b_shift_init = freqs[np.argmax(average)]
    fwhm_est = np.std(freqs)

    try:
        popt, cov = curve_fit(gaussian, freqs, average, p0=initial_guess)
        return popt, cov
    except RuntimeError as e:
        return None, None


def add_gaussian_noise(voltage):
    """
    Adds Gaussian noise to a given voltage signal.

    Parameters:
    voltage (array-like): The input voltage signal to which Gaussian noise will be added.
    Returns:
    array-like: The voltage signal with added Gaussian noise.
    """
    noise = np.random.normal(loc=0, scale=NOISE_LEVEL, size=len(voltage))
    return voltage + noise

def add_sinusoidal_noise(time, voltage, omega):
    """
    Adds sinusoidal noise to a given voltage signal.

    Parameters:
    time (numpy.ndarray): Array of time values.
    voltage (numpy.ndarray): Array of voltage values.
    omega (float): 2 pi f

    Returns:
    numpy.ndarray: Voltage signal with added sinusoidal noise.
    """
    noise = np.sin(omega * time) * NOISE_LEVEL
    noise += np.random.normal(loc=0, scale=NOISE_LEVEL, size=len(voltage))
    return voltage + noise


def gaussian(x, a, b, c):
    """
    Calculate the value of a Gaussian function.

    Parameters:
    x (array-like): The input values where the Gaussian function is evaluated.
    a (float): The amplitude of the Gaussian function.
    b (float): The centre of the Gaussian function.
    c (float): The std of the Gaussian function.

    Returns:
    float or array-like: The value(s) of the Gaussian function at the input x.
    """
    return a * np.exp(-(x - b)**2 /  (2*c**2))


def __main__():
    file_list = list_files_in_folder(folder_path)

    # Import data from all valid files
    data_frame = pd.concat([import_data(file) for file in file_list], ignore_index=True)

    # Sort dataframe by run and frequency
    run_map = calculate_run_map(file_list)
    data_frame['run_order'] = data_frame['run'].map(run_map)
    data = data_frame.sort_values(by=['run_order', 'freq'], ignore_index=True)
    data = data.drop(columns=['run_order'])

    # Plot data
    global NOISE_LEVEL
    NOISE_LEVEL = 0.

    plot_res_freqs(data)


if __name__ == '__main__':
    __main__()