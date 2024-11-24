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

    try:
        return [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    except Exception as e:
        print(f"An error occurred during file import: {e}")
        sys.exit()


def import_data(file):

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

    run_numbers = set()
    for file in files_list:
        run_number, _ = parse_filename(file)
        run_numbers.add(str(run_number))
    run_numbers = sorted(run_numbers, key=lambda x: (0 if x == 0 else 1, x))
    run_map = {run: idx for idx, run in enumerate(run_numbers)}
    return run_map


def parse_filename(file):

    res_freq = ''.join(filter(str.isdigit, file))
    if file[0].isalpha():
        run_number = file[0]
    else:
        run_number = 0
    return str(run_number), int(res_freq)

def add_legend_background(ax, legend, color="white"):
    # Matplotlib has a bug where the legend background is not drawn when using seaborn-v0_8 style. 
    # This function adds a white background to the legend. If the bug is fixed, this function can be removed.
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

    background = calculate_background(df)
    print(background)
    flag = pd.DataFrame(columns = ['res_freq', 'x_pos', 'y_pos', 'p_value', 'snr'])

    for i in df["freq"].unique():
        df_f = df[df["freq"] == i]

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns
        plt.style.use("seaborn-v0_8")


        length = int(np.shape(df_f[df_f['run']=='A'])[0]/2 - 1)
        average_df = np.zeros((len(df_f["run"].unique()), length))
        #average_df = np.zeros((1, length))
        num = 0
        for idx, j in enumerate(df_f["run"].unique()):
            #vt
            if True:
                df_r = df_f[df_f["run"] == j]
                #df_r.loc[:, "voltage"] = add_noise(df_r["voltage"])
                df_r.loc[:, "voltage"] = add_sinusoidal_noise(df_r["time"], df_r["voltage"], 2*np.pi*1.5e9)
                
                axs[0].scatter(df_r["time"], df_r["voltage"], label=f'{j}', s = 4)

                sample_spacing = 1.562000e-10
                N = N = len(df_r['voltage'])
                
                y = np.array(df_r['voltage'])
                yf = fft(y)
                xf = fftfreq(N, sample_spacing)#[:N//2]

                freqs = xf

                yf = yf[xf > 0]
                freqs = xf[xf > 0]

                average_df[idx] = np.abs(yf)
                
                #average_df[idx] -= background


            elif num < 1:
                # Plot a single run
                df_r_0 = df_f[df_f["run"] == '0']
                df_r_0.loc[:, "voltage"] = add_noise(df_r_0["voltage"])
                
                axs[0].scatter(df_r_0["time"], df_r_0["voltage"], label=f'{j}', s = 3)


                sample_spacing = df_r_0['time'][1] - df_r_0['time'][0]
                N = N = len(df_r_0['voltage'])

                y = np.array(df_r_0['voltage'])
                yf = fft(y)
                xf = fftfreq(N, sample_spacing)[:N//2]

                average_df[idx] = np.abs(yf)
                
                axs[0].set_xlim(0, df_r_0["time"].max())
                num += 1



        axs[0].grid(True)
        axs[0].legend(loc='upper right', title='Run', facecolor = 'white')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Voltage (V)')
        legend = axs[0].legend(loc = 'upper right')
        add_legend_background(axs[0], legend, color="white")
        
        average = np.mean(average_df, axis = 0)
        average = np.abs(average)
        std = np.std(average_df, axis = 0)

        axs[1].errorbar(freqs, average[:N//2], yerr = std[:N//2], label = 'Average', linestyle = '--', markersize = 2, fmt = 'o', capsize = 2, color = 'black', ecolor = 'darkviolet')
        #axs[1].plot(freqs, gaussian(freqs, 0.19 , 7.5e7, 1.1e7) , color = 'green', label = 'Approximated fit with known parameters')
        axs[1].grid(True)
        #axs[1].set_xlim(0, 1e9)
        axs[1].set_ylim(0, 0.22)
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('Amplitude')
        #axs[1].plot(freqs, background, color = 'yellow', label = 'Average Background')

       # _, _ = SNR(freqs, average, axs)


        fig.suptitle(f'Resonance Frequency: {i} MHz', fontsize=16, fontweight='bold')
        add_legend_background(axs[1], legend, color="white")


        # find peaks
        n_sections = 10  # Number of sections, adjust as needed
        freq_min = np.min(freqs)
        freq_max = np.max(freqs)
        section_edges = np.linspace(freq_min, freq_max, n_sections + 1)

        flagged_local_frequencies = np.array([])
        flagged_local_amps = np.array([])
        flagged_p_values = np.array([])
        flagged_snr_values = np.array([])
        for k in range(n_sections):
            idx_section = (freqs >= section_edges[k]) & (freqs < section_edges[k+1])
            freqs_section = freqs[idx_section]
            average_section = average[idx_section]

            if len(average_section) == 0:
                continue  

            noise_floor = np.median(average_section)
            noise_std = np.std(average_section)

            threshold_factor = 3 #4  
            height_threshold = noise_floor + threshold_factor * noise_std
            prominence_threshold = noise_std * threshold_factor #0.025 # Can be adjusted

            peaks, _ = find_peaks(average_section, height = height_threshold, prominence=prominence_threshold)

            peaks_pos = freqs_section[peaks]
            peaks_height = average_section[peaks]


            noise_indices = np.ones(len(average_section), dtype=bool)
            noise_indices[peaks] = False
            noise_data = average_section[noise_indices]

            sigma_hat = np.sqrt(np.sum(noise_data**2) / (2 * len(noise_data)))

            p_values = rayleigh.sf(peaks_height, scale=sigma_hat)


            flagged_local_frequencies = np.append(flagged_local_frequencies, peaks_pos)
            flagged_local_amps = np.append(flagged_local_amps, peaks_height)
            flagged_p_values = np.append(flagged_p_values, p_values)


            #SNR
            if len(peaks) > 0:
                w = section_edges[k+1] - section_edges[k]
                noise_power = np.sum(noise_floor**2)

                signal_power = np.sum(peaks_height**2)
                snr = signal_power / noise_power
                flagged_snr_values = np.append(flagged_snr_values, snr)

    

        #print(flagged_local_frequencies)
        #print(flagged_local_amps)
        #print(flagged_p_values)


        axs[1].scatter(section_edges, np.ones((len(section_edges)))*0.025, color = 'blue', label = 'Section Edges', s = 2)
        axs[1].scatter(flagged_local_frequencies, flagged_local_amps, color = 'grey', label = 'Flagged Peaks')

        if len(flagged_p_values) > 0:
            alpha = 3.726653e-6 #0.05

            # Perform Benjamini-Hochberg FDR correction
            rejected, pvals_corrected, _, _ = multipletests(flagged_p_values, alpha=alpha, method='fdr_bh')

            significant_freqs = flagged_local_frequencies[rejected]
            significant_amps = flagged_local_amps[rejected]
            significant_p_values = flagged_p_values[rejected]

            axs[1].scatter(significant_freqs,significant_amps,color='red',label='Significant Peaks')
            for freq, amp, p, s in zip(significant_freqs, significant_amps, significant_p_values, flagged_snr_values):
                #print(f'Frequency: {freq:.2f} Hz, Amplitude: {amp:.2f}')
                
                new_row = pd.DataFrame([{'res_freq': i, 'x_pos': freq, 'y_pos': amp, 'p_value': p, 'snr': s}])
                flag = pd.concat([flag, new_row], ignore_index=True)

        



        axs[1].legend(loc = 'upper right')
        plt.show()
    print(flag)



def calculate_background(df):

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
                df_r.loc[:, "voltage"] = add_sinusoidal_noise(df_r["time"], df_r["voltage"], 2*np.pi*1.5e9)
                sample_spacing = 1.562000e-10
                N =  len(df_r['voltage'])
                    
                y = np.array(df_r['voltage'])
                yf = fft(y)
                xf = fftfreq(N, sample_spacing)#[:N//2]

                freqs = xf

                yf = yf[xf > 0]
                freqs = xf[xf > 0]

                average_df[idx] = np.abs(yf)
                #average_df[idx] -= background

            average = np.mean(average_df, axis = 0)
            average = np.abs(average)

            background.append(average)
    
    background = np.array(background)
    background = np.mean(background, axis = 0)
    return background






def SNR(freqs, average, axs):
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



        #axs[1].hlines(np.mean(background), 0, np.max(freqs), color = 'red', label = 'Average Background')
        #axs[1].hlines(noise_height, 0, np.max(freqs), color = 'blue', label = f'Average Noise\n{NOISE_LEVEL}') 
        #axs[1].scatter(signal_freqs, signal_fft, color = 'green', label = 'Signal')
        noise_power = np.mean(noise_height) * w

        popt, cov = do_fit(freqs, average)
        #axs[1].plot(freqs, gaussian(freqs, 0.19, 7.5e7, 1.1e7), color = 'red', label = 'Fit')
        signal = quad(gaussian, np.min(signal_freqs), np.max(signal_freqs), args = (0.19, 7.5e7, 1.1e7))

    
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


def add_noise(voltage):
    noise = np.random.normal(loc=0, scale=NOISE_LEVEL, size=len(voltage))
    return voltage + noise

def add_sinusoidal_noise(time, voltage, omega):
    noise = np.sin(omega * time) * NOISE_LEVEL
    noise += np.random.normal(loc=0, scale=NOISE_LEVEL, size=len(voltage))
    return voltage + noise

def gaussian(x, a, b, c):
    return a * np.exp(-(x - b)**2 /  (2*c**2))

def do_fit(freqs, average):
    amplitude_init = np.max(average)
    b_shift_init = freqs[np.argmax(average)]
    fwhm_est = np.std(freqs)
    #initial_guess = [amplitude_init, b_shift_init, fwhm_est]
    initial_guess = [0.19, 7.5e7, 1.1e7]
    
    bounds = ([0.125, 6e7, 1e7], [0.2, 8e7, 1.2e7]) 
    try:
        popt, cov = curve_fit(gaussian, freqs, average, p0=initial_guess, bounds = bounds)
        return popt, cov
    except RuntimeError as e:
        return None, None


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
    NOISE_LEVEL = 0.0001#0.006

    plot_res_freqs(data)


    
if __name__ == '__main__':
    __main__()