import os
import numpy as np
from typing import List, Tuple, Optional

import pandas as pd
#import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.cosmology import Planck15 as cosmo  # Using Planck15 cosmology by default

class spectra_redshift_type:
    def __init__(self, data_dir = "../data/ZTFBTS", 
                 spectra_dir = "../data/ZTFBTS_spectra",
                 drop_non_SN = True, 
                 drop_SN_in_type_name = True, 
                 filter = np.array(['II-pec', 'II', 'IIP', 'IIb', 'IIn', 'Iax', 'Ia-91T',
                                    'Ia-91bg', 'Ia-CSM', 'Ia', 'Ia-pec', 'Ib', 'Ib-pec',
                                    'Ibn', 'Ic-BL', 'Ic', 'Ic-pec'])
                 ,quiet = False):
        if not quiet:
            print("Loading master sheet...")
        df = pd.read_csv(f"{data_dir}/ZTFBTS_TransientTable.csv")
        df["redshift"] = pd.to_numeric(df["redshift"], errors="coerce")
        df = df.dropna(subset=["redshift", "type"])
        if drop_non_SN:
            df = df[df['type'].str.contains("SN ")]
        if drop_SN_in_type_name:
            df['type'] = df['type'].str.replace("SN ", "")
        
        self.mastersheet = df
        self.data_dir = data_dir
        self.spectra_dir = spectra_dir
        self.n = df.shape[0]
        self.n_valid = None

        self.file_names_list = []
        self.spectra_list = []
        self.type_list = []
        self.redshift_list = []
        self.shuffle = None
        self.failed_to_read = []
    
    def readin_spectra(self): # actually load things
        print("Loading spectra data...")
        self.n_valid = 0
        for i in tqdm(range(self.n)): # for each file
            try:
                file_name = np.array(self.mastersheet['ZTFID'])[i]
                freq_ary, spec_ary, _, _, _ = load_spectras(self.spectra_dir, filenames = [file_name])
                argsoted_freq = np.argsort(freq_ary[0])
                self.spectra_list.append(np.array([freq_ary[0][argsoted_freq], 
                                               spec_ary[0][argsoted_freq]]))
                
                self.file_names_list.append(file_name)
                self.type_list.append(np.array(self.mastersheet['type'])[i])
                self.redshift_list.append(np.array(self.mastersheet['redshift'])[i])
                self.n_valid += 1
            except:
                self.failed_to_read.append(np.array(self.mastersheet['ZTFID'])[i])
                #print(np.array(self.mastersheet['ZTFID'])[i], " having issue")
            

    def reshuffle(self):
        idx = np.array([i for i in range(self.n_valid)])
        np.random.shuffle(idx)
        self.shuffle = idx

    def get_split(self, proportion_cal = 0.5):
        assert proportion_cal < 1.
        if self.shuffle is None:
            self.reshuffle()
        n_cal = int(proportion_cal * self.n_valid)

        cal_set = spectra_redshift_type(data_dir=self.data_dir, spectra_dir=self.spectra_dir, quiet = True)
        test_set = spectra_redshift_type(data_dir=self.data_dir, spectra_dir=self.spectra_dir, quiet = True)

        cal_set.n = n_cal
        test_set.n = self.n_valid-n_cal

        cal_set.file_names_list = [self.file_names_list[i] for i in self.shuffle[:n_cal]]
        cal_set.spectra_list = [self.spectra_list[i] for i in self.shuffle[:n_cal]]
        cal_set.type_list = [self.type_list[i] for i in self.shuffle[:n_cal]]
        cal_set.redshift_list = [self.redshift_list[i] for i in self.shuffle[:n_cal]]

        test_set.file_names_list = [self.file_names_list[i] for i in self.shuffle[n_cal:]]
        test_set.spectra_list = [self.spectra_list[i] for i in self.shuffle[n_cal:]]
        test_set.type_list = [self.type_list[i] for i in self.shuffle[n_cal:]]
        test_set.redshift_list = [self.redshift_list[i] for i in self.shuffle[n_cal:]]

        return cal_set, test_set




### utils for data, ported from https://github.com/ThomasHelfer/multimodal-supernovae/tree/main/src

def filter_files(filenames_avail, filenames_to_filter, data_to_filter=None):
    """
    Function to filter filenames and data based on the filenames_avail

    Args:
    filenames_avail (list): List of filenames available
    filenames_to_filter (list): List of filenames to filter
    data_to_filter (List[np.ndarray]): Data to filter based on filenames_to_filter

    Returns:
    inds_filt (np.ndarray): Indices of filtered filenames in filenames_to_filter
    filenames_to_filter (list): List of filtered filenames
    data_to_filter (np.ndarray): Filtered data
    """
    # Check which each filenames_to_filter are available in filenames_avail
    inds_filt = np.isin(filenames_to_filter, filenames_avail)
    if data_to_filter:
        for i in range(len(data_to_filter)):
            data_to_filter[i] = data_to_filter[i][inds_filt]

    filenames_to_filter = np.array(filenames_to_filter)[inds_filt]

    return inds_filt, filenames_to_filter, data_to_filter

def load_redshifts(data_dir: str, filenames: List[str] = None) -> np.ndarray:
    """
    Load redshift values from a CSV file in the specified directory.

    Args:
    data_dir (str): Directory path containing the redshift CSV file.
    filenames (List[str]): List of filenames corresponding to the loaded data; default is None.

    Returns:
    np.ndarray: Array of redshift values.
    filenames (List[str]): List of filenames corresponding to the returned data.
    """
    print("Loading redshifts...")

    # Load values from the CSV file
    df = pd.read_csv(f"{data_dir}/ZTFBTS_TransientTable.csv")
    df["redshift"] = pd.to_numeric(df["redshift"], errors="coerce")
    df = df.dropna(subset=["redshift"])

    if filenames is None:
        redshifts = df["redshift"].values
        filenames_redshift = df["ZTFID"].values
    else:
        # Filter redshifts based on the filenames
        redshifts = df[df["ZTFID"].isin(filenames)]["redshift"].values

        filenames_redshift = df[df["ZTFID"].isin(filenames)]["ZTFID"].values

    print("Finished loading redshift")
    return redshifts, filenames_redshift


def load_lightcurves(
    data_dir: str,
    abs_mag: bool = False,
    n_max_obs: int = 100,
    filenames: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load light curves from CSV files in the specified directory; load files that are available if
    filenames are provided.

    Args:
    data_dir (str): Directory path containing light curve CSV files.
    abs_mag (bool): If True, convert apparent magnitude to absolute magnitude.
    n_max_obs (int): Maximum number of data points per lightcurve.
    filenames (List[str], optional): List of filenames to load. If None, all files are loaded.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]: A tuple containing:
        - time_ary: Numpy array of time observations.
        - mag_ary: Numpy array of magnitude observations.
        - magerr_ary: Numpy array of magnitude error observations.
        - mask_ary: Numpy array indicating the presence of an observation.
        - nband: Number of observation bands.
        - filenames_loaded: List of filenames corresponding to the loaded data.
    """

    print("Loading light curves...")
    dir_light_curves = f"{data_dir}/light-curves/"

    def open_light_curve_csv(filename: str) -> pd.DataFrame:
        """Helper function to open a light curve CSV file."""
        file_path = os.path.join(dir_light_curves, filename)
        return pd.read_csv(file_path)

    bands = ["R", "g"]
    nband = len(bands)
    if filenames is None:
        filenames = sorted(os.listdir(dir_light_curves))  # Sort file names
    else:  # If filenames are provided, filter the filenames
        _, filenames, _ = filter_files(
            sorted(os.listdir(dir_light_curves)), [f + ".csv" for f in filenames]
        )

    mask_list, mag_list, magerr_list, time_list, filenames_loaded = [], [], [], [], []

    for filename in tqdm(filenames):
        if filename.endswith(".csv"):
            light_curve_df = open_light_curve_csv(filename)

            if not all(
                col in light_curve_df.columns
                for col in ["time", "mag", "magerr", "band"]
            ):
                continue

            time_concat, mag_concat, magerr_concat, mask_concat = [], [], [], []
            for band in bands:
                df_band = light_curve_df[light_curve_df["band"] == band]

                if len(df_band["mag"]) > n_max_obs:
                    # Sample n_max_obs observations randomly (note order doesn't matter and the replace flag guarantees no double datapoints)
                    indices = np.random.choice(
                        len(df_band["mag"]), n_max_obs, replace=False
                    )
                    mask = np.ones(n_max_obs, dtype=bool)
                else:
                    # Pad the arrays with zeros and create a mask
                    indices = np.arange(len(df_band["mag"]))
                    mask = np.zeros(n_max_obs, dtype=bool)
                    mask[: len(indices)] = True

                time = np.pad(
                    df_band["time"].iloc[indices],
                    (0, n_max_obs - len(indices)),
                    "constant",
                )
                mag = np.pad(
                    df_band["mag"].iloc[indices],
                    (0, n_max_obs - len(indices)),
                    "constant",
                )
                magerr = np.pad(
                    df_band["magerr"].iloc[indices],
                    (0, n_max_obs - len(indices)),
                    "constant",
                )

                # Normalise time if there is anything to normalise
                if sum(mask) != 0:
                    time[mask] = time[mask] - np.min(time[mask])

                time_concat += list(time)
                mag_concat += list(mag)
                magerr_concat += list(magerr)
                mask_concat += list(mask)

            mask_list.append(mask_concat)
            time_list.append(time_concat)
            mag_list.append(mag_concat)
            magerr_list.append(magerr_concat)
            filenames_loaded.append(filename.replace(".csv", ""))

    time_ary = np.array(time_list)
    mag_ary = np.array(mag_list)
    magerr_ary = np.array(magerr_list)
    mask_ary = np.array(mask_list)

    if abs_mag:
        print("Converting to absolute magnitude...", flush=True)

        zs = load_redshifts(data_dir, filenames)
        inds = ~np.isnan(zs)

        # Convert from apparent magnitude to absolute magnitude
        mag_ary -= cosmo.distmod(zs).value[:, None]

        time_ary = time_ary[inds]
        mag_ary = mag_ary[inds]
        magerr_ary = magerr_ary[inds]
        mask_ary = mask_ary[inds]

        filenames = np.array(filenames)[inds]

    return time_ary, mag_ary, magerr_ary, mask_ary, nband, filenames_loaded


def load_spectras(
    data_dir: str,
    n_max_obs: int = 5000,
    zero_pad_missing_error: bool = True,
    rescalefactor: int = 1e14,
    filenames: List[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load spectra data from CSV files in the specified directory; load files that are available if
    filneames are provided.

    Args:
        data_dir (str): Path to the directory containing the CSV files.
        n_max_obs (int) default 5000: maximum length of data, shorter data is padded and masked and longer data is shorted by randomly choosing points
        zero_pad_missing_error (bool) default True: if there is missing error in a file, pad the error with zero, otherwise it will be removed
        rescalefactor (int) default 1e14: factor to rescale the spectrum data
        filenames (List[str], optional): List of filenames to load. If None, all files are loaded.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]: A tuple containing
        arrays of time, magnitude, magnitude error, mask, and filenames.
        - freq (np.ndarray): Array of frequency values for each observation.
        - spec (np.ndarray): Array of spectrum values for each observation.
        - specerr (np.ndarray): Array of spectrum error values for each observation.
        - mask (np.ndarray): Array indicating which observations are not padding.
        - filenames_loaded (List[str]): List of filenames corresponding to the loaded data.
    """

    #print("Loading spectra ...")
    dir_data = f"{data_dir}"

    def open_spectra_csv(filename: str) -> pd.DataFrame:
        """Helper function to open a light curve CSV file."""
        file_path = os.path.join(dir_data, filename)
        return pd.read_csv(file_path, header=None)

    if filenames is None:
        # Getting filenames
        filenames = sorted(os.listdir(dir_data))
    else:
        _, filenames, _ = filter_files(
            sorted(os.listdir(dir_data)), [f + ".csv" for f in filenames]
        )

    mask_list, spec_list, specerr_list, freq_list, filenames_loaded = [], [], [], [], []

    for filename in filenames:
        if filename.endswith(".csv"):
            spectra_df = open_spectra_csv(filename)
            max_columns = spectra_df.shape[1]

            # Checking size and naming dependent on that
            # Note: not all spectra have errors
            if max_columns == 2:
                spectra_df.columns = ["freq", "spec"]
            elif max_columns == 3:
                spectra_df.columns = ["freq", "spec", "specerr"]
                # Fill missing data with zeros
                if zero_pad_missing_error:
                    spectra_df["specerr"] = spectra_df["specerr"].fillna(0)
                # If no zero-pad remove whole colums with missing data
                else:
                    spectra_df.dropna(subset=["specerr"], inplace=True)
            else:
                ValueError("spectra csv should have 2 or three columns only")

            # Checking if the file is too long
            if len(spectra_df["spec"]) > n_max_obs:
                # Sample n_max_obs observations randomly (note order doesn't matter and the replace flag guarantees no double datapoints)
                indices = np.random.choice(
                    len(spectra_df["spec"]), n_max_obs, replace=False
                )
                mask = np.ones(n_max_obs, dtype=bool)
            else:
                # Pad the arrays with zeros and create a mask
                indices = np.arange(len(spectra_df["spec"]))
                mask = np.zeros(n_max_obs, dtype=bool)
                mask[: len(indices)] = True

            # Pad time and mag
            freq = np.pad(
                spectra_df["freq"].iloc[indices],
                (0, n_max_obs - len(indices)),
                "constant",
            )
            spec = rescalefactor * np.pad(
                spectra_df["spec"].iloc[indices],
                (0, n_max_obs - len(indices)),
                "constant",
            )

            # If there is no error, then just give an empty array with zeros
            if max_columns == 3:
                specerr = rescalefactor * np.pad(
                    spectra_df["specerr"].iloc[indices],
                    (0, n_max_obs - len(indices)),
                    "constant",
                )
            else:
                specerr = np.zeros_like(spec)

            mask_list.append(mask)
            freq_list.append(freq)
            spec_list.append(spec)
            specerr_list.append(specerr)
            filenames_loaded.append(filename.replace(".csv", ""))

    freq_ary = np.array(freq_list)
    spec_ary = np.array(spec_list)
    specerr_ary = np.array(specerr_list)
    mask_ary = np.array(mask_list)

    return freq_ary, spec_ary, specerr_ary, mask_ary, filenames_loaded
