import os
import re
import glob
import xml.etree.ElementTree as ET
from astropy.io import fits
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple, Union


def readFITS_auto(filepath: str
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    int
]:
    """
    Read a FITS file containing a correlation function table and
    automatically extract the first three columns as s, μ, and ξ arrays.

    Parameters
    ----------
    filepath : str
        Path to the FITS file to be read.

    Returns
    -------
    tuple
        A tuple `(s_array, mu_array, xi_array, nData)` where:
        - `s_array` : np.ndarray
            Array of separation values s extracted from the first column of the table.
        - `mu_array` : np.ndarray
            Array of μ values extracted from the second column of the table.
        - `xi_array` : np.ndarray
            Array of 2-points correlation function values ξ extracted from the third column.
        - `nData` : int
            Number of rows (data points) in the FITS table.
    """
    
    with fits.open(filepath) as hdul:
        
        table_hdu = hdul[1] # HDU 0 is an empty header that precedes the actual table
        table_data = table_hdu.data # type: ignore
                                    # comment to ignore Pylance warning
        nData = table_data.shape[0]
        # nColumns = len(table_data.columns)

        names = table_data.columns.names
        s_array = table_data[names[0]]
        mu_array = table_data[names[1]]
        xi_array = table_data[names[2]]

    return s_array, mu_array, xi_array, nData


def readFITS_multipoles(filepath: str
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    int
]:
    """
    Read a FITS file containing multipole data and extract the separation
    array along with the multipole values stacked into a matrix.

    Parameters
    ----------
    filepath : str
        Path to the FITS file to be read.

    Returns
    -------
    tuple
        A tuple `(s_array, xi_matrix, nData)` where:
        - `s_array` : np.ndarray
            Array of separation values s extracted from the first column of the table.
        - `xi_matrix` : np.ndarray
            2D array where each column corresponds to a multipole (e.g., ℓ=0,1,2,...) 
            extracted from subsequent columns of the table.
        - `nData` : int
            Number of rows (data points) in the FITS table.
    """
    
    with fits.open(filepath) as hdul:
        
        # print()
        # hdul.info()
        
        table_hdu = hdul[1] # HDU 0 is an empty header that precedes the actual table
        table_data = table_hdu.data # type: ignore
                                    # comment to ignore Pylance warning
        nData = table_data.shape[0]

        names = table_data.columns.names
        s_array = table_data[names[0]]
        xi_matrix = np.column_stack((table_data[names[1]], table_data[names[2]], table_data[names[3]], table_data[names[4]],
                                    table_data[names[5]]))

    return s_array, xi_matrix, nData


def readFITS_auto_series_SMU(base_path: str,
                             n_files: int,
                             n_elements: int
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64]
]:
    """
    Read a series of FITS files containing 2-points correlation function data
    in (μ, s) coordinates and store the extracted s, μ, and ξ values
    in matrices, one row per file.

    Parameters
    ----------
    base_path : str
        Path to the directory containing the FITS files.
    n_files : int
        Number of FITS files to read. Files are expected to be named with
        zero-padded indices following the pattern
        `EUC_LE3_GCL_2PCF_EuclidLargeMocksXXXX_Rot30degCircle_m3_z0p9-1p1_Correlation_AUTO_2DPOL.fits`.
    n_elements : int
        Expected number of rows (data points) in each FITS table.

    Returns
    -------
    tuple
        A tuple `(s_matrix, mu_matrix, xi_matrix)` where:
        - `s_matrix` : np.ndarray
            Array of shape `(n_files, n_elements)` containing the s values
            from each file.
        - `mu_matrix` : np.ndarray
            Array of shape `(n_files, n_elements)` containing the μ values
            from each file.
        - `xi_matrix` : np.ndarray
            Array of shape `(n_files, n_elements)` containing the ξ values
            from each file.

    Notes
    -----
    If a file does not match the expected number of elements or columns,
    a message is printed and that file's row in the matrices remains zeros.
    """

    s_matrix = np.zeros((n_files, n_elements))
    mu_matrix = np.zeros((n_files, n_elements))
    xi_matrix = np.zeros((n_files, n_elements))
    
    for i in range(0,n_files):
        filepath = f"{base_path}/EUC_LE3_GCL_2PCF_EuclidLargeMocks{i+1:04d}_Rot30degCircle_m3_z0p9-1p1_Correlation_AUTO_2DPOL.fits"
                                                                  # we need to have 000i not i

        with fits.open(filepath) as hdul:

            table_hdu = hdul[1]  # HDU 0 is an empty header that precedes the actual table
            table_data = table_hdu.data # type: ignore
                                        # comment to ignore Pylance warning
            nData = table_data.shape[0]
            nColumns = len(table_data.columns)

            if (nData == n_elements) and (nColumns == 3):

                names = table_data.columns.names

                s_matrix[i] = table_data[names[0]]
                mu_matrix[i] = table_data[names[1]]
                xi_matrix[i] = table_data[names[2]]
            
            else:
                print(f"File {i+1}: {nData} points instead of {n_elements} and {nColumns} columns instead of 3")
    
    return s_matrix, mu_matrix, xi_matrix


def readFITS_auto_series_RpPI(root_folder: str,
                              n_elements: int,
                              kind: str
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64]
]:
    """
    Read a series of FITS files containing 2-points correlation function data
    in (r_p, π) coordinates from folders following a specific Euclid
    simulation structure, and store the extracted r_p, π, and ξ values
    in matrices, one row per folder.

    Parameters
    ----------
    root_folder : str
        Path to the root directory containing the subfolders with the FITS files.
    n_elements : int
        Expected number of rows (data points) in each FITS table.
    kind : str
        Type of data to read; must be either "measured" or "correct". This
        determines the folder pattern to search for.

    Returns
    -------
    tuple
        A tuple `(rp_matrix, pi_matrix, xi_matrix)` where:
        - `rp_matrix` : np.ndarray
            Array of shape `(n_files, n_elements)` containing the r_p values
            from each folder.
        - `pi_matrix` : np.ndarray
            Array of shape `(n_files, n_elements)` containing the π values
            from each folder.
        - `xi_matrix` : np.ndarray
            Array of shape `(n_files, n_elements)` containing the ξ values
            from each folder.

    Notes
    -----
    - Folders are searched using the pattern `m_z1_*_{kind}_cart/data` and
      sorted according to the number in `m_z1_X`.
    - Each folder is expected to contain a single `.xml` file which specifies
      the corresponding FITS filename.
    - The function assumes the first column of the FITS table is π, the second
      column is r_p, and the third column is ξ.
    - If a file does not match the expected number of elements or columns,
      a message is printed and that row in the matrices remains zeros.
    """

    # check for kind to be either "measured" or "correct"
    valid_kinds = ["measured", "correct"]
    if kind not in valid_kinds:
        raise ValueError(f"kind must be one of {valid_kinds}, got '{kind}'")
    
    # directories root_folder/m_z1_X_kind_cart/data
    pattern = f"m_z1_*_{kind}_cart/data"
    folders = glob.glob(os.path.join(root_folder, pattern))

    # extract X from m_z1_X_...
    def folder_index(path):
        num = re.search(r"m_z1_(\d+)_measured_cart", path)
        return int(num.group(1)) if num else 99999

    folders = sorted(folders, key=folder_index)
    
    n_files = len(folders)
    print(f"{n_files} folders found")

    rp_matrix = np.zeros((n_files, n_elements))
    pi_matrix = np.zeros((n_files, n_elements))
    xi_matrix = np.zeros((n_files, n_elements))

    for i, folder in enumerate(folders):
        
        # find the .xml file in the folder
        xml_path = glob.glob(os.path.join(folder, "*.xml"))[0]

        # parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # find only the FileName within the CorrelationFile header
        corr_tag = root.find(".//CorrelationFile/DataContainer/FileName")

        if corr_tag is None:
            raise RuntimeError(f"CorrelationFile not found in {xml_path}")
        
        fits_name = corr_tag.text.strip() # type: ignore
        fits_path = os.path.join(folder, fits_name)

        with fits.open(fits_path) as hdul:

            table_hdu = hdul[1]  # HDU 0 is an empty header that precedes the actual table
            table_data = table_hdu.data # type: ignore
                                        # comment to ignore Pylance warning
            nData = table_data.shape[0]
            nColumns = len(table_data.columns)

            if (nData == n_elements) and (nColumns == 3):

                names = table_data.columns.names

                pi_matrix[i] = table_data[names[0]] # apparently the first column in the file is pi
                rp_matrix[i] = table_data[names[1]] # and the second is r_p
                xi_matrix[i] = table_data[names[2]]
            
            else:
                print(f"File {i+1}: {nData} points instead of {n_elements} and {nColumns} columns instead of 3")
    
    return rp_matrix, pi_matrix, xi_matrix


def readFITS_pairs_series_SMU(base_path: str,
                              n_files: int,
                              n_elements: int
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64]
]:
    """
    Read a series of FITS files containing 2-points correlation estimators
    (pair counts) in (μ, s) coordinates and store the extracted s, μ, and 
    DD, DR and RR values in matrices, one row per file.

    Parameters
    ----------
    base_path : str
        Path to the directory containing the FITS files.
    n_files : int
        Number of FITS files to read. Files are expected to be named with
        zero-padded indices following the pattern
        `EUC_LE3_GCL_2PCF_EuclidLargeMocksXXXX_Rot30degCircle_m3_z0p9-1p1_PAIRS_AUTO_2DPOL.fits`.
    n_elements : int
        Expected number of rows (data points) in each FITS table.

    Returns
    -------
    tuple
        A tuple `(s_matrix, mu_matrix, dd_matrix, dr_matrix, rr_matrix)` where:
        - `s_matrix` : np.ndarray
            Array of shape `(n_files, n_elements)` containing the s values
            from each file.
        - `mu_matrix` : np.ndarray
            Array of shape `(n_files, n_elements)` containing the μ values
            from each file.
        - `dd_matrix` : np.ndarray
            Array of shape `(n_files, n_elements)` containing the data-data
            pair counts from each file.
        - `dr_matrix` : np.ndarray
            Array of shape `(n_files, n_elements)` containing the data-random
            pair counts from each file.
        - `rr_matrix` : np.ndarray
            Array of shape `(n_files, n_elements)` containing the random-random
            pair counts from each file.

    Notes
    -----
    If a file does not match the expected number of elements or columns,
    a message is printed and that file's row in the matrices remains zeros.
    """

    s_matrix = np.zeros((n_files, n_elements))
    mu_matrix = np.zeros((n_files, n_elements))
    dd_matrix = np.zeros((n_files, n_elements))
    dr_matrix = np.zeros((n_files, n_elements))
    rr_matrix = np.zeros((n_files, n_elements))
    
    for i in range(0,n_files):
        filepath = f"{base_path}/EUC_LE3_GCL_2PCF_EuclidLargeMocks{i+1:04d}_Rot30degCircle_m3_z0p9-1p1_PAIRS_AUTO_2DPOL.fits"
                                                                  # we need to have 000i not i

        with fits.open(filepath) as hdul:

            table_hdu = hdul[1]  # HDU 0 is an empty header that precedes the actual table
            table_data = table_hdu.data # type: ignore
                                        # comment to ignore Pylance warning
            
            nData = table_data.shape[0]
            nColumns = len(table_data.columns)

            if (nData == n_elements) and (nColumns == 5):

                names = table_data.columns.names

                s_matrix[i] = table_data[names[0]]
                mu_matrix[i] = table_data[names[1]]
                dd_matrix[i] = table_data[names[2]]
                dr_matrix[i] = table_data[names[3]]
                rr_matrix[i] = table_data[names[4]]
            
            else:
                print(f"File {i+1}: {nData} points instead of {n_elements} and {nColumns} columns instead of 5")
    
    return s_matrix, mu_matrix, dd_matrix, dr_matrix, rr_matrix


def readFITS_pairs_series_RpPI(root_folder: str,
                              n_elements: int,
                              kind: str
) -> Tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64]
]:
    """
    Read a series of FITS files containing 2-points correlation function estimators
    (pair counts) in (r_p, π) coordinates from folders following a specific Euclid
    simulation structure, and store the extracted r_p, π, and DD, DR and RR values
    in matrices, one row per folder.

    Parameters
    ----------
    root_folder : str
        Path to the root directory containing the subfolders with the FITS files.
    n_elements : int
        Expected number of rows (data points) in each FITS table.
    kind : str
        Type of data to read; must be either "measured" or "correct". This
        determines the folder pattern to search for.

    Returns
    -------
    tuple
        A tuple `(rp_matrix, pi_matrix, dd_matrix, dr_matrix, rr_matrix)` where:
        - `rp_matrix` : np.ndarray
            Array of shape `(n_files, n_elements)` containing the r_p values
            from each folder.
        - `pi_matrix` : np.ndarray
            Array of shape `(n_files, n_elements)` containing the π values
            from each folder.
        - `dd_matrix` : np.ndarray
            Array of shape `(n_files, n_elements)` containing the data-data
            pair counts from each file.
        - `dr_matrix` : np.ndarray
            Array of shape `(n_files, n_elements)` containing the data-random
            pair counts from each file.
        - `rr_matrix` : np.ndarray
            Array of shape `(n_files, n_elements)` containing the random-random
            pair counts from each file.

    Notes
    -----
    - Folders are searched using the pattern `m_z1_*_{kind}_cart/data` and
      sorted according to the number in `m_z1_X`.
    - Each folder is expected to contain a single `.xml` file which specifies
      the corresponding FITS filename.
    - The function assumes the first column of the FITS table is π, the second
      column is r_p, and the third column is ξ.
    - If a file does not match the expected number of elements or columns,
      a message is printed and that row in the matrices remains zeros.
    """

    # check for kind to be either "measured" or "correct"
    valid_kinds = ["measured", "correct"]
    if kind not in valid_kinds:
        raise ValueError(f"kind must be one of {valid_kinds}, got '{kind}'")
    
    # directories root_folder/m_z1_X_kind_cart/data
    pattern = f"m_z1_*_{kind}_cart/data"
    folders = glob.glob(os.path.join(root_folder, pattern))

    # extract X from m_z1_X_...
    def folder_index(path):
        num = re.search(r"m_z1_(\d+)_measured_cart", path)
        return int(num.group(1)) if num else 99999

    folders = sorted(folders, key=folder_index)
    
    n_files = len(folders)
    print(f"{n_files} folders found")

    rp_matrix = np.zeros((n_files, n_elements))
    pi_matrix = np.zeros((n_files, n_elements))
    dd_matrix = np.zeros((n_files, n_elements))
    dr_matrix = np.zeros((n_files, n_elements))
    rr_matrix = np.zeros((n_files, n_elements))

    for i, folder in enumerate(folders):
        
        # find the .xml file in the folder
        xml_path = glob.glob(os.path.join(folder, "*.xml"))[0]

        # parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # find only the FileName within the PairsFile header
        corr_tag = root.find(".//PairsFile/DataContainer/FileName")

        if corr_tag is None:
            raise RuntimeError(f"PairsFile not found in {xml_path}")
        
        fits_name = corr_tag.text.strip() # type: ignore
        fits_path = os.path.join(folder, fits_name)

        with fits.open(fits_path) as hdul:

            table_hdu = hdul[1]  # HDU 0 is an empty header that precedes the actual table
            table_data = table_hdu.data # type: ignore
                                        # comment to ignore Pylance warning
            nData = table_data.shape[0]
            nColumns = len(table_data.columns)

            if (nData == n_elements) and (nColumns == 5):

                names = table_data.columns.names

                pi_matrix[i] = table_data[names[0]] # apparently the first column in the file is pi
                rp_matrix[i] = table_data[names[1]] # and the second is r_p
                dd_matrix[i] = table_data[names[2]]
                dr_matrix[i] = table_data[names[3]]
                rr_matrix[i] = table_data[names[4]]
            
            else:
                print(f"File {i+1}: {nData} points instead of {n_elements} and {nColumns} columns instead of 5")
    
    return rp_matrix, pi_matrix, dd_matrix, dr_matrix, rr_matrix