import os
import re
import glob
import xml.etree.ElementTree as ET
from astropy.io import fits
import numpy as np


def readFITS_auto(filepath: str):
    
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


def readFITS_multipoles(filepath: str):
    
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


def readFITS_auto_series_SMU(base_path: str, n_files: int, n_elements: int):

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


def readFITS_auto_series_RpPI(root_folder: str, n_elements: int, kind: str):

    # check for pattern to be either "measured" or "correct"
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
        ns = {"two": "http://ecdm.euclid-ec.org/schema/dpd/le3/gc/twopcfautocart"}
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

                pi_matrix[i] = table_data[names[0]]
                rp_matrix[i] = table_data[names[1]]
                xi_matrix[i] = table_data[names[2]]
            
            else:
                print(f"File {i+1}: {nData} points instead of {n_elements} and {nColumns} columns instead of 3")
    
    return rp_matrix, pi_matrix, xi_matrix