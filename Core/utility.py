import os
import urllib.request
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

def makeDirs(conf: dict) -> None:
    """
    Create directories for the project
    Args:
        conf (dict) - configuration dictionary
    """

    os.makedirs(conf.JP2_DIR, exist_ok=True)
    os.makedirs(conf.PNG_DIR, exist_ok=True)
    os.makedirs(conf.RESULTS_DIR, exist_ok=True)
    os.makedirs(conf.PLOT_DIR, exist_ok=True)

def downloadJP2(conf: dict) -> None:
    """
    Download the JP2 image from the PDS server
    Args:
        conf (dict): Configuration dictionary
    Return:
        None
    """

    if os.path.exists(conf.JP2_FPATH):
        print(f'{conf.JP2_FPATH} already exists')
    else:
        print(f'Downloading :: {conf.imID} \n')

        path = conf.imID.split('_')[0]
        ORB = conf.imID.split('_')[1][:4]

        URL = f'https://hirise-pds.lpl.arizona.edu/PDS/RDR/{path}/ORB_{ORB}00_{ORB}99/{conf.imID}/{conf.imID}_RED.JP2'
        urllib.request.urlretrieve(URL, conf.JP2_FPATH)

def save_results(conf: dict, contours: np.ndarray) -> None:
    """
    Save the contour coordinates to a .npy file & Save the contour image
    Args:
        contours (np.ndarray): Array of contour coordinates
        conf (dict): Configuration dictionary
    """
    
    np.save(conf.RESULTS_FPATH, contours)

    image = plt.imread(conf.PNG_FPATH)
    tck, u = splprep(contours.T, u=None, s=0.0, per=1) 
    u_new = np.linspace(u.min(), u.max(), 1000)
    x_new, y_new = splev(u_new, tck, der=0)

    plt.imshow(image, cmap='gray')
    plt.plot(x_new, y_new, 'b')
    plt.plot(contours[:, 0], contours[:, 1], 'o', color='red', markersize=2)
    plt.savefig(conf.PLOT_FPATH)



    
