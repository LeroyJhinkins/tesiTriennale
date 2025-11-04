import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.widgets import Slider, Button, CheckButtons
from astropy.io import fits
plt.rcParams.update({'font.size': 14})
plt.rcParams['text.usetex'] = True  # use real LaTeX
plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'  # use siunitx

try:
    matplotlib.use("QtAgg")
except ImportError:
    print("QtAgg backend not available, using default backend.")
matplotlib.rcParams['toolbar'] = 'None' # disabling window bars


# read FITS file ------------------------------------------------------------------------------------------------------------------
filepath = "data/catalogues/data_PDFz_EuclidLargeBox0001_Rot30degCircle_m3_obsz_z0.9-1.1.fits" # measured
# filepath = "data/catalogues/data_target_PDFz_EuclidLargeBox0001_Rot30degCircle_m3_obsz_z0.9-1.1.fits" # correct

with fits.open(filepath) as hdul:
    
    print()
    hdul.info()
    
    table_hdu = hdul[1]  # HDU 0 is an empty header that precedes the actual table
    table_data = table_hdu.data # type: ignore
                                # comment to ignore Pylance warning
    nData = table_data.shape[0]
    nColumns = len(table_data.columns)
    
    names = table_data.columns.names
    print(names)
    id_array = table_data[names[0]]
    ra_array = table_data[names[1]]
    dec_array = table_data[names[2]]
    z_array = table_data[names[3]]
    weight_array = table_data[names[4]]
    
    print("\nTotal number of points:", nData)
    print(f"Total Right Ascension range: [{np.min(ra_array)}; {np.max(ra_array)}]")
    print(f"Total Declination range: [{np.min(dec_array)}; {np.max(dec_array)}]")
    print(f"Total Redshift range: [{np.min(z_array)}; {np.max(z_array)}]")

# print("\nFirst five points:")
# for id, ra, dec, z, w in zip(id_array[:5], ra_array[:5], dec_array[:5], z_array[:5], weight_array[:5]):
#     print(id, ra, dec, z, w)

# using LSSGALPY_wedges program ---------------------------------------------------------------------------------------------------
# https://github.com/margudo/LSSGALPY

ra_tot, dec_tot, z_tot = [ra_array], [dec_array], [z_array]
rad_tot = [np.radians(raval) for raval in ra_tot] # R.A. from degrees to radian for the polar representation

# Condition for the declination range in the plots
cond_dec = lambda decCeni, decDeltai: [((decval > decCeni) & (decval < decCeni + decDeltai)) for decval in dec_tot]

# Default values for reset button
decCen0, decDelta0, alpha0 = 50., 2.5, .1

# Main plot: Wedge diagram
fig = plt.figure("Wedge diagram", figsize=(16.3, 8.4))
ax1, ax2 = [plt.axes(posval, polar=val, projection=pj) for posval, val, pj in zip([[0.05, 0.2, 0.9, 0.7], [0.74, 0.1, 0.25, 0.25]], [True, False], [None, 'mollweide'])]
[ax.grid(True) for ax in [ax1, ax2]]

# Plotting galaxy positions in the selected declination range
xyplt = [ax1.plot(rad_tot[0][cond_dec(decCen0, decDelta0)[0]], z_tot[0][cond_dec(decCen0, decDelta0)[0]], 'k.', ms=1, alpha=alpha0, visible=True)[0]]

# Additional plot: Mollweide projection
def mollfig(decCeni, decDeltai):
    x, y = [np.radians(val) for val in [-1*(ra_array - 180), dec_array]]
    H, xedges, yedges = np.histogram2d(x.T, y.T, bins=50)
    extent, levels = [xedges[0], xedges[-1], yedges[0], yedges[-1]], [99,1.0e2]
    ax2.contourf(H.T, levels, origin='lower', colors='b', extent=extent, alpha=.3)
    ax2.grid(True)
    raCen, raDelta = 0., 180.
    x_rect, y_rect = [np.radians(val) for val in [raCen+np.array([-1, -1, 1, 1, -1])*raDelta, decCeni+np.array([0, 1, 1, 0, 0])*decDeltai]]
    ax2.fill(x_rect, y_rect, 'r', lw=0, alpha=.5)
    [plt.setp(getval, fontsize=fontval, alpha=.6) for getval, fontval in zip([ax2.get_xticklabels(), ax2.get_yticklabels()], [8, 12])]
mollfig(decCen0, decDelta0)

# Location of sliders in the figure
axdec, axrang, axalpha = [plt.axes(val, facecolor='lightgoldenrodyellow') for val in [[0.25, 0.06, 0.5, 0.03], [0.25, 0.02, 0.5, 0.03], [0.05, 0.1, 0.07, 0.03]]]

# Definition of the sliders
sdec, srang, salpha = [Slider(axval, name, valmin, valmax, valinit=val0, valfmt=vfmt) for axval, name, valmin, valmax, val0, vfmt in zip([axdec, axrang, axalpha], ['Dec.', 'Range', 'Transp.'], [-20.0, 0.0, 0.0], [90.0, 90.0, 1.], [decCen0, decDelta0, alpha0], ['%1.1f', '%1.1f', '%1.2f'])]

# Text in the figure
fig_text = plt.figtext(0.5, 0.11, 'Wedge diagram within %1.1f$^\circ$ < Dec. < %1.1f$^\circ$' % (sdec.val, sdec.val+srang.val), ha='center', color='black', size='large')

# Update of the plot with values in the sliders
def update(val):
    '''This function updates the main and additional plots for new values of redshift, redshift range, and points transparency'''
    decDelta, decCen, newAlpha = [sval.val for sval in [srang, sdec, salpha]]
    # Update the value of the Text object
    fig_text.set_text('Wedge diagram within %1.1f$^\circ$ < Dec. < %1.1f$^\circ$' % (round(decCen,1), round(decCen,1)+round(decDelta,1)))
    ax2.cla()
    xyplt[0].set_data(rad_tot[0][cond_dec(decCen, decDelta)[0]], z_tot[0][cond_dec(decCen, decDelta)[0]])
    xyplt[0].set_alpha(alpha=newAlpha)
    mollfig(decCen, decDelta)
    plt.draw()
[sval.on_changed(update) for sval in [sdec, srang, salpha]]

# Samples selection box
# rax, labels = plt.axes([0.05, 0.15, 0.07, 0.15]), ('LSS', 'Isolated', 'Pairs', 'Triplets')
# check = CheckButtons(rax, labels, (True, True, False, False))
# def func(label):
#     '''This function allows the sample selection'''
#     lab = labels.index(label)
#     xyplt[lab].set_visible(not xyplt[lab].get_visible())
#     plt.draw()
# check.on_clicked(func)

# Reset button
resetax = plt.axes([0.05, 0.035, 0.07, 0.04])
button = Button(resetax, 'Reset', color='red', hovercolor='green')
def reset(event):
    '''This function reset the default values in the plots'''
    [sval.reset() for sval in [sdec, srang, salpha]]
button.on_clicked(reset)

plt.show()