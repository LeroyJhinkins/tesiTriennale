from matplotlib.colors import LogNorm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.io import fits

plt.rcParams.update({'font.size': 14})
plt.rcParams['text.usetex'] = True  # use real LaTeX
plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'  # use siunitx

try:
    matplotlib.use("QtAgg")
except ImportError:
    print("QtAgg backend not available, using default backend.")
matplotlib.rcParams['toolbar'] = 'None'  # disable window bars


# Read FITS file ------------------------------------------------------------------------------------------------------------------
filepath = "data/catalogues/data_PDFz_EuclidLargeBox0001_Rot30degCircle_m3_obsz_z0.9-1.1.fits"
# filepath = "data/catalogues/data_target_PDFz_EuclidLargeBox0001_Rot30degCircle_m3_obsz_z0.9-1.1.fits" # correct

with fits.open(filepath) as hdul:
    hdul.info()

    table_hdu = hdul[1]
    table_data = table_hdu.data # type: ignore
    nData = table_data.shape[0]
    names = table_data.columns.names

    id_array = table_data[names[0]]
    ra_array = table_data[names[1]]
    dec_array = table_data[names[2]]
    z_array = table_data[names[3]]
    weight_array = table_data[names[4]]

    print("\nTotal number of points:", nData)
    print(f"Total Right Ascension range: [{np.min(ra_array)}; {np.max(ra_array)}]")
    print(f"Total Declination range: [{np.min(dec_array)}; {np.max(dec_array)}]")
    print(f"Total Redshift range: [{np.min(z_array)}; {np.max(z_array)}]")

# wedge selection -----------------------------------------------------------------------------------------------------------------
DEC_MIN = 48
DEC_MAX = 53
RA_MIN = 30
RA_MAX = 90

mask = (dec_array > DEC_MIN) & (dec_array < DEC_MAX) & (ra_array > RA_MIN) & (ra_array < RA_MAX)

ra_slice = ra_array[mask]
z_slice = z_array[mask]
ra_slice_rad = ra_slice * np.pi / 180.0

Z_MIN = np.min(z_slice)
Z_MAX = np.max(z_slice)

print(f"\nPlotting {len(ra_slice)} galaxies...")

# scatter plot --------------------------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 8), subplot_kw={'projection': 'polar'}, num="Wedge plot")

sc = ax.scatter(
    ra_slice_rad,
    z_slice,
    c='black',
    s=1,
    alpha=0.2,
    edgecolors='none'
)

# orientation
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)

# custom ticks
theta_ticks = np.linspace(np.min(ra_slice_rad), np.max(ra_slice_rad), 5)
r_ticks = np.concatenate([
    np.linspace(0, Z_MIN, 6, endpoint=False),
    np.linspace(Z_MIN, Z_MAX, 4)
])

ax.set_xticks(theta_ticks)
ax.set_xticklabels([f"{np.degrees(t):.0f}°" for t in theta_ticks])
ax.set_yticks(r_ticks)
ax.set_yticklabels([f"{t:.2f}" for t in r_ticks])

# placing radial labels just after RA_max
offset_deg = 15  # tweak for spacing
# ax.set_rlabel_position((np.max(ra_slice_rad)) + np.deg2rad(offset_deg))

# limits
ax.set_thetalim(RA_MIN * np.pi / 180.0, RA_MAX * np.pi / 180.0)
# ax.set_rlim(0, Z_MAX)

ax.tick_params(axis='x', pad=10)
ax.tick_params(axis='y', pad=5)

# axis labels
r_for_theta_label = Z_MAX * 1.15
theta_for_theta_label = np.mean([np.min(ra_slice_rad), np.max(ra_slice_rad)])

ax.text(
    theta_for_theta_label,
    r_for_theta_label,
    "Right Ascension (deg)",
    ha='center',
    va='top',
    rotation= np.degrees(theta_for_theta_label)*0.5 - 90, # tangential alignment
    rotation_mode='anchor',
    fontsize=14,
)

theta_for_r_label = np.min(ra_slice_rad) - np.radians(offset_deg)
r_for_r_label = 0.5 * Z_MAX

ax.text(
    theta_for_r_label,
    r_for_r_label,
    "Redshift $z$",
    ha='left',
    va='center',
    rotation= 90 - np.degrees(np.min(ra_slice_rad)), # radial alignment
    rotation_mode='anchor',
    fontsize=14,
)

ax.set_title('Wedge diagram within %1.1f$^\circ$ $\leq$ DEC $\leq$ %1.1f$^\circ$' % (DEC_MIN, DEC_MAX), fontsize=16, pad=20) # type: ignore

plt.show()