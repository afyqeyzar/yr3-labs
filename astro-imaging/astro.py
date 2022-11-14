from astropy.io import fits
import matplotlib.pyplot as plt

hdulist = fits.open("./A1_mosaic.fits")

hist_data = hdulist[0].data

plt.hist(hist_data[0])
plt.show()