import gstools as gs
import numpy as np
import skgstat as skg
import matplotlib.pyplot as plt
import time

# Grid/Raster of 1000 x 1000 pixels
shape = (1000, 1000)
x = np.arange(0, 1000)
y = np.arange(0, 1000)

# Using GSTools, let's generate a correlated signal at two different length: 5 and 100 (spherical)
model = gs.Spherical(dim=2, var=0.5, len_scale=5)
srf = gs.SRF(model, seed=42)

model2 = gs.Spherical(dim=2, var=0.5, len_scale=100)
srf2 = gs.SRF(model2, seed=42)

# We combine the two random correlated fields (e.g, short-range could represent resolution, and long-range the noise)
field = srf.structured([x, y]) + srf2.structured([x, y])

# Let's see how the correlated field looks
plt.imshow(field, vmin=-3, vmax=3)
plt.colorbar()

# Get grid coordinates, and fatten everything because we don't care about the 2D
xx, yy = np.meshgrid(x, y)
coords = np.dstack((xx.flatten(), yy.flatten())).squeeze()

# We generate a RasterEquidistantMetricSpace instance, providing the shape and coordinate extent of the grid.
# We run 100 samplings of pairwise distances between a random center sample of 25 points and several samples of
# equidistant points
# t0 = time.time()
# rems = skg.RasterEquidistantMetricSpace(coords, shape=shape, extent=(x[0], x[-1], y[0], y[-1]), samples=20,
#                                         runs=1000)
#
# # We compute the variogram with custom bins to look at all ranges more precisely
# custom_bins = np.concatenate([np.arange(0, 20)] + [np.arange(20, 200, 20)] + [np.arange(200, 1200, 200)])
#
# V1 = skg.Variogram(rems, field.flatten(), bin_func=custom_bins)
#
# t1 = time.time()
# print('Elapsed: '+str(t1-t0)+ ' seconds')

t2 = time.time()
rems2 = skg.RasterEquidistantMetricSpace2(coords, shape=shape, extent=(x[0], x[-1], y[0], y[-1]), samples=20,
                                        runs=1000)

# We compute the variogram with custom bins to look at all ranges more precisely
custom_bins = np.concatenate([np.arange(0, 20)] + [np.arange(20, 200, 20)] + [np.arange(200, 1200, 200)])

V2 = skg.Variogram(rems2, field.flatten(), bin_func=custom_bins)

t3 = time.time()
print('Elapsed: '+str(t3-t2)+ ' seconds')