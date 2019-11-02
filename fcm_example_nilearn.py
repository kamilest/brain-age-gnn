import numpy as np
import matplotlib.pyplot as plt

from nilearn import connectome

# generate data
# dim(x) = number ROI x time points
x = np.random.rand(10, 100)

def compute_connectivity_nilearn(x):
  conn_measure = \
    connectome.ConnectivityMeasure(kind='correlation', vectorize=True, discard_diagonal=True)
  flat_fcm_nilearn = conn_measure.fit_transform([np.transpose(x)])[0]
  
  return flat_fcm_nilearn

def compute_connectivity(x):
    mean = x.mean(1)
    std = x.std(1, ddof=x.shape[1] - 1)
    cov = np.dot(x, x.T) - (x.shape[1] * np.dot(mean[:, np.newaxis], mean[np.newaxis, :]))
    return cov / np.dot(std[:, np.newaxis], std[np.newaxis, :])

# fcm
fcm = compute_connectivity(x)

# mask fcm for lower triangle values
mask = np.invert(np.tri(fcm.shape[0], k=-1, dtype=bool))
m = np.ma.masked_where(mask == 1, mask)
lower_tri_fcm = np.ma.masked_where(m, fcm)

# plot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
ax[0].imshow(fcm)
ax[1].imshow(lower_tri_fcm)

# flatten
flat_fcm = lower_tri_fcm.compressed()
assert flat_fcm.size == (fcm.shape[0] * (fcm.shape[0] - 1))*0.5

flat_fcm_nilearn = compute_connectivity_nilearn(x)

print(flat_fcm)
print(flat_fcm_nilearn)
