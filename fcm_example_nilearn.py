import numpy as np
import matplotlib.pyplot as plt

from nilearn import connectome
from sklearn.covariance import EmpiricalCovariance

# generate data
# dim(x) = number ROI x time points
x = np.random.rand(10, 100)

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

# nilearn
# cov_estimator = EmpiricalCovariance(assume_centered=False, store_precision=False)
conn_measure = connectome.ConnectivityMeasure(kind='correlation')
flat_conn_measure = connectome.ConnectivityMeasure(kind='correlation', vectorize=True, discard_diagonal=True)
    
    
fcm_nilearn = conn_measure.fit_transform([np.transpose(x)])[0]

# mask fcm for lower triangle values
mask = np.invert(np.tri(fcm.shape[0], k=-1, dtype=bool))
m = np.ma.masked_where(mask == 1, mask)
lower_tri_fcm_nilearn = np.ma.masked_where(m, fcm_nilearn)

# plot
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
ax[0].imshow(fcm_nilearn)
ax[1].imshow(lower_tri_fcm_nilearn)

# flatten
flat_fcm_nilearn = flat_conn_measure.fit_transform([np.transpose(x)])[0]

# flat_fcm
# [-0.22396498 -0.06469943  0.05901101 -0.12767227  0.04273985 -0.114231
#  -0.09133419 -0.06872676 -0.1485927   0.10269797  0.34135601 -0.09145457
#  -0.05290028  0.04392535 -0.03695624  0.06197338  0.08983608 -0.04054554
#  -0.06384553  0.07748559  0.09122788 -0.02771935 -0.03934788 -0.14328504
#   0.11016629  0.13398746  0.03600536 -0.0763943  -0.02114351 -0.13015345
#  -0.0940339  -0.0441423   0.085561    0.1492538   0.19296644  0.06398614
#  -0.08157816 -0.07716618 -0.10816207  0.11659565  0.05677561 -0.03518457
#  -0.02496297  0.18802075 -0.00261855]

# flat_fcm_nilearn
# [-0.01821334 -0.0052615   0.00479891 -0.0103826   0.0034757  -0.00928952
#  -0.0074275  -0.00558902 -0.01208389  0.00835163  0.02775984 -0.00743729
#  -0.00430197  0.00357211 -0.00300536  0.00503982  0.00730567 -0.00329725
#  -0.00519206  0.0063013   0.00741886 -0.0022542  -0.00319986 -0.01165226
#   0.00895897  0.01089616  0.00292804 -0.00621256 -0.00171944 -0.01058437
#  -0.00764705 -0.00358975  0.00695801  0.01213766  0.01569247  0.0052035
#  -0.00663412 -0.00627533 -0.00879598  0.00948182  0.00461712 -0.00286129
#  -0.00203005  0.01529027 -0.00021295]

# flattened correlation matrices for one UKB subject (UKB1000028)
# [0.38082493 0.54088703 0.23219155 ... 0.38100959 0.11460386 0.31701128]
# [0.39161042 0.55620571 0.23876754 ... 0.39180031 0.1178496  0.32598948]