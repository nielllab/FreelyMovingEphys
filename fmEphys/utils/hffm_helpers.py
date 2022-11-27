
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

import scipy.interpolate
import sklearn.linear_model

import fmEphys


def calc_offset_drift(eyeT, dEye_dpf, imuT, gyro_z, pdf=None):

    # Time range to use when testing cross correlation    
    t1 = np.arange(5, len(dEye_dpf)/60 - 120, 20).astype(int) # was np.arange(5,1600,20), changed for shorter videos
    t2 = t1 + 60

    lag_range = np.arange(-0.2, 0.2, 0.002)
    cc = np.zeros(np.shape(lag_range))

    offset = np.zeros(np.shape(t1))
    ccmax = np.zeros(np.shape(t1))
    imu_interp = scipy.interpolate.interp1d(imuT, gyro_z)
    for tstart in tqdm(range(len(t1))):
        for lag in range(len(lag_range)):
            try:
                c, _ = fmEphys.nanxcorr(-dEye_dpf[t1[tstart]*60 : t2[tstart]*60],
                                        imu_interp(eyeT[t1[tstart]*60 : t2[tstart]*60] + lag_range[lag]),
                                        1)
                cc[lag] = c[1]
            except:
                cc[lag] = np.nan
        offset[tstart] = lag_range[np.argmax(cc)]    
        ccmax[tstart] = np.max(cc)
    offset[ccmax < 0.2] = np.nan

    if pdf is not None:
        fig, [[ax0,ax1,ax2],[ax3,ax4,ax5]] = plt.subplots(2,3, figsize=(11,8.5), dpi=300)

        ax0.plot(eyeT[t1*60], offset)
        ax0.set_xlabel('secs')
        ax0.set_ylabel('offset (secs)')
        
        ax1.plot(eyeT[t1*60], ccmax)
        ax1.set_xlabel('secs')
        ax1.set_ylabel('max cc')

    # Fit regression to timing drift
    timingRegression = sklearn.linear_model.LinearRegression()
    dataT = np.array(eyeT[t1*60 + 30*60])
    timingRegression.fit(dataT[~np.isnan(offset)].reshape(-1,1),
                         offset[~np.isnan(offset)])
    model_offset = timingRegression.intercept_
    model_drift = timingRegression.coef_

    if pdf is not None:

        dataT = dataT[~np.isnan(dataT)]
        offset = offset[~np.isnan(dataT)]

        ax2.plot(dataT, offset, 'k.', label='nanxcorr offset')
        ax2.plot(dataT, model_offset+dataT*model_drift, color='r',
                                        label='corrected with regression')
        ax2.set_xlabel('secs')
        ax2.set_ylabel('offset (secs)')
        ax2.set_title('offset={0:.2f}; drift={0:.6f}'.format(model_offset, model_drift))

        for _ax in [ax3,ax4,ax5]:
            _ax.axis('off')

        fig.tight_layout()
        pdf.savefig(); plt.close()

    return model_offset, model_drift