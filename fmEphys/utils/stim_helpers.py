import numpy as np


def align_timing():

    lag_range = np.arange(-0.2, 0.2, 0.002)
    cc = np.zeros(np.shape(lag_range))
    t1 = np.arange(5, len(self.dEye)/60 - 120, 20).astype(int) # was np.arange(5,1600,20), changed for shorter videos
    t2 = t1 + 60
    offset = np.zeros(np.shape(t1))
    ccmax = np.zeros(np.shape(t1))
    imu_interp = interp1d(self.imuT_raw, self.gyro_z)
    for tstart in tqdm(range(len(t1))):
        for l in range(len(lag_range)):
            try:
                c, lag = nanxcorr(-self.dEye[t1[tstart]*60 : t2[tstart]*60] , imu_interp(self.eyeT[t1[tstart]*60 : t2[tstart]*60]+lag_range[l]), 1)
                cc[l] = c[1]
            except:
                cc[l] = np.nan
        offset[tstart] = lag_range[np.argmax(cc)]    
        ccmax[tstart] = np.max(cc)
    offset[ccmax<0.2] = np.nan

    # figure
    self.check_imu_eye_alignment(t1, offset, ccmax)

    # fit regression to timing drift
    model = LinearRegression()
    dataT = np.array(self.eyeT[t1*60 + 30*60])
    model.fit(dataT[~np.isnan(offset)].reshape(-1,1), offset[~np.isnan(offset)]) 
    self.ephys_offset = model.intercept_
    self.ephys_drift_rate = model.coef_
    self.plot_regression_timing_fit(dataT, offset)

if self.fm:
    self.imuT = self.imuT_raw - (self.ephys_offset + self.imuT_raw * self.ephys_drift_rate)

for i in self.ephys_data.index:
    self.ephys_data.at[i,'spikeT'] = np.array(self.ephys_data.at[i,'spikeTraw']) - (self.ephys_offset + np.array(self.ephys_data.at[i,'spikeTraw']) * self.ephys_drift_rate)
self.cells = self.ephys_data.loc[self.ephys_data['group']=='good']