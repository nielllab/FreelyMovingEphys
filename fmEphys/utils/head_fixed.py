






def head_fixed(cfg):

    offset=0.1, drift_rate=-0.000114

    # treadmill data
    treadmill_data_path = os.path.join(cfg['_rpath'], '{}_treadmill_preprocessing.h5'.format(cfg['_rname']))
    treadmill_data = fmEphys.read_h5(ephys_data_path)