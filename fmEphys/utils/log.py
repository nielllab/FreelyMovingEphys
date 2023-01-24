"""
FreelyMovingEphys/src/utils/log.py
"""
import os
import sys

import fmEphys

class Logger(object):
    """
    call as

    sys.stdout = fmEphys.Logger(os.path.split(cfg_path)[0])


    """

    def __init__(self, writepath):

        date_str, time_str = fmEphys.fmt_now()

        log_path = os.path.join(writepath,
                        'errlog_{}_{}.log'.format(date_str, time_str))

        self.terminal = sys.stdout
        self.log = open(log_path, "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass