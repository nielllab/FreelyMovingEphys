"""
fmEphys/utils/log.py

Log errors.


Written by DMM, 2021
"""


import os
import sys
import fmEphys as fme


class Logger(object):
    """ Log messasges and errors.

    Parameters
    ----------
    writepath : str
        Filepath where the log file will be written.

    Example use
    -----------
    Call as
    $ sys.stdout = fme.Logger(os.path.split(cfg_path)[0])

    """


    def __init__(self, writepath):

        date_str, time_str = fme.fmt_now()

        # Automaticlly name the logging file.
        log_path = os.path.join(writepath,
                'errlog_{}_{}.log'.format(date_str,
                                          time_str))

        # Antyhing logged into terminal will also go
        # into the log object.
        self.terminal = sys.stdout
        self.log = open(log_path, "a")
   

    def write(self, message):
        """Write message to file.
        
        Parameters
        ----------
        message : str
            Message to write in the log file.

        """

        self.terminal.write(message)
        self.log.write(message)  


    def flush(self):
        pass

    