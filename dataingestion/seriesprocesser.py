""" This class represents tasks created by the Data Ingestion Service used for the parallel
processing of input series """

from multiprocessing import Process
import logging
import sys
import coloredlogs
import pandas as pd
import numpy as np


class SeriesProcesser(Process):

    """
    #==============================================================================================#
    #                   CLASS OBJECTS ATTRIBUTES (Extended from the Superclass)                    #
    #==============================================================================================#
    #             TYPE                   NAME                         DESCRIPTION                  #
    #==============================================================================================#
    #  DataIngestionConfiguration   _conf             #The Data Ingestion Service Configuration    #
    #  Logger                       _log              #The Logger object used by the task          #
    #  list (String)                _series_columns   #The column headers' format of valid series  #
    #  Function                     _process_series   #Address of the series processing function   #
    #  Pipe                         _pipe             #Pipe for communicating with the main task   #
    #==============================================================================================#
    """

    def run(self):

        """
        DESCRIPTION:  Task runnable function, completing the seriesProcesser task initializations
                      and containing its main cycle

        ARGUMENTS:    None

        RETURNS:      None

        RAISES:       Nothing
        """

        # Initialize the task's logger object with a console handler only
        self._log = logging.getLogger(__name__)
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = coloredlogs.ColoredFormatter("%(asctime)s [%(name)s] %(levelname)s: "
                                                         "%(task_name)s %(message)s",
                                                         "%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(console_formatter)
        if self._conf.log.log_level_console == "CRITICAL":
            console_handler.setLevel(logging.CRITICAL)
        elif self._conf.log.log_level_console == "WARNING":
            console_handler.setLevel(logging.WARNING)
        elif self._conf.log.log_level_console == "INFO":
            console_handler.setLevel(logging.INFO)
        else:
            console_handler.setLevel(logging.DEBUG)
        self._log.addHandler(console_handler)
        self._log.setLevel(console_handler.level)

        # Add the task name to the logging messages
        self._log = logging.LoggerAdapter(self._log, {"task_name": " [Task " + self.name + "]:\t"})

        # Create a template Pandas dataframe with the column headers' of the expected format to be
        # be used as template for generating temporary buffers for storing valid series
        df_temp_template = pd.DataFrame(np.empty((0, self._conf.sample_size + 2)),
                                        columns=self._series_columns)

        # --------------------------- Main seriesProcessor Task cycle -----------------------------#
        while True:

            # Initialize the temporary dataframe to be used in the next run
            df_temp = df_temp_template.copy()

            # Await for the next processing command containing the type of series to be processed
            # and the list of series assigned to this task
            command = self._pipe.recv()

            # If the "stop" command was received, perform shutdown operations
            if command[0] == "stop":
                self._pipe.close()   # Close the pipe used for communicating with the main task
                self.close()         # Set this task as closed
                return               # Return to the caller, which deallocates the task

            # Define a pointer to the configuration of the type of series to be processed
            if command[0] == "labeled":
                current_type = self._conf.labeled
            else:
                current_type = self._conf.unlabeled

            # Retrieve the list of series to be processed by this task
            input_series = command[1]

            # Start processing series
            for index, series in enumerate(input_series):

                self._log.info("Processing %s series \"%s\"\t(%i/%i)", current_type.series_type,
                               series.name, index + 1, len(input_series))
                try:
                    self._process_series(series, current_type, df_temp, self._log)
                    # NOTE: in multi-core mode the return of the _process_series telling whether
                    #       the series is valid or malformed is of no use

                # If an OSError was propagated from the _process_series function, attempt to
                # process the next series, if any
                except OSError as oserr:
                    self._log.critical("A \"%s\" exception has occurred while processing input"
                                       " series \"%s\", attempting to process the next one (%s)",
                                       type(oserr).__name__, series.name, oserr.args[1])

            # Return the temporary dataframe containing the valid series the task has processed
            self._pipe.send(df_temp)

    def __init__(self, task_id, conf, series_columns, series_proc_fun, pipe):
        """
        DESCRIPTION:  Class Constructor, initializing the task configuration to the one used by the
                      Data Ingestion Service

        ARGUMENTS:    - task_id:           The task identifier used as its name

                      - conf:              A copy of the configuration used by the Data Ingestion
                                           Service

                      - series_columns:    The column headers' format of valid series

                      - series_proc_fun:   The address of the "_process_series" function to be used
                                           for processing series

                      - pipe:              The pipe used by the task for synchronizing with the
                                           Data Ingestion Service main task

        RETURNS:      None

        RAISES:       Nothing
        """

        # Call the "Process" parent constructor setting the task "name" attribute to the "id"
        # parameter (+1)
        super().__init__(name=str(task_id+1))

        # Set the service configuration to the same of the Data Ingestion Service
        self._conf = conf

        # Declare the task's logger (initialization must be performed in the run() for some reason)
        self._log = 0

        # Set the address of the series' processing function to the method of the Data Ingestion
        # Service
        self._process_series = series_proc_fun

        # Set the column headers' format of valid series to the same of the Data Ingestion Service
        self._series_columns = series_columns

        # Set the pipe used by the task for synchronizing with the Data Ingestion Service main task
        self._pipe = pipe
