""" Data Ingestion Service Main Class """
from pathlib import Path
from shutil import copyfile
from multiprocessing import Pipe, cpu_count
import datetime
import logging
import pandas as pd
import numpy as np
from dataingestion.dataingestionconfiguration import DataIngestionConfiguration
from dataingestion.seriesprocesser import SeriesProcesser


class DataIngestionService:

    """
    #==============================================================================================#
    #                                   CLASS OBJECTS ATTRIBUTES                                   #
    #==============================================================================================#
    #             TYPE                   NAME                         DESCRIPTION                  #
    #==============================================================================================#
    #  DataIngestionConfiguration   _conf             #The Data Ingestion Service Configuration    #
    #  Logger                       _log              #The Logger object used by the service       #
    #  list<String>                 _series_columns   #The column headers' format of valid series  #
    #  list<Pipe>                   _tasks_pipes      #Pipes for communicating with worker tasks   #
    #==============================================================================================#
    """

    def start_ingest_labeled_series(self, max_series=None):
        """
        DESCRIPTION:  Interface function for ingesting labeled series

        ARGUMENTS:    - max_series:  The maximum number of valid labeled series to be processed
                                     (optional, overrides the configuration "max_series_per_run"
                                      parameter)

        RETURNS:      The number of labeled series that were ingested into the output file

        RAISES:       Nothing
        """
        return self._ingest_series("labeled", max_series)

    def start_ingest_unlabeled_series(self, max_series=None):
        """
        DESCRIPTION:  Interface function for ingesting unlabeled series

        ARGUMENTS:    - max_series:  The maximum number of valid unlabeled series to be processed
                                     (optional, overrides the configuration "max_series_per_run"
                                      parameter)

        RETURNS:      The number of unlabeled series that were ingested into the output file

        RAISES:       Nothing
        """
        return self._ingest_series("unlabeled", max_series)

    def _process_series(self, series_file, current_type, df_temp, log):

        """
        DESCRIPTION:  Processes an input series checking for its validity as specified in the
                      service's configuration, where:

                      - If the series is VALID, it is appended to the temporary data frame "df_temp"

                      - If the series is MALFORMED, if specified in the configuration it is copied
                        to the output malformed directory of the relative type with its malformation
                        cause appended on a new line

                      Regardless of its type, the input series is removed from the relative input
                      folder after it has been processed

        ARGUMENTS:    - series_file:   The file path of the input series to be processed
                      - current_type:  The SeriesTypeConfiguration of the type of series to
                                       be processed (self._conf.labeled or self._conf.unlabeled)
                      - df_temp:       The temporary data frame where to append the series, if valid
                      - log:           The Logger object to be used for logging purposes (the one
                                       of the Data Ingestion Service in case of single-core mode
                                       or the one of a SeriesProcesser task in case of multi-core
                                       mode)

        RETURNS:      A bool representing whether the series is valid (True) or malformed (False)

        RAISES:       - OSError:          Generic error in accessing the file system
        """

        malformed_cause = ""        # Used to store the series' malformed cause, if any

        # Check whether the series is empty
        if series_file.stat().st_size == 0:
            malformed_cause = "empty series"
            log.warning(" The series \"%s\" is empty", series_file.name)
        else:

            # Extract the series' timestamp from the file name, falling back to the file creation
            # date in case its name doesn't match the DateTime format specified in the configuration
            try:
                series_timestamp = datetime.datetime.strptime(series_file.name[:-4],
                                                              current_type.
                                                              input_file_datetime_format)
            except ValueError:
                log.warning(" The file name of series \"%s\" doesn't match the DateTime format"
                            " specified in the configuration, and its creation date will be used"
                            " as timestamp", series_file.name)
                series_timestamp = datetime.datetime.fromtimestamp(int(series_file.stat().st_ctime))
                # NOTE: The semantic of the "st_ctime" value varies depending on the OS:
                #         - Windows systems  ->  File Creation Date
                #         - Linux systems    ->  File Last Modified Date

            # Process the series by importing it as a Pandas data frame
            try:
                df = pd.read_csv(series_file, header=None,
                                 sep=current_type.input_series_separator, dtype="float64")

                # Check the series to consist in a single row
                if df.shape[0] != 1:
                    malformed_cause = "multi-row series"
                    log.warning(" The series \"%s\" presents multiple rows (%i)",
                                series_file.name, df.shape[0])
                else:

                    # In case of labeled series, check the first "ANOMALOUS" field value to be
                    # valid (0 or 1)
                    if (current_type.series_type == "labeled") and (df.at[0, 0] != 0)\
                            and (df.at[0, 0] != 1):
                        malformed_cause = "invalid \"ANOMALOUS\" value"
                        log.warning(" The labeled series \"%s\" presents an invalid \"ANOMALOUS\" "
                                    "value (%s)", series_file.name, str(df.at[0, 0]))
                    else:

                        # In case of unlabeled series, add a first "ANOMALOUS" field, initializing
                        # it to "-1"
                        if current_type.series_type == "unlabeled":
                            df.insert(0, column="ANOMALOUS", value=-1)

                        # Compute the delta between the expected and the actual number of samples
                        # in the series ("+1" refers to the "ANOMALOUS" column)
                        samples_delta = self._conf.sample_size + 1 - df.shape[1]

                        # If there are MORE samples than expected, truncate them to the expected
                        if samples_delta < 0:
                            log.warning(" The series \"%s\" contains %i more sample(s) than "
                                        "expected, which will be truncated", series_file.name,
                                        abs(samples_delta))
                            # NOTE: In the expression below the Python Type Checker complains on the
                            #       parameter being passed to the df.columns, but it's the fastest
                            #       way to obtain the desired result
                            # noinspection PyTypeChecker
                            df.drop(df.columns[[i for i in range(self._conf.sample_size+1,
                                                                 df.shape[1])]], axis=1,
                                    inplace=True)

                        # Else, if there are less samples than expected, fill with NULL values
                        elif samples_delta > 0:
                            log.warning(" The input series \"%s\" contains %i less sample(s) than"
                                        " expected, which will be treated as NULL value(s)",
                                        series_file.name, samples_delta)
                            for i in range(df.shape[1], self._conf.sample_size+1):
                                df[i] = np.nan

                        # Compute the total and maximum consecutive NULL values in the series
                        null_samples_count = df.iloc[0].isnull().sum()
                        max_consec_null = pd.Series.max(df.iloc[0].isnull().astype(int).groupby
                                                        (df.iloc[0].notnull().astype(int).cumsum())
                                                        .sum())

                        # Check the total number of NULL values to be in bound for the series
                        # to be valid
                        if null_samples_count > current_type.max_null_perc * self._conf.sample_size:
                            malformed_cause = "NULL values threshold exceeded (" + \
                                             str(null_samples_count) + " > " + \
                                             str(int(current_type.max_null_perc *
                                                     self._conf.sample_size)) + ")"
                            log.warning(" The total number of NULL values in the input series"
                                        " \"%s\" exceeds the maximum threshold (%i > %i)",
                                        series_file.name, null_samples_count,
                                        int(current_type.max_null_perc * self._conf.sample_size))

                        # Check the number of consecutive NULL values to be in bound for the series
                        # to be valid
                        elif max_consec_null > current_type.max_consec_null:
                            malformed_cause = "Consecutive NULL values threshold exceeded " \
                                             "(" + str(max_consec_null) + " > " +\
                                             str(current_type.max_consec_null) + ")"
                            log.warning(" The maximum number of consecutive NULL values in the"
                                        " input series \"%s\" exceeds the maximum threshold (%i"
                                        " > %i)", series_file.name, max_consec_null,
                                        current_type.max_consec_null)

                        # --- From this point the series is VALID ---

                        # Fill the NULL values in the series as specified in the configuration
                        else:
                            if current_type.null_filling_strategy == "linearInterpolation":
                                df.interpolate(method="linear", axis=1, inplace=True)
                            elif current_type.null_filling_strategy == "zeroFill":
                                df.fillna(0, axis=1, inplace=True)
                            else:
                                df = df.fillna(method=current_type.null_filling_strategy, axis=1)
                                # BUG: setting fillna(inplace=True) in this last case
                                #      raises a Pandas "notImplemented" Exception

                            # Insert the series' timestamp as a new column in front
                            df.insert(0, column="timestamp", value=series_timestamp)

                            # Set the series column headers' to the specified format
                            df.columns = self._series_columns

                            # Append the series to the temporary data frame as a new row
                            df_temp.loc[df_temp.shape[0]] = df.iloc[0]

            # If Pandas raises a ValueError one or more samples in the series could not be
            # interpreted as float64, signifying to a malformed series
            except ValueError as val_err:
                malformed_cause = "ValueError in parsing the series (" + str(val_err.args[0]) + ")"
                log.warning(" Error in parsing the values of input series \"%s\" (%s)",
                            series_file.name, val_err.args[0])

        # ---------------------------------- Cleanup Operations -----------------------------------#
        try:

            # If the series is malformed and should be copied to the "malformedOutputDir"
            if malformed_cause != "" and current_type.malformed_policy != "drop":

                # Copy the malformed series to the malformed_output_dir relative to its type
                malformed_output_dir = Path(current_type.malformed_output_dir_path)
                series_copy = copyfile(series_file, malformed_output_dir / series_file.name)

                # Append the "malformed_cause" to the copied series on a new line
                with open(series_copy, "a") as out_file:
                    out_file.write("\n"+malformed_cause)

            # Remove the series from the input folder
            Path.unlink(series_file)

        except OSError as oserr:
            log.warning(" A \"%s\" exception has occurred while performing cleanup operations on"
                        " the input series \"%s\" (%s)", type(oserr).__name__, series_file.name,
                        oserr.args[1])

        # Return True or False depending on whether the series is valid or malformed
        if malformed_cause != "":
            return False
        else:
            return True

    def _ingest_series(self, series_type, max_series=None):

        """
        DESCRIPTION:  Main procedure of the Data Ingestion Service, ingesting labeled or unlabeled
                      series as specified by the "type" attribute from their input folder to their
                      output file

        ARGUMENTS:    - type:       A string specifying the type of series to be ingested
                                    ("labeled" or "unlabeled")
                      - max_series: The maximum number of valid series of the appropriate type to
                                    be ingested
                                    (NOTE: this functionality is provided in single-core mode only)

        RETURNS:      The number of series that have been ingested in the output file of the
                      associated series type

        RAISES:       Nothing
        """

        # Set the maximum number of valid series to be processed to the "max_series" argument, if
        # passed, or to the configuration "max_series_per_run" otherwise
        #
        # NOTE: if multicore processing is enabled such functionality is NOT provided
        if max_series is None:
            max_series = self._conf.max_series_per_run
        else:

            # Check the "max_series" to be a positive integer
            if not(isinstance(max_series, int) and (max_series > 0)):
                self._log.critical("The \"max_series\" parameter, if passed, must be a positive"
                                   " integer (received \"%s\")", str(max_series))
                return 0

            # If multi-core processing is enabled, warn the user that passing the max_series
            # argument has not effect
            if self._conf.multi_core_enable is True:
                self._log.warning(" Limiting the processing to \"%i\" valid series has no"
                                  " effect if multicore is enabled: all valid series will be"
                                  " processed", max_series)

        # Define a pointer to the configuration of the type of series to be ingested in this
        # execution of the service
        if series_type == "labeled":
            current_type = self._conf.labeled
        else:
            current_type = self._conf.unlabeled

        # Log the beginning of the ingestion
        self._log.info("    [Data Ingestion Service]: Ingesting %s Series",
                       current_type.series_type.capitalize())
        self._log.debug("   Configuration: %s", self._conf)
        self._log.debug("   Series Column Headers: %s", str(self._series_columns))
        self._log.info("    ==============================================================")
        if max_series > 0:
            self._log.info("    Maximum number of valid %s series to be processed:  %i",
                           series_type, max_series)

        # Create a temporary Pandas dataframe with the column headers' of the expected format for
        # it to be used as temporary buffer for storing valid series
        df_temp = pd.DataFrame(np.empty((0, self._conf.sample_size + 2)),
                               columns=self._series_columns)

        # Retrieve the list of files in the input directory relative to the type of series to
        # ingest, separating series from other files on whether they present the expected extension
        input_series = []
        wrong_files = []
        try:
            for file in Path(current_type.input_dir_path).iterdir():
                if file.name[-len(current_type.input_file_extension):] ==\
                        current_type.input_file_extension:
                    input_series.append(file)
                else:
                    wrong_files.append(file)
        except OSError as oserr:
            self._log.critical("A \"%s\" exception has occurred while attempting to read files "
                               "from the \"inputDirPath\" where fetch series from, aborting the"
                               " ingestion (%s)", type(oserr).__name__, oserr.args[1])
            self._log.info("    ==============================================================\n")
            return 0

        # The number of input series and other files that was found in the input directory
        input_series_count = len(input_series)
        wrong_files_count = len(wrong_files)

        # Log the results of the analysis of the input directory
        if wrong_files_count > 0:
            self._log.warning(" The specified \"inputDirPath\" where fetch series from contains"
                              " the following files of unexpected extension, that will be"
                              " ignored: %s", [file.name for file in wrong_files])
        if input_series_count > 0:
            self._log.info("    Number of %s series found in the input folder:  %i", series_type,
                           input_series_count)
        else:                                                # If no series were found, return
            self._log.info("    No %s series were found in the input folder, stopping ingestion",
                           series_type)
            self._log.info("    ==============================================================\n")
            return 0

        # -------------------------------- Input Series Processing --------------------------------#

        # If multi-core processing is enabled
        if self._conf.multi_core_enable is True:

            # Set the number of CPUs to be used for parallel processing to the minimum among the
            # available seriesProcesser tasks and the series to be parsed
            used_cpu = min(input_series_count, len(self._pipes))

            # Divide evenly the list of input series to be processed in "used_cpu" sub-lists to be
            # assigned to the seriesProcesser tasks
            input_series_division = np.array_split(input_series, used_cpu)

            # Send a command message to each seriesProcesser task to be used containing the type
            # and his list of sub-series to be processed
            for i in range(0, used_cpu):
                self._pipes[i].send([series_type, input_series_division[i]])

            # Await the temporary dataframes returned by each seriesProcesser task and merge them
            # into the final temporary dataframe
            for i in range(0, used_cpu):
                df_temp = df_temp.append(self._pipes[i].recv(), ignore_index=True)

        # Otherwise, if multicore processing is disabled
        else:

            # Define the total number of series that will be processed in this run
            if max_series == 0:
                series_to_process = input_series_count
            else:
                series_to_process = min(input_series_count, max_series)

            # Initialize the number of valid and malformed series found during processing
            valid_series_count = 0
            malformed_series_count = 0

            # For each series in the input folder
            for index, series in enumerate(input_series):
                self._log.info("    Processing %s series \"%s\"  (%i/%i)", series_type,
                               series.name, index+1, series_to_process)
                try:

                    # Process the series, incrementing the relative count variable depending on
                    # whether it is valid or malformed
                    if self._process_series(series, current_type, df_temp, self._log) is True:
                        valid_series_count += 1
                    else:
                        malformed_series_count += 1

                    # If the number of maximum valid series to process per run has been reached,
                    # stop
                    # processing series
                    if (max_series > 0) and (valid_series_count == max_series):
                        self._log.info("    The maximum number of valid series has been processed"
                                       "  (%i)", max_series)
                        break

                # If an OSError was propagated from the _process_series function, attempt to
                # process the next series, if any
                except OSError as oserr:
                    self._log.critical("A \"%s\" exception has occurred while processing input"
                                       " series\"%s\", attempting to process the next one (%s)",
                                       type(oserr).__name__, series.name, oserr.args[1])

                # If a keyboard interrupt was received, stop processing series
                except KeyboardInterrupt:
                    self._log.warning(" Keyboard interrupt received, stopping processing series")
                    break

        # --------------------------------- End series processing ---------------------------------#

        # Regardless of whether they were tracked during the analysis (single-core mode) or not
        # (multi-core mode), the number of valid series that were found is equal to the number of
        # rows in the temporary dataframe
        valid_series_count = df_temp.shape[0]
        malformed_series_count = input_series_count - valid_series_count

        # Initialize the number of duplicated series (for visibility purposes)
        duplicated_series_count = 0

        # Initialize the number of series to be ingested to the output file to the number of rows
        # of the temporary dataframe (which apart from the case of OSError exception propagating
        # from the _process_series function corresponds to the valid_series_count)
        ingested_series_count = df_temp.shape[0]

        # If no series to ingest were found, just print a message
        if ingested_series_count == 0:
            self._log.info("    No valid %s series to ingest were found", series_type)

        # Otherwise, attempt to persist the new series into the output file
        else:

            try:
                # Load the output file relative to the type of series in a Pandas DataFrame
                # (The "parse_dates" and dtype="float64" arguments are for optimization purposes)
                out_df = pd.read_csv(current_type.output_file_path,
                                     sep=current_type.output_series_separator,
                                     parse_dates=['timestamp'], dtype="float64")

                # Cast the "ANOMALOUS" column in both the temporary and the output dataframe to int
                df_temp["ANOMALOUS"] = df_temp["ANOMALOUS"].astype("int32")
                out_df["ANOMALOUS"] = out_df["ANOMALOUS"].astype("int32")

                # Append the temporary to the output dataframe
                out_df = out_df.append(df_temp, ignore_index=True)

                # If duplicated series must be filtered from the results
                if current_type.duplicated_policy != "save":

                    self._log.info("    -----------------------------"
                                   "---------------------------------")
                    self._log.info("    Checking for duplicates between the valid series and the"
                                   " ones in the output file")
                    # Obtain a mask of booleans of size equal to the number of rows in the output
                    # dataframe telling whether each series is a duplicate of a series with lower
                    # index (where the "timestamp" column is not considered for duplicate purposes)
                    duplicates_mask = out_df.duplicated(subset=self._series_columns[1:])

                    # To prevent altering the original contents of the output data frame, reset all
                    # values in the masks not referring to the temporary data frame that was
                    # appended
                    duplicates_mask[:out_df.shape[0] - df_temp.shape[0]] = False

                    # Count the number of duplicates in the series that were added
                    duplicated_series_count = duplicates_mask.sum()

                    # If duplicate series were found, drop them and set accordingly the number of
                    # ingested series
                    if duplicated_series_count > 0:
                        self._log.info("    %i out of %i of the valid series were found duplicated"
                                       ", and will be discarded",
                                       duplicated_series_count, valid_series_count)
                        out_df.drop(out_df[duplicates_mask].index, inplace=True)
                        ingested_series_count = ingested_series_count - duplicated_series_count

                # If after duplicates filtering there are still series to ingest, persist the
                # output dataframe as the new output file
                if ingested_series_count > 0:
                    out_df.to_csv(current_type.output_file_path,
                                  sep=current_type.output_series_separator, index=False)

            # If an exception occurred while attempting to persist changes, print an error and
            # reset the number of series that have been ingested
            except OSError as oserr:
                self._log.critical("A \"%s\" exception has occurred while attempting to persisting"
                                   " the processed valid series in the output file \"%s\","
                                   "(changes are NOT saved (%s)", type(oserr).__name__,
                                   current_type.output_file_path, oserr.args[1])
                ingested_series_count = 0  # Reset the number of series that were ingested

        # Print a summary of the processed series
        self._log.info("    --------------------------------------------------------------")
        if duplicated_series_count > 0:
            self._log.info("    Processed Series Summary:  Ingested: %i, Valid: %i (of which %i "
                           "duplicated), Malformed: %i", ingested_series_count, valid_series_count,
                           duplicated_series_count, malformed_series_count)
        else:
            self._log.info("    Processed Series Summary:  Ingested: %i, Valid: %i, Malformed: %i",
                           ingested_series_count, valid_series_count, malformed_series_count)
        self._log.info("    ==============================================================\n")

        # Return the number of ingested series
        return ingested_series_count

    def _check_resource_valid(self, res, exp_res_type, current_series):

        """
        DESCRIPTION:  Utility function used to assert that an resource in terms of an output file
                      or a folder used by the service exists, also performing error correction
                      operations according to the following resource

                      If the resource is a file:
                      -------------------------
                        - If the file exists, assert its column headers' format to match the one
                          expected by the service, raising a ValueError exception otherwise
                        - If the file doesn't exist or is empty, initialize it with the expected
                          column headers' format

                      If the resource is a directory:
                      ------------------------------
                         - If the directory doesn't exist, create it

                      Where in both cases if an expected resource is of the wrong type (a file
                      is a directory and viceversa) a FileExistsException is raised

        ARGUMENTS:    - res:              The absolute path of the resource to check the
                      - exp_res_type:     The expected resource type ("file" or "dir")
                      - current_series:   The SeriesTypeConfiguration associated to the resource
                                          (used for retrieving the "output_file_series_separator"
                                          when creating output files)

        RETURNS:      None

        RAISES:       - FileExistsError:  If a file is in fact a directory or viceversa
                      - ValueError:       If the column headers' format in the file differ from the
                                          one expected by the service as of its configuration
        """

        # Path to the resource
        res_path = Path(res)

        # If the resource exists
        if res_path.exists():

            # Check the resource to be of the correct type, raising a ValueError Exception otherwise
            if res_path.is_file() and exp_res_type == "dir":
                self._log.critical("The expected directory \"%s\" is a file", res_path.absolute())
                raise FileExistsError("The resource " + str(res_path.absolute()) +
                                      " is a file and not a directory")
            if res_path.is_dir() and exp_res_type == "file":
                self._log.critical("The output file for %s series corresponds to a directory (%s)",
                                   current_series.series_type, res_path.absolute())
                raise FileExistsError("The output file for " + current_series.series_type +
                                      " series corresponds to a directory (" +
                                      str(res_path.absolute()) + ")")

            # If the resource is a file
            if res_path.is_file():

                # If the file is empty, initialize it with the expected column headers' format
                if res_path.stat().st_size == 0:
                    self._log.warning(" The output file for %s series is empty, and will be"
                                      " initialized with the expected column headers' format "
                                      "(%s)", current_series.series_type, res_path.absolute())
                    with open(res_path, "w") as out_file:
                        out_file.write(current_series.output_series_separator.join
                                       (self._series_columns))

                # Otherwise, assert its first line to match the expected column headers' format
                else:
                    with open(res_path) as out_file:
                        out_file_first_line = out_file.readline()
                    out_file_columns = out_file_first_line. \
                        split(self._conf.labeled.output_series_separator)
                    if out_file_columns[-1][-1] == "\n":                   # Remove a trailing '\n'
                        out_file_columns[-1] = out_file_columns[-1][:-1]   # in the last column
                    if out_file_columns != self._series_columns:
                        self._log.critical("The column headers' format in the output file for %s "
                                           "series does not match with the one expected by the "
                                           "service (expected: %s, in file: %s)",
                                           current_series.series_type, self._series_columns[:5] +
                                           ["..."] + self._series_columns[-5:],
                                           out_file_columns[:5] + ["..."] + out_file_columns[-5:])
                        raise ValueError("The column headers' format in the output file for %s "
                                         "series does not match with the one expected by the "
                                         "service (expected %s, in file: %s)" +
                                         current_series.series_type, self._series_columns[:5] +
                                         ["..."] + self._series_columns[-5:],
                                         out_file_columns[:5] + ["..."] + out_file_columns[-5:])

        # Otherwise, if the resource doesn't exist
        else:

            # If the resource is an output file, create it with the expected column headers' format
            if exp_res_type == "file":
                self._log.warning(" The output file for %s series doesn't exist, and will be "
                                  "created with the expected column headers' format (%s)",
                                  current_series.series_type, res_path.absolute())
                Path.mkdir(res_path.parents[0], parents=True, exist_ok=True)
                with open(res_path, "w") as out_file:
                    out_file.write(current_series.output_series_separator.join
                                   (self._series_columns))

            # Otherwise, if it is a directory, create it
            else:
                self._log.warning(" The following directory doesn't exist, and will be created"
                                  " (%s)", res_path.absolute())
                Path.mkdir(res_path, parents=True, exist_ok=True)

    def get_config(self):

        """
        DESCRIPTION:  Returns the service configuration as a Python dictionary

        ARGUMENTS:    None

        RETURNS:      The service configuration as a Python dictionary

        RAISES:       Nothing
        """

        return self._conf.to_dict()

    def close(self):

        """
        DESCRIPTION:  Shuts down the seriesProcesser tasks associated to the Data Ingestion
                      Service, if any

        ARGUMENTS:    None

        RETURNS:      None

        RAISES:       Nothing
        """

        # If the service was executed in multi-core mode, close all pipes and send the stop signal
        # to all the seriesProcesser tasks
        if self._conf.multi_core_enable is True:
            for pipe in self._pipes:
                pipe.send(["stop", ])
                pipe.close()
            self._log.info("    [Data Ingestion Service]: Service closed")

    def __init__(self, custom_config=None):

        """
        DESCRIPTION:  Class Constructor, Initializing the Data Ingestion Service

        ARGUMENTS:    - custom_config:    The absolute path to a custom JSON configuration to be
                                          applied (optional)

        RETURNS:      None

        RAISES:       - JSONDecodeError:  If the service JSON schema or configuration cannot be
                                          decoded correctly
                      - SchemaError:      If the service JSON schema is invalid
                      - ValidationError:  If the service JSON configuration is invalid
                      - FileExistsError:  If a file used by the service is in fact a directory or
                                          viceversa
                      - ValueError:       If the column headers' format in the labeled or unlabeled
                                          output files do not match with the ones specified in the
                                          service configuration
                      - OSError:          Generic error in accessing the file system
        """

        # Initialize the logger object used by the service
        self._log = logging.getLogger(__name__)

        # Initialize the service configuration
        self._conf = DataIngestionConfiguration(self._log, custom_config)

        # Initialize the column headers' format of valid series as the following pattern:
        # ['timestamp','ANOMALOUS','defaultLabel'+'startingIndex','defaultLabel'+'startingIndex+1'.]
        self._series_columns = ["timestamp", "ANOMALOUS"]
        self._series_columns.extend([str(self._conf.default_label) +
                                     str(self._conf.starting_index+i)
                                     for i in range(0, self._conf.sample_size)])

        # File System Checks
        # ------------------
        # The following checks ensure that the file and directories used by the service as
        # specified in its configuration exist and present the expected column headers' format in
        # case of output files, where error correction operations are performed whenever possible

        # Input Directories
        self._check_resource_valid(self._conf.labeled.input_dir_path, "dir", self._conf.labeled)
        self._check_resource_valid(self._conf.unlabeled.input_dir_path, "dir", self._conf.unlabeled)

        # Output Files
        self._check_resource_valid(self._conf.labeled.output_file_path, "file", self._conf.labeled)
        self._check_resource_valid(self._conf.unlabeled.output_file_path, "file",
                                   self._conf.unlabeled)

        # Output Malformed Directories
        if self._conf.labeled.malformed_policy != "drop":
            self._check_resource_valid(self._conf.labeled.malformed_output_dir_path, "dir",
                                       self._conf.labeled)
        if self._conf.unlabeled.malformed_policy != "drop":
            self._check_resource_valid(self._conf.unlabeled.malformed_output_dir_path, "dir",
                                       self._conf.unlabeled)
        # ------------------

        # If the service is to be enabled in single-core mode, just print a message
        if self._conf.multi_core_enable is False:
            self._log.info("    [Data Ingestion Service]: Service successfully initialized in"
                           " single-core mode")

        # Otherwise perform multi-core mode initializations
        else:

            self._log.info("    [Data Ingestion Service]: Initializing the service in multi-core"
                           " mode (this may take a while...)")

            # Get the number of available CPUs in the system
            available_cpu = cpu_count()

            # Check whether a limit to the number of CPUs to use for parallel processing is set
            if (self._conf.multi_core_limit > 0) and (self._conf.multi_core_limit < available_cpu):
                available_cpu = self._conf.multi_core_limit

            # If there is just a CPU available, initializing the service in multi-core mode carries
            # no sense
            if available_cpu == 1:
                self._log.warning(" [Data Ingestion Service]: While configured in multi-core mode,"
                                  " a single core was detected, falling back to single-core mode")
                self._conf.multi_core_enable = False

            # Otherwise, if at least two processors are available
            else:

                # Initialize the lists for holding the pipes used for communicating with the
                # seriesProcesser tasks
                self._pipes = []

                # For each serviceProcesser task to be created
                for i in range(0, available_cpu):

                    # Obtain the two ends of a Pipe object that will be used for communicating with
                    # the seriesProcesser task
                    my_pipe, its_pipe = Pipe()

                    # Append our end of the pipe to the "_pipes" queue
                    self._pipes.append(my_pipe)

                    # Initialize the serviceProcesser task
                    service_processer = SeriesProcesser(i, self._conf, self._series_columns,
                                                        self._process_series, its_pipe)

                    # Start the serviceProcesser task
                    service_processer.start()

                # Print that the Data Ingestion Service has been initialized in multi-core mode
                self._log.info("    [Data Ingestion Service]: Service successfully initialized in"
                               " multi-core mode (%i CPUs)", available_cpu)


if __name__ == "__main__":
    serv = DataIngestionService()
    serv.start_ingest_labeled_series()
    serv.start_ingest_unlabeled_series()
    serv.close()
