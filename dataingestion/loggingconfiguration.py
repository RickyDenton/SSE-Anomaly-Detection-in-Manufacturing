""" Data Ingestion Service logging configuration """

from pathlib import Path
import sys
import logging
import coloredlogs


class LoggingConfiguration:

    """
    #==============================================================================================#
    #                                   CLASS OBJECTS ATTRIBUTES                                   #
    #==============================================================================================#
    #   TYPE           NAME                                   DESCRIPTION                          #
    #==============================================================================================#
    #  string   log_level_console     #The min. priority level for events to be logged to std.out  #
    #  bool     log_to_file           #Whether to output logging events to a log file              #
    #  string   log_level_file        #The min. priority level for events to be logged to file     #
    #  string   log_file_write_mode   #The writing mode of logging events to the log file          #
    #  string   log_file_path         #The relative path to the logging file                       #
    #==============================================================================================#
    """

    def to_dict(self):

        """
        DESCRIPTION:  Class Dictionary Serializer

        ARGUMENTS:    None

        RETURNS:      A Python dictionary containing the object's attributes

        RAISES:       Nothing
        """

        return {"log_level_console": self.log_level_console, "log_to_file": self.log_to_file,
                "log_level_file": self.log_level_file, "log_file_write_mode":
                self.log_file_write_mode, "log_file_path": self.log_file_path}

    def __str__(self):

        """
        DESCRIPTION:  Class String Serializer

        ARGUMENTS:    None

        RETURNS:      A String containing the object's attributes serialized as a Python Dictionary

        RAISES:       Nothing
        """

        return str(self.to_dict())

    def __init__(self, config, logger):

        """
        DESCRIPTION:  Class Constructor, which initializes the Data Ingestion Service logger

        ARGUMENTS:    - config:   A dictionary specifying the logging configuration to be applied
                      - logger:   The logger object used by the service to be configured

        RETURNS:      None

        RAISES:       Nothing
        """

        # Absolute path of the file's directory
        curr_dir = Path(__file__).parent

        # Initialize the logging configuration
        self.log_level_console = config["logLevelConsole"]
        self.log_to_file = config["logToFile"]
        self.log_level_file = config["logLevelFile"]
        self.log_file_write_mode = config["logFileWriteMode"]
        self.log_file_path = curr_dir / config["logFilePath"]

        """ Logger Object Initialization """

        # Create a Console Handler for the logger
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = coloredlogs.ColoredFormatter("%(asctime)s [%(name)s] %(levelname)s: "
                                                         "%(message)s", "%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(console_formatter)
        if self.log_level_console == "CRITICAL":
            console_handler.setLevel(logging.CRITICAL)
        elif self.log_level_console == "WARNING":
            console_handler.setLevel(logging.WARNING)
        elif self.log_level_console == "INFO":
            console_handler.setLevel(logging.INFO)
        else:
            console_handler.setLevel(logging.DEBUG)
        logger.addHandler(console_handler)
        logger.setLevel(console_handler.level)

        # If logging to a file is required
        if self.log_to_file is True:

            # Check the "log_file_path" parent folder in which to write the log file to exist
            log_file_dir = Path(self.log_file_path).parents[0]
            if log_file_dir.exists():                            # Check the log_file_dir to exist
                pass
            else:                                  # If the log_file_dir does not exist, create it
                logger.warning(" The specified parent folder of the \"logFilePath\" where to write"
                               " the log file to does not exist, and will be created (%s)",
                               log_file_dir.absolute())
                Path.mkdir(log_file_dir, parents=True, exist_ok=True)

            # Create a file handler for the logger
            file_handler = logging.FileHandler(self.log_file_path, mode=self.log_file_write_mode)
            file_formatter = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: "
                                               "%(message)s", "%Y-%m-%d %H:%M:%S")
            file_handler.setFormatter(file_formatter)
            if self.log_level_file == "CRITICAL":
                file_handler.setLevel(logging.CRITICAL)
            elif self.log_level_file == "WARNING":
                file_handler.setLevel(logging.WARNING)
            elif self.log_level_file == "INFO":
                file_handler.setLevel(logging.INFO)
            else:
                file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
            if logger.getEffectiveLevel() > file_handler.level:
                logger.setLevel(file_handler.level)
