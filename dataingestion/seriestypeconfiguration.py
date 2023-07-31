""" Data Ingestion Service Configuration with respect to a type of series (labeled or unlabeled) """

from pathlib import Path


class SeriesTypeConfiguration:

    """
    #==============================================================================================#
    #                                   CLASS OBJECTS ATTRIBUTES                                   #
    #==============================================================================================#
    #  TYPE               NAME                                   DESCRIPTION                       #
    #==============================================================================================#
    # string   series_type                  #The type of series this configuration refers to       #
    # string   input_dir_path               #The path of the folder where to fetch input series    #
    # string   input_file_extension         #The file extension of the input series                #
    # string   input_file_datetime_format   #The datetime format used as name for the input series #
    # string   input_series_separator       #The values separator used in the input series         #
    # string   output_file_path             #The path of the file where to append valid series to  #
    # string   output_series_separator      #The separator to be used to append valid series       #
    # float    max_null_perc                #The max perc. of null values for a series to be valid #
    # int      max_consec_null              #The max. consec. NULL values for a series to be valid #
    # string   null_filling_strategy        #How NULL values in a series should be filled          #
    # string   duplicated_policy            #If duplicated valid input series should be discarded  #
    # string   malformed_policy             #If malformed series should be saved into their dir.   #
    # string   malformed_output_dir_path    #The path of the folder where to put malformed series  #
    #==============================================================================================#
    """

    def to_dict(self):

        """
        DESCRIPTION:  Class Dictionary Serializer

        ARGUMENTS:    None

        RETURNS:      A Python dictionary containing the object's attributes

        RAISES:       Nothing
        """

        return {"series_type": self.series_type, "input_dir_path": self.input_dir_path,
                "input_file_extension": self.input_file_extension, "input_file_datetime_format":
                self.input_file_datetime_format, "input_series_separator":
                self.input_series_separator, "output_file_path": self.output_file_path,
                "output_series_separator": self.output_series_separator, "max_null_perc":
                self.max_null_perc, "max_consec_null": self.max_consec_null,
                "null_filling_strategy": self.null_filling_strategy, "duplicated_policy":
                self.duplicated_policy, "malformed_strategy": self.malformed_policy,
                "malformed_output_dir_path": self.malformed_output_dir_path}

    def __str__(self):

        """
        DESCRIPTION:  Class String Serializer

        ARGUMENTS:    None

        RETURNS:      A String containing the object's attributes serialized as a Python Dictionary

        RAISES:       Nothing
        """

        return str(self.to_dict())

    def __init__(self, series_type, config):

        """
        DESCRIPTION:  Class Constructor, which initializes the configuration of the Data Ingestion
                      Service with respect to a type of series (labeled or unlabeled)

        ARGUMENTS:    - series_type:  A string specifying the type of series this object refers to
                                      ("labeled" or "unlabeled")
                      - config:       A dictionary specifying the configuration to be applied

        RETURNS:      None

        RAISES:       Nothing
        """

        # Absolute path of the file's directory
        curr_dir = Path(__file__).parent

        # Initialize the SeriesType configuration
        self.series_type = series_type
        self.input_dir_path = curr_dir / config["inputDirPath"]
        self.input_file_extension = config["inputFileExtension"]
        self.input_file_datetime_format = config["inputFileDatetimeFormat"]
        self.input_series_separator = config["inputSeriesSeparator"]
        self.output_file_path = curr_dir / config["outputFilePath"]
        self.output_series_separator = config["outputSeriesSeparator"]
        self.max_null_perc = config["maxNULLperc"]
        self.max_consec_null = config["maxConsecNULL"]
        self.null_filling_strategy = config["NULLFillingStrategy"]
        self.duplicated_policy = config["duplicatedPolicy"]
        self.malformed_policy = config["malformedPolicy"]
        self.malformed_output_dir_path = curr_dir / config["malformedOutputDirPath"]
