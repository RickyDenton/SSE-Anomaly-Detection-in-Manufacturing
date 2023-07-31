""" Data Ingestion Service configuration class, extending the general "Configuration" class """

from pathlib import Path
from configuration import Configuration
from .seriestypeconfiguration import SeriesTypeConfiguration
from .loggingconfiguration import LoggingConfiguration


class DataIngestionConfiguration(Configuration):

    """
    #==============================================================================================#
    #                   CLASS OBJECTS ATTRIBUTES (Extended from the Superclass)                    #
    #==============================================================================================#
    #          TYPE                   NAME                           DESCRIPTION                   #
    #==============================================================================================#
    # static string            schema_path          #The path to the service JSON schema file      #
    # static string            default_conf_path    #The path to the service default configuration #
    # int                      sample_size          #The expected sample size of each series       #
    # string                   default_label        #The base label of samples within a series     #
    # int                      starting_index       #The starting index of samples within a series #
    # int                      max_series_per_run   #Max. num. of valid series to process per run  #
    # bool                     multi-core-enable    #Whether to use multiple cores, if available   #
    # int                      multi-core-limit     #The maximum number of cores to be used        #
    # SeriesTypeConfiguration  labeled              #Service labeled series configuration          #
    # SeriesTypeConfiguration  unlabeled            #Service unlabeled series configuration        #
    # LoggingConfiguration     log                  #Service logging configuration                 #
    #==============================================================================================#
    """

    # The relative path of the Data Ingestion Service JSON Schema file
    schema_path = "configuration/DataIngestionService_configuration.schema.json"

    # The relative path of the Data Ingestion Service default JSON Configuration
    default_conf_path = "configuration/DataIngestionService_configuration_default.json"

    def to_dict(self):

        """
        DESCRIPTION:  Class Dictionary Serializer

        ARGUMENTS:    None

        RETURNS:      A Python dictionary containing the object's attributes

        RAISES:       Nothing
        """

        return {"sample_size": self.sample_size, "default_label": self.default_label,
                "starting_index": self.starting_index, "max_series_per_run":
                self.max_series_per_run, "multi-core-enable": self.multi_core_enable,
                "multi-core-limit": self.multi_core_limit, "labelled_series_configuration":
                self.labeled.to_dict(), "unlabeled_series_configuration": self.unlabeled.to_dict(),
                "logging_configuration": self.log.to_dict()}

    def __str__(self):

        """
        DESCRIPTION:  Class String Serializer

        ARGUMENTS:    None

        RETURNS:      A String containing the object's attributes serialized as a Python Dictionary

        RAISES:       Nothing
        """

        return str(self.to_dict())

    def __init__(self, logger, custom_config=None):

        """
        DESCRIPTION:  Class Constructor, which initializes the Data Ingestion Service configuration

        ARGUMENTS:    - logger:           The logger used by the service to be configured
                      - custom_config:    The absolute path to a custom JSON configuration
                                          to be applied

        RETURNS:      None

        RAISES:       - JSONDecodeError:  If the service JSON schema or configuration cannot be
                                          decoded correctly
                      - SchemaError:      If the service JSON schema is invalid
                      - ValidationError:  If the service JSON configuration is invalid
                      - OSError:          If the service JSON schema or configuration cannot be read
        """

        # Absolute path of the file's directory
        curr_dir = Path(__file__).parent

        # Pass the absolute path of the custom configuration being passed (or the relative path
        # to the default configuration if none was provided) to the parent's constructor, which
        # validates is against the service's JSON schema
        if custom_config is not None:
            super().__init__(custom_config, curr_dir / DataIngestionConfiguration.schema_path)
        else:
            super().__init__(curr_dir / DataIngestionConfiguration.default_conf_path,
                             curr_dir / DataIngestionConfiguration.schema_path)

        # Initialize the configuration general parameters
        self.sample_size = self.field["sampleSize"]
        self.default_label = self.field["defaultLabel"]
        self.starting_index = self.field["startingIndex"]
        self.max_series_per_run = self.field["maxSeriesPerRun"]
        self.multi_core_enable = self.field["multiCoreEnable"]
        self.multi_core_limit = self.field["multiCoreLimit"]

        # Initialize the service configuration with respect to labeled series
        self.labeled = SeriesTypeConfiguration("labeled", self.field["labeledSeriesConfiguration"])

        # Initialize the service configuration with respect to unlabeled series
        self.unlabeled = SeriesTypeConfiguration("unlabeled",
                                                 self.field["unlabeledSeriesConfiguration"])

        # Initialize the service logging configuration
        self.log = LoggingConfiguration(self.field["loggingConfiguration"], logger)

        # Deallocates the "field" dictionary, being no longer necessary
        del self.field
