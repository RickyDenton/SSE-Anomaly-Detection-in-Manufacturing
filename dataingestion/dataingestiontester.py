""" Test Class for the Data Ingestion Service """

from pathlib import Path
import datetime
import pandas as pd
from dataingestion import DataIngestionService


class DataIngestionTester(DataIngestionService):

    """
    #==============================================================================================#
    #                   CLASS OBJECTS ATTRIBUTES (Extended from the Superclass)                    #
    #==============================================================================================#
    #     TYPE           NAME                                DESCRIPTION                           #
    #==============================================================================================#
    #   DataFrame    _labeled_df      #The reference dataframe used to generate labeled series     #
    #   Dataframe    _unlabeled_df    #The reference dataframe used to generate unlabeled series   #
    #==============================================================================================#
    """

    def simulate_labeled_series(self, num_series, clean_output_file=False):

        """
        DESCRIPTION:  Interface function for simulating the arrival of new labeled series in the
                      input folder of the Data Ingestion Service object under test by placing
                      EXACTLY "num_series" series in such folder

        ARGUMENTS:    - num_series:         The number of labeled series to be placed in the
                                            input folder of the Data Ingestion Service object
                                            under test

                      - clean_output_file:  Whether the output file should be cleaned after the
                                            series have been placed (to improve performance
                                            under test)

        RETURNS:      Nothing

        RAISES:       Nothing
        """

        return self._simulate_series(num_series, "labeled", clean_output_file)

    def simulate_unlabeled_series(self, num_series, clean_output_file=False):

        """
        DESCRIPTION:  Interface function for simulating the arrival of new unlabeled series in the
                      input folder of the Data Ingestion Service object under test by placing
                      EXACTLY "num_series" series in such folder

        ARGUMENTS:    - num_series:         The number of unlabeled series to be placed in the
                                            input folder of the Data Ingestion Service object
                                            under test

                      - clean_output_file:  Whether the output file should be cleaned after the
                                            series have been placed (to improve performance
                                            under test)

        RETURNS:      Nothing

        RAISES:       Nothing
        """

        return self._simulate_series(num_series, "unlabeled", clean_output_file)

    def _simulate_series(self, num_series, series_type, clean_output_file=False):

        """
        DESCRIPTION:  Simulates the arrival of new labeled or unlabeled series in the input
                      folder of the Data Ingestion Service object under test by placing
                      EXACTLY "num_series" series in such folder

        ARGUMENTS:    - num_series:         The number of series to be placed in the input folder
                                            of the Data Ingestion Service object under test

                      - series_type:        Whether labeled or unlabeled series should be added
                                            to their respective input directories

                      - clean_output_file:  Whether the output file should be cleaned after the
                                            series have been placed (to improve performance
                                            under test)

        RETURNS:      Nothing

        RAISES:       Nothing
        """

        # Define a pointer to the configuration and the dataframe that should be used to generate
        # series depending on the type of series to be added
        if series_type == "labeled":
            current_type = self._conf.labeled
            df = self._labeled_df
        else:
            current_type = self._conf.unlabeled
            df = self._unlabeled_df

        # Clean the input folder of the relative series type
        for file in Path(current_type.input_dir_path).iterdir():
            Path.unlink(file)

        # --- Add "num_series" new series to the input folder of the appropriate type

        # Full dataframe cycles
        full_dataframe_cycles = int(num_series/df.shape[0])

        for _ in range(0, full_dataframe_cycles):
            for (_, row) in df.iterrows():
                se = pd.DataFrame(row)
                se = se.transpose()
                se.to_csv(current_type.input_dir_path / Path(str(se.index[0]).replace(":", "-") +
                                                             ".dat"), index=False, header=False)

            # Add a constant time offset to all timestamps to ensure file names uniqueness
            df.rename(index=lambda date: date + datetime.timedelta(weeks=6), inplace=True)

        # Last dataframe cycle
        single_cycle = num_series % df.shape[0]
        for j in range(0, single_cycle):
            se = pd.DataFrame(df.iloc[j])
            se = se.transpose()
            se.to_csv(current_type.input_dir_path / Path(str(se.index[0]).replace(":", "-") +
                                                         ".dat"), index=False, header=False)

            # Add a constant time offset to all timestamps to ensure file names uniqueness
            df.rename(index=lambda date: date + datetime.timedelta(weeks=6), inplace=True)

        # --- Clean the output file of the relative series type, if required
        if clean_output_file is True:
            with open(current_type.output_file_path, "w") as out_file:
                out_file.write(current_type.output_series_separator.join(self._series_columns))

    def __init__(self, custom_config=None):

        """
        DESCRIPTION:  Class Constructor, Initializing a Tester for the Data Ingestion Service

        ARGUMENTS:    - custom_config:    The absolute path to a custom JSON configuration to
                                          initialize the service with (optional)

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

        # Absolute path of the file's directory
        curr_dir = Path(__file__).parent

        # Initialize the Data Ingestion Service with the custom configuration, if passed, or with
        # the default testing configuration otherwise
        if custom_config is not None:
            super().__init__(custom_config)
        else:
            super().__init__(curr_dir /
                             "configuration/DataIngestionService_configuration_testing.json")

        # Load the reference dataframes that will be used to generate labeled and unlabeled series
        self._labeled_df = pd.read_csv(curr_dir / "utils/datasets/labeledSeries_full.csv",
                                       sep=self._conf.labeled.output_series_separator,
                                       parse_dates=['timestamp'], dtype="float64",
                                       index_col="timestamp")
        self._unlabeled_df = pd.read_csv(curr_dir / "utils/datasets/unlabeledSeries_full.csv",
                                         sep=self._conf.unlabeled.output_series_separator,
                                         parse_dates=['timestamp'], dtype="float64",
                                         index_col="timestamp")

        # Cast the "timestamp" columns in both dataframe to "int32"
        self._labeled_df["ANOMALOUS"] = self._labeled_df["ANOMALOUS"].astype("int32")
        self._unlabeled_df["ANOMALOUS"] = self._unlabeled_df["ANOMALOUS"].astype("int32")

        # Drop the "ANOMALOUS" column of the unlabeled data frame, since it not needed
        self._unlabeled_df.drop(columns=["ANOMALOUS"], inplace=True)


# if __name__ == "__main__":
#    test = DataIngestionTester()
#    test.close()
