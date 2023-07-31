"""Module containing the Configuration class for a generic service in the pipeline."""
import json
import jsonschema


class Configuration:
    """Configuration class representing a JSON configuration file
    for a generic service in the pipeline.

    Attributes:
        configuration_path (str): The absolute path of the JSON configuration file.
        schema_path (str): The absolute path of the JSON schema file.
        field (dict): The JSON configuration file in dictionary form.
            The dictionary contains the parsed configuration, regardless of its validity.
            An invalid configuration is notified by an exception.
    """

    def parse_configuration(self):
        """Parse JSON configuration from a file and store it as a dictionary
        in the self.field attribute.

        Raises:
            OSError: If the configuration file cannot be opened.
            JSONDecodeError: If the JSON configuration cannot be decoded correctly.
        """
        with open(self.configuration_path, 'r') as configuration_file:
            self.field = json.load(configuration_file)

    def validate_configuration(self):
        """Validate JSON configuration stored in the self.field attribute
        against a JSON schema.

        Raises:
            OSError: If the schema file cannot be opened.
            JSONDecodeError: If the JSON schema cannot be decoded correctly.
            ValidationError: If the JSON configuration is invalid.
            SchemaError: If the JSON schema is invalid.
        """
        with open(self.schema_path, 'r') as schema_file:
            json_schema = json.load(schema_file)

        jsonschema.validate(self.field, json_schema)

    def __init__(self, configuration_path, schema_path):
        """Initializer.

        Args:
            configuration_path (str): The absolute path of the JSON configuration file.
            schema_path (str): The absolute path of the JSON schema file,

        Raises:
            OSError: If the configuration or schema file cannot be opened.
            JSONDecodeError: If the JSON configuration or schema cannot be decoded correctly.
            ValidationError: If the JSON configuration is invalid.
            SchemaError: If the JSON schema is invalid.
        """
        self.configuration_path = configuration_path
        self.schema_path = schema_path
        self.field = []

        self.parse_configuration()
        self.validate_configuration()
