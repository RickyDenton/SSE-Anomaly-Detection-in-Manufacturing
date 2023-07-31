# Anomaly detection in manufacturing

## Overview
The pipeline steps are implemented in the following folders, each one representing a package:
- _dataingestion_
- _datapreparation_
- _datasegregation_
- _candidatemodelevaluation_
- _anomalydetectionandperformance_

The remaining sections of the repository are organized in this way:
- _configuration_ contains the configuration class used by the services;
- _test_ contains the scripts used to perform the elasticity and responsiveness tests.

## Requirements
The external Python packages needed to run the pipeline can be installed with pip:
```bash
pip install -r requirements.txt
```

## Execution

###### WARNING: By default the _dataingestion_ module is configured to run in ***multi-core mode*** using ***all cores available in the system***, which may lead to temporary system hangs during its execution.<br>The module's execution mode (single- or multi-core) as well the maximum number of cores to be used for processing can be set by modifying its *"multiCoreEnable"* and/or *"multiCoreLimit"* configuration parameters (_"dataingestion/configuration/DataIngestionService_configuration_default.json"_)

1. Simulate the arrival of raw labeled and unlabeled input series to be ingested.<br>This can be obtained by copying within the  _dataingestion_ module the template time series stored in the _"utils/series/labeledSeries"_ and _"utils/series/unlabeledSeries"_ folders into the _"datastore/raw/labeledSeries"_ and _"datastore/raw/unlabeledSeries"_ folders respectively (the latter thar are also automatically created when the data ingestion service is initialized).
   ```
     .
     ├── ...
     ├── dataingestion            
     │   ├── ...
     │   ├── datastore   
     │   │   ├── ...
     │   │   ├── raw
     │   │   │   ├── labeledSeries
     │   │   │   │   ├── 2020-09-01 00-00-00.dat
     │   │   │   │   ├── 2020-09-01 00-08-00.dat
     │   │   │   │   │
     │   │   │   │   └── ...
     │   │   │   └── unlabeledSeries
     │   │   │       ├── 2020-10-10 00-00-00.dat
     │   │   │       ├── 2020-10-10 00-08-00.dat
     │   │   │       │
     │   │   │       └── ...
     │   │   └── ...
     │   └── ...                
     └── ...  
   ```
   
2. Run _pipeline.py_ in training mode:
   ```bash
   python pipeline.py --mode=training
   ```

3. After the training has been completed successfully, run _pipeline.py_ in classification mode:
   ```bash
   python pipeline.py --mode=classification
   ```