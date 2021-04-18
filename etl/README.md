# Provider: EODhistoricaldata.com

Provides wide variaty of Data through API access. Payment needed.

## Prerequisites

* Create a file named `API_KEY` (it is ignored from git) and place it in this folder (etl).
* **IMPORTANT:** Run ETL files with iPython or from within directory `etl`
* To get an API_KEY you need to buy a package from [eodhistoricaldata.com](https://eodhistoricaldata.com). They offer 50% off for students and academic reasons. Even though they aren't that expansive compared to other providers.

## Run the ETL pipeline

Just go to the folder `./etl` (e.g. with `cd etl`) and run `python run_etl.py`. If the script runs without errors you have all the necessary data inside `./data` and you can start training some algorithms.

