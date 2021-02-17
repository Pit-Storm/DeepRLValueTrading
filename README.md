# Deep Reinforcement Learning for value stocktrading

This repository is the work related to the masters thesis **Deep Reinforcement Learning for algorithmic stock trading**. written in 2020/2021.

The goal was to train some DRL Agents for trading stocks and investigate limitations and chances for this algorithmic trading strategy. Additionally I compared performance to benchmarks and other basic algorithms.

## Prerequisites

At first setup your environment. I developed with Python v3.6.12 due to some restrictions at university server. There is a requirements file. Use it to install all necesarry packages: `pip install -r requirements.txt`.

I hardly recommend to use VisualStudio Code with Development container. There is a complete configuration. So you only have to startup Docker and open VSCode. Just click "Reopen in Container" and docker will do the rest.

If you are not able to use Docker and VSCode create a virtual environment and install packages there.

## Run the scripts

At first you should check the [README of etl folder](./etl/README.md) to see what kind of data you need. Because of licensing limitations I am not able to provide the data but you can re-download it for yourself.

1. Run the ETL pipeline. (See the [readme there](./etl/README.md))
2. Run `python main.py --algo A2C` to train, evaluate and test an A2C Agent.

There are lot more parameters you can pass to the script. Just run `python main.py --help` for help or check [the config](./config.py) for further information.

## Contact

I'm open for questions and discussions. Just open a github issue.

## Disclaimer

Useage of the software on your own risk. There are no warranties for anything!

## License

See [LICENSE](./LICENSE) file in this repo. In short: It's MIT.