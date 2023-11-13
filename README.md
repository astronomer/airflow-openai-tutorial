Orchestrate OpenAI operations with Apache Airflow
==================================================

This repository contains the DAG code used in the [Orchestrate OpenAI operations with Apache Airflow tutorial](https://docs.astronomer.io/learn/airflow-openai). 

The DAG in this repository uses the following package:

- [OpenAI Airflow provider](https://airflow.apache.org/docs/apache-airflow-providers-openai/stable/index.html).
- [OpenAI Python client](https://pypi.org/project/openai/)
- [scikit-learn](https://scikit-learn.org/stable/).
- [pandas](https://pandas.pydata.org/).
- [numpy](https://numpy.org/).
- [matplotlib](https://matplotlib.org/).
- [seaborn](https://seaborn.pydata.org/).

# How to use this repository

This section explains how to run this repository with Airflow. Note that you will need to copy the contents of the `.env_example` file to a newly created `.env` file. You will need to have a valid OpenAI API key of at [least tier 1](https://platform.openai.com/docs/guides/rate-limits/usage-tiers) to run this repository.

Download the [Astro CLI](https://docs.astronomer.io/astro/cli/install-cli) to run Airflow locally in Docker. `astro` is the only package you will need to install locally.

1. Run `git clone https://github.com/astronomer/use-case-mlflow.git` on your computer to create a local clone of this repository.
2. Install the Astro CLI by following the steps in the [Astro CLI documentation](https://docs.astronomer.io/astro/cli/install-cli). Docker Desktop/Docker Engine is a prerequisite, but you don't need in-depth Docker knowledge to run Airflow with the Astro CLI.
3. Create a `.env` file in the root of your cloned repository and copy the contents of the `.env_example` file to it. Provide your own OpenAi API key in the `.env` file.
4. Run `astro dev start` in your cloned repository.
5. After your Astro project has started. View the Airflow UI at `localhost:8080`.
6. Run the `captains_dag` DAG manually by clicking the play button. Provide your own question and adjust the parameters in the DAG to your liking.

In this project `astro dev start` spins up 4 Docker containers:

- The Airflow webserver, which runs the Airflow UI and can be accessed at `https://localhost:8080/`.
- The Airflow scheduler, which is responsible for monitoring and triggering tasks.
- The Airflow triggerer, which is an Airflow component used to run deferrable operators.
- The Airflow metadata database, which is a Postgres database that runs on port 5432.

## Resources

- [Orchestrate OpenAI operations with Apache Airflow](https://docs.astronomer.io/learn/airflow-openai).
- [OpenAI Airflow provider documentation](https://airflow.apache.org/docs/apache-airflow-providers-openai/stable/index.html).
- [OpenAI documentation](https://platform.openai.com/docs/introduction).