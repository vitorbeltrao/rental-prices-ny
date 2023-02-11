# Rental Price Prediction - v1.0.0

## Table of Contents

1. [Project Description](#Description)
2. [Files Description](#files)
3. [Running Files](#running)
4. [Licensing and Authors](#licensingandauthors)
***

## Project Description <a name="Description"></a>

The goal of the project is to create a machine learning model that predicts rents in real estate located in New York. This model can be very valuable for people who are looking to start their own AirBnb and don't know how to price it. In this case, the model can provide a first inference based on some characteristics of the property, so that the person can start selling their property on Airbnb without initially worrying about what value to put.

## Files Description <a name="files"></a>

In "rental-prices-ny" repository we have:

* **components**: Inside this folder, we have all the files needed to run the entire model pipeline, from raw data collection to final predictions for never-before-seen data. These are the final files for the production environment. Each component is a block in the model that performs some task and in general generates some output artifact to feed the next steps.

* **main.py file**: Main script in Python that runs all the components. All this managed by *MLflow* and *Hydra*.

* **conda.yaml file**: File that contains all the libraries and their respective versions so that the system works perfectly.

* **config.yaml**: This is the file where we have the environment variables necessary for the components to work.

* **environment.yaml**: This file is for creating a virtual *conda* environment. It contains all the necessary libraries and their respective versions to be created in this virtual environment.
***

## Running Files <a name="running"></a>

### Clone the repository

Go to [rental-prices-ny](https://github.com/vitorbeltrao/rental-prices-ny) and click on Fork in the upper right corner. This will create a fork in your Github account, i.e., a copy of the repository that is under your control. Now clone the repository locally so you can start working on it:

`git clone https://github.com/[your_github_username]/rental-prices-ny.git`

and go into the repository:

`cd rental-prices-ny`

### Create the environment

Make sure to have conda installed and ready, then create a new environment using the *environment.yaml* file provided in the root of the repository and activate it. This file contain list of module needed to run the project:

`conda env create -f environment.yaml`
`conda activate rental-prices-ny`

### Get API key for Weights and Biases

Let's make sure we are logged in to Weights & Biases. Get your API key from W&B by going to https://wandb.ai/authorize and click on the + icon (copy to clipboard), then paste your key into this command:

`wandb login [your API key]`

You should see a message similar to:

`wandb: Appending key for api.wandb.ai to your netrc file: /home/[your username]/.netrc`

### The configuration

The parameters controlling the pipeline are defined in the `config.yaml` file defined in the root of the repository. We will use Hydra to manage this configuration file.

Open this file and get familiar with its content. Remember: this file is only read by the `main.py` script (i.e., the pipeline) and its content is available with the `go` function in `main.py` as the `config` dictionary. For example, the name of the project is contained in the `project_name` key under the `main` section in the configuration file. It can be accessed from the `go` function as `config["main"]["project_name"]`.

### Running the entire pipeline or just a selection of steps

In order to run the pipeline when you are developing, you need to be in the root of the repository, then you can execute this command:

`mlflow run .`

This will run the entire pipeline.

If you want to run a certain steps you can use the examples of command bellow:

`mlflow run . -P steps=upload_raw_data`

This is useful for testing whether steps that have been added or developed can be performed or not.

If you want to run multiple steps (ex: `upload_raw_data` and the `transform_raw_data` steps), you can similarly do:

`mlflow run . -P steps=upload_raw_data,transform_raw_data`

> NOTE: Make sure the previous artifact step is available in W&B. Otherwise we recommend running each step in order.

You can override any other parameter in the configuration file using the Hydra syntax, by providing it as a `hydra_options` parameter. For example, say that we want to set the parameter 06_train_model -> random_forest -> n_estimators to 10 and 04_basic_clean->min_price to 50:

`mlflow run . -P steps=upload_raw_data,transform_raw_data,basic_cleaning -P hydra_options="06_train_model.random_forest.n_estimators=10 04_basic_clean.min_price=50"`

### Run existing pipeline

We can directly use the existing pipeline to do the training process without the need to fork the repository. All it takes to do that is to conda environment with MLflow and wandb already installed and configured. To do so, all we have to do is run the following command:

`mlflow run -v [pipeline_version] https://github.com/vitorbeltrao/rental-prices-ny.git`

`[pipeline_version]` is a release version of the pipeline. For example this repository has currently been released for version `1.0.0`. So we need to input `1.0.0` in place of `[pipeline_version]`.

We have successfully run the pipeline, we can see that in the W&B account there is a new project with the name 'nyc_airbnb_dev'. Step running pipelines can be seen in the artifact section of Graph View.

![wandb](rental-prices-ny\images\wandb.png)

## Licensing and Author <a name="licensingandauthors"></a>

Vítor Beltrão - Data Scientist

Reach me at: 

- vitorbeltraoo@hotmail.com

- [linkedin](https://www.linkedin.com/in/v%C3%ADtor-beltr%C3%A3o-56a912178/)

- [github](https://github.com/vitorbeltrao)

- [medium](https://pandascouple.medium.com)

Licensing: [MIT LICENSE](https://github.com/vitorbeltrao/customer_churn/blob/main/LICENSE)