# ======================================================================================
#
# Project:          Python Machine Learning - Perceptron
# Description:      Perceptron controller script
# Author:           Peter Melson
# Version:          0.1
# Date:             December 2023
#
# ======================================================================================

import logging
import numpy as np
import pandas as pd
import sys
import yaml
import matplotlib.pyplot as plt
import ml_perceptron as mlp


# --------------------------------------------------------------------------------------
# ConfigurationClass
# --------------------------------------------------------------------------------------
class Configuration:
    """
    Configuration class to manage application configuration data.
    """

    def __init__(self, cfg_filename):
        """
        Constructor.
        :param cfg_filename: (str) The file name and path of the
        configuration data yaml file.
        """

        # Load configuration details from yaml file.
        with open(cfg_filename, "r") as yamlfile:
            self.__cfg = yaml.load(stream=yamlfile, Loader=yaml.FullLoader)

    def print(self):
        """
        Print method to output configuration data to the console.
        :return:
        """

        # Output loaded configuration data to console.
        print()
        for key in self.__cfg:
            print("{0:25} {1}".format(key, self.__cfg[key]))
        print()

    def get_item(self, key):
        """
        Accessor to retrieve a specific configuration data item from the
        collection.
        :param key: (str) The configuration data item to be returned.
        :return: The value associated with the configuration data item.
        """

        # Return configuration item for supplied key.
        return self.__cfg[key]

    def size(self):
        """
        Method to return the number of configuration data items in the
        collection.
        :return: (int) The number of configuration data items in the
        collection.
        """

        # Return the number of configuration items.
        return self.__cfg.__len__()


# --------------------------------------------------------------------------------------
# Load Configuration Data
# --------------------------------------------------------------------------------------
LOCAL = (len(sys.argv) > 1) and (sys.argv[1] == "local")
if LOCAL:
    ML_CFG = Configuration(cfg_filename="src/ml_cfg_local.yaml")
else:
    ML_CFG = Configuration(cfg_filename="/app/src/ml_cfg.yaml")
ML_CFG.print()
MAJOR_VERSION_NAME = ML_CFG.get_item(key="MAJOR_VERSION_NAME")
VERSION = ML_CFG.get_item(key="VERSION")
OUTPUT_FILEPATH = ML_CFG.get_item(key="OUTPUT_FILEPATH")

# --------------------------------------------------------------------------------------
# Logging Configuration
# --------------------------------------------------------------------------------------
LOG_LEVEL = ML_CFG.get_item(key="LOG_LEVEL")
LOG_FORMAT = ML_CFG.get_item(key="LOG_FORMAT")
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
LOGGER = logging.getLogger(name=__name__)

# --------------------------------------------------------------------------------------
# IRIS Dataset
# --------------------------------------------------------------------------------------
IRIS_URL = ML_CFG.get_item(key="IRIS_URL")

# --------------------------------------------------------------------------------------
# Main Controller
# --------------------------------------------------------------------------------------
def main():
    """
    Main controller function for the application.

    :return: None.
    """

    LOGGER.debug(msg="main started")

    try:
        LOGGER.info(msg="Read IRIS Dataset Started")
        df = pd.read_csv(filepath_or_buffer=IRIS_URL,
                         header=None,
                         encoding='utf-8')
        plot(df_iris=df)
    except Exception as e:
        LOGGER.error(msg="Read IRIS Dataset failed: {0}".format(e))
    else:
        LOGGER.info(msg="Read IRIS Dataset Complete")

    LOGGER.debug(msg="main ended")


# --------------------------------------------------------------------------------------
# Plot sepal length v petal length for Iris-setosa and Iris-versicolour.
# --------------------------------------------------------------------------------------
def plot(df_iris: pd.DataFrame):
    """
    Generate a scatter plot of sepal length v petal length.

    :param df_iris: (Dataframe) The iris dataset.
    :return: None
    """

    LOGGER.debug(msg="plot started")

    # Select setosa and versicolour records.
    y = df_iris.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)

    # Extract sepal length and petal length.
    X = df_iris.iloc[0:100, [0, 2]].values

    # Plot data.
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x',
                label='versicolor')
    plt.xlabel('sepal length (cm)')
    plt.ylabel('petal length (cm)')
    plt.legend(loc='upper left')
    
    # Output plot to file.
    filename = OUTPUT_FILEPATH + 'plot1.png'
    plt.savefig(filename)

    # Create perceptron.
    LOGGER.info(msg="Creating Perceptron")
    ppn = mlp.Perceptron(eta=0.01, n_iter=10)
    ppn.fit(X=X, y=y)
    plt.close()
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Number of Updates')
    filename = OUTPUT_FILEPATH + 'plot2.png'
    plt.savefig(filename)
    LOGGER.info(msg="Perceptron Complete")

    LOGGER.debug(msg="plot ended")


# --------------------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    LOGGER.info(msg="*" * 80)
    LOGGER.info(msg="Perceptron")
    LOGGER.info(msg="Major Version Name: {}".format(MAJOR_VERSION_NAME))
    LOGGER.info(msg="Version: {}".format(VERSION))
    LOGGER.info(msg="*" * 80)
    LOGGER.info(msg="Execution Mode: Local = {}".format(LOCAL))
    main()
