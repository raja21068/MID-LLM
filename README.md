# MID-LLM-Enhancing-Medical-Image-Diagnostics-with-LLMs-in-a-Decentralized-AI-Framework
## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Running the Project](#running-the-project)
- [Federated Learning Overview](#federated-learning-overview)
- [IPFS Integration](#ipfs-integration)
- [OpenAI API Integration](#openai-api-integration)
- [License](#license)

## Introduction

This project combines machine learning, federated learning, and blockchain technologies to create a decentralized approach to brain tumor detection. After training a model using federated learning across multiple simulated clients, the model is used to classify MRI images of the brain. The classification result is then used to generate a detailed report using the OpenAI API.

## Features

- Federated Learning with multiple clients (hospitals) and a central aggregator (research center).
- Integration with IPFS for decentralized storage and retrieval of model parameters.
- Use of OpenAI's GPT model to generate detailed reports based on classification results.
- Visualization of 3D brain MRI data.

## Project Structure
llm/
│
(`│`)config.py # Configuration settings and seeding
(`│`) ipfs_blockchain.py # IPFS and Blockchain integration
(`│`) utils.py # Utility functions and classes for image processing and metrics
(`│`) dataset.py # Dataset class for loading and preprocessing data
(`│`) models.py # Model architectures for brain tumor detection
(`│`)client.py # Client (Hospital) class for local training and parameter sharing
(`│`)aggregator.py # Aggregator (Research Center) class for parameter aggregation
├── inference.py # Inference and report generation using the global model
├── train.py # Main script for training and federated learning simulation
├── requirements.txt # Python dependencies
└── README.md # Project overview and instructions

## Requirements

- Python 3.7 or later
- PyTorch
- torchvision
- numpy
- pandas
- scikit-learn
- nibabel
- requests
- tqdm
- imageio
- matplotlib
- seaborn
- IPFS (installed and running locally)
- OpenAI API key

## Installation

1. **Clone the repository**:

    ```bash
    git clone https://github.com/raja21068/MID-LLM.git
    cd MID-LLM
    ```

2. **Install the required Python packages**:

    ```bash
    pip install -r requirements.txt
    ```

3. **Set up IPFS**:

    - Install IPFS from [https://ipfs.io/](https://ipfs.io/).
    - Start the IPFS daemon:

    ```bash
    ipfs daemon
    ```

4. **Set up OpenAI API**:

    - Create an account at [OpenAI](https://www.openai.com/).
    - Obtain an API key and add it to your environment or directly in the code.

## Data Preparation

1. **Download the dataset**:
    - The code is designed to work with the BRATS 2020 dataset. Ensure that you have downloaded this dataset and placed it in the appropriate directories.

2. **Update paths**:
    - Ensure that the paths in the `GlobalConfig` class match the location of your dataset on your filesystem.

3. **Preprocess the data**:
    - Follow the instructions or code provided to preprocess the data, if necessary.

## Running the Project

1. **Run the Federated Learning Script**:

    The main script simulates federated learning by training models on different clients (hospitals), sharing their parameters via IPFS, and then aggregating them at a central location.

    ```bash
    python main.py
    ```

2. **Predict and Generate a Report**:

    After the global model is aggregated, use it to predict on new MRI images and generate a report:

    ```bash
    python predict_and_generate_report.py
    ```

3. **View the Report**:

    The report will be saved in the current directory as `classification_report.txt`.

## Federated Learning Overview

The project simulates a federated learning environment where:
- Multiple clients (hospitals) train local models on their data.
- Model parameters are uploaded to IPFS.
- A central research center aggregates these parameters using a simple averaging strategy.

## IPFS Integration

The project uses IPFS to:
- Store and retrieve model parameters.
- Decentralize the storage of data and models.
  
Ensure that the IPFS daemon is running before executing the script.

## OpenAI API Integration

The project integrates with OpenAI's GPT model to generate reports based on the classification results of MRI images. Ensure you have your API key configured.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
