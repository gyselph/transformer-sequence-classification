# Simple demo of a transformer encoder implementation in TensorFlow

This project demonstrates how to use a transformer encoder to do multivariate sequence classification.

For this demonstration, a public dataset from Kaggle is used: [CareerCon 2019 - Help Navigate Robots](https://www.kaggle.com/c/career-con-2019/data)

## Run demo

Perfom these steps:
- Make sure you can run Python and install all dependencies from the Python scripts, including TensorFlow.
- Download the public sequence dataset (https://www.kaggle.com/c/career-con-2019/data) and store in folder `./career-con-2019`.
- Run `create_dataset.py` to create the numpy dataset `./dataset_career_con_2019.npz`, which will serve as training data for the transformer model.
- Optional: Run `analyze_dataset.ipynb` to do some dataset analysis.
- Run `transformer_classifier.py` to train a transformer encoder on sequence classification

## Transformer encoder

(Coming soon...)