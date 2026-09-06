# Wine Quality Prediction

A notebook-based machine-learning project that classifies red wines as **good quality** or **not good quality** from physicochemical measurements. The workflow explores the data, visualizes relationships between variables, trains a Random Forest classifier, and makes a prediction for a single wine sample.

> **Classification rule:** a wine is labelled `1` (**good**) when its recorded quality score is **7 or higher**; otherwise it is labelled `0` (**not good**).

## Project contents

```text
.
├── sample_data/
│   └── winequality-red.csv       # Red-wine measurements and quality scores
├── wine_quality_prediction.ipynb # Analysis, training, evaluation, and prediction workflow
└── README.md
```

## Dataset

The included dataset contains **1,599 red-wine samples**. Each row has 11 numerical input features and one `quality` score:

| Feature | Description |
| --- | --- |
| `fixed acidity` | Fixed-acid concentration |
| `volatile acidity` | Volatile-acid concentration |
| `citric acid` | Citric-acid concentration |
| `residual sugar` | Residual sugar concentration |
| `chlorides` | Chloride concentration |
| `free sulfur dioxide` | Free sulfur dioxide level |
| `total sulfur dioxide` | Total sulfur dioxide level |
| `density` | Density of the wine |
| `pH` | Acidity/alkalinity measure |
| `sulphates` | Sulphate concentration |
| `alcohol` | Alcohol content |
| `quality` | Original sensory quality score; used to create the binary target |

The CSV is comma-delimited and is loaded directly from `sample_data/winequality-red.csv` by the notebook.

## Workflow

The notebook performs the following steps:

1. Loads the dataset into a pandas DataFrame and checks its shape, preview, and missing values.
2. Examines summary statistics, quality counts, selected feature/quality bar charts, and a correlation heatmap.
3. Separates the 11 input features from `quality` and converts scores to the binary target described above.
4. Creates an 80/20 train/test split using `random_state=2`.
5. Trains a `RandomForestClassifier`.
6. Calculates training accuracy and predicts the class for a supplied 11-value sample.

## Requirements

- Python 3.11 or later
- Jupyter Notebook or JupyterLab
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Getting started

1. Clone the repository and move into it:

   ```bash
   git clone <repository-url>
   cd wine-quality-prediction-
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell, run `.venv\\Scripts\\Activate.ps1` instead.

3. Install the notebook dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install jupyter numpy pandas matplotlib seaborn scikit-learn
   ```

4. Start Jupyter from the repository root:

   ```bash
   jupyter notebook
   ```

5. Open `wine_quality_prediction.ipynb` and run the cells in order. Running Jupyter from the repository root is important because the notebook uses the relative dataset path `./sample_data/winequality-red.csv`.

## Making a prediction

The model expects values in this exact order:

```text
fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,
free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol
```

For example, the notebook predicts from this sample:

```python
input_data = (7.4, 0.7, 0.0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4)
```

A prediction of `1` means the model classifies the wine as good quality; `0` means it does not meet the project's good-quality threshold.

## Notes and limitations

- This is a binary classification exercise, not a predictor of the original multi-class quality score.
- The model is created without a fixed `random_state`, so results can vary between runs.
- The target is imbalanced because wines with scores of 7 or above are less common; accuracy alone may not fully describe model performance.
- The test-accuracy cell currently assigns the result to `t_data_accuracy` but prints `test_data_accuracy`. Change the final line to `print(t_data_accuracy)` before running that cell, or use the clearer version below:

  ```python
  test_predictions = model.predict(X_test)
  test_accuracy = accuracy_score(Y_test, test_predictions)
  print(test_accuracy)
  ```

## Future improvements

- Set a model seed and report repeatable evaluation metrics.
- Add precision, recall, F1 score, ROC-AUC, and a confusion matrix.
- Address class imbalance with class weighting or resampling.
- Tune Random Forest hyperparameters with cross-validation.
- Save the trained model and provide a small command-line or web prediction interface.
