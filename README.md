# 🐧 Penguin Species Classification

A machine learning project that classifies penguin species (**Adelie, Chinstrap, Gentoo**) from physical measurements using **K-Nearest Neighbors (KNN)** and **Decision Tree** models.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📊 Dataset

Palmer Penguins dataset (`data/penguins_lter.csv`), with **344 records** collected at Palmer Station, Antarctica.

| Item | Details |
|------|---------|
| **Features used** | Culmen Length (mm), Culmen Depth (mm), Flipper Length (mm), Body Mass (g) |
| **Target** | Species (Adelie, Chinstrap, Gentoo) |
| **Missing values** | Handled with median imputation inside the model pipeline |

## 🔄 Workflow

1. Data loading and understanding
2. Exploratory data analysis (species distribution, feature histograms, pair plot, correlation heatmap)
3. Preprocessing (median imputation, standard scaling, stratified 80/20 train-test split)
4. Model training: KNN (k=5) and Decision Tree
5. Evaluation: accuracy, precision, recall, F1-score, confusion matrices
6. 5-fold cross-validation and final model comparison

## 🏆 Results

| Model | Test Accuracy | Precision | Recall | F1 Score | CV Mean Accuracy | CV Std Dev |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| **KNN** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9855 | 0.0092 |
| Decision Tree | 0.9855 | 0.9861 | 0.9855 | 0.9855 | 0.9621 | 0.0220 |

**KNN performed best overall.** It also had a lower standard deviation across the cross-validation folds, which means more consistent performance.

> **Note:** The test set has only 69 samples, so a perfect 100% test score should be read with care. The cross-validated accuracy (98.55% for KNN) is the more reliable estimate of real-world performance.

## 📁 Project Structure

```
penguin_classification/
├── data/
│   └── penguins_lter.csv
├── Penguin_Classification.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/ashk-k10/penguin_classification.git
cd penguin_classification

# 2. (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook
```

Open `Penguin_Classification.ipynb` and choose **Kernel → Restart & Run All**.

## 🛠️ Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Jupyter Notebook

## 🙏 Acknowledgements

Data were collected and made available by Dr. Kristen Gorman and the Palmer Station Long Term Ecological Research (LTER) Program, part of the US LTER Network.

## 👤 Author

**ashk-k10**
GitHub: [@ashk-k10](https://github.com/ashk-k10)

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
