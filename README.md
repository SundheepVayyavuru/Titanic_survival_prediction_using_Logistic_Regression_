# Titanic_survival_prediction_using_Logistic_Regression_
Predict Titanic passenger survival using logistic regression in Python. This project covers data cleaning, feature engineering, model training, and evaluation with scikit-learn. Ideal for beginners, it demonstrates core ML concepts and a clean, modular project structure for reproducibility.
🚢 Titanic Survival Prediction using Logistic Regression
This project predicts the survival of passengers aboard the Titanic using a logistic regression model. It demonstrates a complete machine learning pipeline — from data preprocessing to model evaluation and visualization.
📁 Project Structure
├── titanic.csv                  # Dataset file
├── titanic_confusion.png       # Confusion matrix heatmap
├── titanic_results.md          # Exported model evaluation metrics
├── titanic_survival.py         # Main Python script
├── README.md                   # Project documentation
├── requirements.txt            # Python dependencies
📊 Dataset
The dataset used is the Titanic dataset, which includes information about passengers such as:
- Features: Sex, Age, Fare, Pclass, Embarked
- Target: Survived (0 = No, 1 = Yes)
🧹 Data Preprocessing
- Selected relevant features for modeling.
- Dropped rows with missing values using dropna().
- Split the dataset into training and testing sets (80/20 split).
🧠 Model Pipeline
- Numerical Features: Age, Fare scaled using StandardScaler.
- Categorical Features: Sex, Pclass, Embarked encoded using OneHotEncoder.
- Model: LogisticRegression with max_iter=1000.
- Pipeline: Combined preprocessing and modeling using Pipeline and ColumnTransformer.
📈 Evaluation Metrics
After training, the model is evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score
The results are saved as a Markdown table in titanic_results.md.
📊 Confusion Matrix
A heatmap of the confusion matrix is generated and saved as titanic_confusion.png.
🛠️ How to Run
- Clone the repository:
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
- Install dependencies:
pip install -r requirements.txt
- Place the dataset:
- Download titanic.csv from Kaggle
- Save it in the project directory or update the path in the script.
- Run the script:
python titanic_survival.py


📦 Dependencies
Add the following to your requirements.txt:
pandas
numpy
matplotlib
seaborn
scikit-learn


🙌 Acknowledgments
- Titanic: Machine Learning from Disaster
- scikit-learn, pandas, seaborn, matplotlib



