---

This project is focused on predicting **medical insurance costs** using a **Linear Regression model**. The model is trained on a dataset (`insurance.csv`) that includes various features such as age, sex, body mass index (BMI), number of children, smoking status, and residential region. These factors are known to influence the cost of medical insurance, and by analyzing them, the model aims to estimate the expected charges for an individual.

The dataset is first loaded using `pandas`, and the categorical columns like `sex`, `smoker`, and `region` are converted into numerical format using `LabelEncoder` for compatibility with the machine learning algorithm. The data is then split into training and testing sets using an 80-20 ratio, and a `LinearRegression` model from the `scikit-learn` library is trained on the training set. After training, the model is evaluated using metrics like **Mean Squared Error (MSE)** and the **R-squared score (R²)**, which help determine the accuracy of the predictions.

To better understand how well the model performs, a scatter plot is created using `matplotlib` and `seaborn` to visualize the relationship between actual and predicted charges. The closer the points lie to the diagonal line, the better the model's predictions are.

The project also includes a custom function named `predict_insurance_cost()` that allows users to input personal information (age, sex, BMI, number of children, smoking status, and region) and receive a predicted insurance charge based on the trained model. This makes the model interactive and practical for real-world use cases.

This project demonstrates the complete machine learning workflow—from data preprocessing to model training, evaluation, and prediction—using Python. Potential future improvements include using more advanced regression models, hyperparameter tuning, enhanced feature engineering, and building a user-friendly web interface using Flask or Streamlit for deployment.# Health-Insurance-Cost-Prediction
