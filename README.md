# Customer Churn Prediction

A machine learning project that predicts customer churn using deep learning techniques. This project analyzes customer data to identify patterns and predict which customers are likely to leave the service.

## 📊 Dataset Overview

The dataset contains information about 10,000 customers with the following features:

- **CustomerId**: Unique identifier for each customer
- **CreditScore**: Customer's credit score
- **Geography**: Customer's location (France, Spain, Germany)
- **Gender**: Customer's gender
- **Age**: Customer's age
- **Tenure**: Number of years as a customer
- **Balance**: Account balance
- **NumOfProducts**: Number of products the customer has
- **HasCrCard**: Whether the customer has a credit card (1/0)
- **IsActiveMember**: Whether the customer is an active member (1/0)
- **EstimatedSalary**: Customer's estimated salary
- **Exited**: Target variable indicating if customer churned (1/0)

## 🚀 Features

- **Data Preprocessing**: Handles categorical variables using one-hot encoding
- **Feature Scaling**: Standardizes features using StandardScaler
- **Deep Learning Model**: Uses TensorFlow/Keras with a neural network architecture
- **Model Evaluation**: Includes accuracy metrics and training visualization
- **Interactive Notebook**: Jupyter notebook with step-by-step analysis

## 🛠️ Technologies Used

- **Python 3.x**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities
- **TensorFlow/Keras**: Deep learning framework
- **Matplotlib**: Data visualization

## 📦 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd customer-churn-prediction
```

2. Install required packages:
```bash
pip install pandas numpy scikit-learn tensorflow matplotlib jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

## 🎯 Usage

1. Open `customer-churn-prediction.ipynb` in Jupyter Notebook
2. Run all cells sequentially to:
   - Load and explore the dataset
   - Preprocess the data
   - Build and train the neural network model
   - Evaluate model performance
   - Visualize training results

## 🏗️ Model Architecture

The neural network model consists of:
- **Input Layer**: 11 features (after preprocessing)
- **Hidden Layer**: 11 neurons with sigmoid activation
- **Output Layer**: 1 neuron with sigmoid activation for binary classification

**Training Configuration:**
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Batch Size: 50
- Epochs: 100
- Validation Split: 20%

## 📈 Results

The model achieves good performance in predicting customer churn:
- Training accuracy improves over epochs
- Validation accuracy stabilizes around 79-80%
- The model learns to distinguish between churning and non-churning customers

## 📁 Project Structure

```
customer-churn-prediction/
├── README.md                           # Project documentation
├── customer-churn-prediction.ipynb     # Main analysis notebook
├── Churn_Modelling.csv                 # Dataset
└── project_summary.txt                 # Project summary
```

## 🔍 Key Insights

- **Data Quality**: Clean dataset with no missing values
- **Feature Engineering**: Categorical variables properly encoded
- **Model Performance**: Neural network shows good predictive capability
- **Scalability**: Model can be easily retrained with new data

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author
**Shazim Javed**
- GitHub: [Shazim Javed](https://github.com/shazimjaved)
- LinkedIn: [Shazim Javed](https://linkedin.com/in/shazimjaved)

## 🙏 Acknowledgments

- Dataset source: Credit Card Customer Churn Prediction
- Kaggle community for inspiration and resources
- TensorFlow team for the excellent deep learning framework

---

⭐ If you found this project helpful, please give it a star!


