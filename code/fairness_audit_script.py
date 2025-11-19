"""
AI Ethics Assignment - Fairness Audit Script
This script analyzes the COMPAS dataset for racial bias in risk assessment scores.
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import ClassificationMetric
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set up visualization style
try:
    plt.style.use('seaborn-v0_8')  # Updated style name for newer matplotlib
    sns.set_theme(style="whitegrid")
except:
    # Fallback to basic styling if seaborn style is not available
    plt.style.use('default')
    sns.set_theme()
    
# Set color palette
sns.set_palette("colorblind")

# 1. Load and preprocess the COMPAS dataset
def load_compas_data():
    """Load and preprocess the COMPAS dataset."""
    # Note: In a real scenario, you would load the actual COMPAS dataset
    # For this example, we'll create a simplified version
    
    # This is a placeholder - in practice, you would load the actual COMPAS data
    # df = pd.read_csv('compas-scores-two-years.csv')
    
    # For demonstration, create a synthetic dataset
    np.random.seed(42)
    n_samples = 5000
    
    data = {
        'age': np.random.normal(30, 10, n_samples).astype(int),
        'race': np.random.choice(['Caucasian', 'African-American', 'Hispanic', 'Other'], 
                               size=n_samples, p=[0.4, 0.3, 0.2, 0.1]),
        'sex': np.random.choice(['Male', 'Female'], size=n_samples, p=[0.7, 0.3]),
        'priors_count': np.random.poisson(2, n_samples),
        'decile_score': np.random.randint(1, 11, n_samples),
        'two_year_recid': np.random.binomial(1, 0.5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create binary race column (sensitive attribute)
    df['race_binary'] = df['race'].apply(lambda x: 1 if x == 'African-American' else 0)
    
    return df

# 2. Prepare the dataset for fairness analysis
def prepare_dataset(df):
    """Prepare the dataset for fairness analysis."""
    # Select features and target
    features = ['age', 'priors_count', 'race_binary']
    target = 'two_year_recid'
    
    # Split into features and target
    X = df[features]
    y = df[target]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=df['race_binary']
    )
    
    # Create AIF360 dataset
    train_data = BinaryLabelDataset(
        df=pd.concat([X_train, y_train], axis=1),
        label_names=['two_year_recid'],
        protected_attribute_names=['race_binary'],
        favorable_label=0,
        unfavorable_label=1
    )
    
    test_data = BinaryLabelDataset(
        df=pd.concat([X_test, y_test], axis=1),
        label_names=['two_year_recid'],
        protected_attribute_names=['race_binary'],
        favorable_label=0,
        unfavorable_label=1
    )
    
    return train_data, test_data, X_train, X_test, y_train, y_test

# 3. Train a baseline model
def train_baseline_model(X_train, y_train):
    """Train a baseline logistic regression model."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

# 4. Evaluate model fairness
def evaluate_fairness(model, X_test, y_test, scaler, protected_attribute):
    """Evaluate model fairness using AIF360 metrics."""
    # Scale test features
    X_test_scaled = scaler.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Create dataset for fairness metrics
    dataset = BinaryLabelDataset(
        df=pd.concat([X_test, y_test, pd.Series(y_pred, name='prediction')], axis=1),
        label_names=['two_year_recid'],
        protected_attribute_names=['race_binary'],
        favorable_label=0,
        unfavorable_label=1
    )
    
    # Calculate fairness metrics
    privileged_groups = [{'race_binary': 0}]
    unprivileged_groups = [{'race_binary': 1}]
    
    metric = ClassificationMetric(
        dataset,
        dataset,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups
    )
    
    # Calculate fairness metrics
    metrics = {
        'statistical_parity_difference': metric.statistical_parity_difference(),
        'disparate_impact': metric.disparate_impact(),
        'equal_opportunity_difference': metric.equal_opportunity_difference(),
        'average_odds_difference': metric.average_odds_difference()
    }
    
    return metrics, y_pred

# 5. Mitigate bias using reweighing
def mitigate_bias(train_data, test_data):
    """Apply reweighing to mitigate bias in the training data."""
    # Apply reweighing
    RW = Reweighing(
        unprivileged_groups=[{'race_binary': 1}],
        privileged_groups=[{'race_binary': 0}]
    )
    
    # Transform the dataset
    dataset_transf_train = RW.fit_transform(train_data)
    
    return dataset_transf_train

# 6. Visualize results
def plot_fairness_metrics(metrics_before, metrics_after):
    """Visualize fairness metrics before and after mitigation."""
    metrics = ['Statistical Parity', 'Disparate Impact', 'Equal Opportunity', 'Average Odds']
    before = [
        metrics_before['statistical_parity_difference'],
        metrics_before['disparate_impact'],
        metrics_before['equal_opportunity_difference'],
        metrics_before['average_odds_difference']
    ]
    
    after = [
        metrics_after['statistical_parity_difference'],
        metrics_after['disparate_impact'],
        metrics_after['equal_opportunity_difference'],
        metrics_after['average_odds_difference']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, before, width, label='Before Mitigation')
    rects2 = ax.bar(x + width/2, after, width, label='After Mitigation')
    
    ax.set_ylabel('Metric Value')
    ax.set_title('Fairness Metrics Before and After Mitigation')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    
    fig.tight_layout()
    plt.savefig('fairness_metrics_comparison.png')
    plt.show()

def plot_confusion_matrices(y_test, y_pred_before, y_pred_after, protected_attribute):
    """Plot confusion matrices before and after mitigation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Before mitigation
    cm_before = confusion_matrix(y_test, y_pred_before)
    sns.heatmap(cm_before, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix - Before Mitigation')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # After mitigation
    cm_after = confusion_matrix(y_test, y_pred_after)
    sns.heatmap(cm_after, annot=True, fmt='d', cmap='Blues', ax=ax2)
    ax2.set_title('Confusion Matrix - After Mitigation')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.show()

# Main execution
def main():
    print("Starting Fairness Audit...")
    
    # 1. Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_compas_data()
    
    # 2. Prepare dataset for fairness analysis
    print("Preparing dataset for fairness analysis...")
    train_data, test_data, X_train, X_test, y_train, y_test = prepare_dataset(df)
    
    # 3. Train baseline model
    print("Training baseline model...")
    model, scaler = train_baseline_model(X_train, y_train)
    
    # 4. Evaluate fairness before mitigation
    print("Evaluating fairness before mitigation...")
    metrics_before, y_pred_before = evaluate_fairness(model, X_test, y_test, scaler, 'race_binary')
    
    # 5. Apply bias mitigation
    print("Applying bias mitigation...")
    dataset_transf_train = mitigate_bias(train_data, test_data)
    
    # 6. Train model on transformed data
    print("Training model on transformed data...")
    # Extract features and labels from transformed dataset
    X_train_transf = dataset_transf_train.features
    y_train_transf = dataset_transf_train.labels.ravel()
    
    # Scale features
    scaler_transf = StandardScaler()
    X_train_transf_scaled = scaler_transf.fit_transform(X_train_transf)
    
    # Train model on transformed data
    model_transf = LogisticRegression(random_state=42)
    model_transf.fit(X_train_transf_scaled, y_train_transf)
    
    # 7. Evaluate fairness after mitigation
    print("Evaluating fairness after mitigation...")
    metrics_after, y_pred_after = evaluate_fairness(
        model_transf, X_test, y_test, scaler_transf, 'race_binary'
    )
    
    # 8. Print results
    print("\nFairness Metrics:")
    print("\nBefore Mitigation:")
    for metric, value in metrics_before.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nAfter Mitigation:")
    for metric, value in metrics_after.items():
        print(f"{metric}: {value:.4f}")
    
    # 9. Plot results
    print("\nGenerating visualizations...")
    plot_fairness_metrics(metrics_before, metrics_after)
    plot_confusion_matrices(y_test, y_pred_before, y_pred_after, 'race_binary')
    
    print("\nFairness audit completed!")

if __name__ == "__main__":
    main()
