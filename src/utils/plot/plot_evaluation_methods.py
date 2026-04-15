import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

def plot_result(y_pred, y_test):
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, color='teal')

    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', lw=2)
    
    plt.xlabel('Actual Market Value')
    plt.ylabel('Predicted Market Value')
    plt.title(f'Actual vs Predicted (R2: {r2_score(y_test, y_pred):.3f})')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

def plot_log_result(y_pred, y_test):
    plt.figure(figsize=(10, 6))
    
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.4, color='teal')
    
    plt.xscale('log')
    plt.yscale('log')
    
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', lw=2)
    
    plt.xlabel('Actual Market Value (Log Scale)')
    plt.ylabel('Predicted Market Value (Log Scale)')
    plt.title('Actual vs Predicted (Log Transformed)')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()

def plot_joint_result(y_pred, y_test):
    grid = sns.jointplot(x=y_test, y=y_pred, kind='reg', 
                         scatter_kws={'alpha':0.3}, 
                         line_kws={'color':'red'},
                         color='teal')
    
    grid.set_axis_labels('Actual Market Value', 'Predicted Market Value')
    grid.figure.suptitle('Joint Plot of Actual and Predicted Values', y=1.02)
    plt.show()


def predict_and_plot(model, X_test, y_test):
    y_pred = model.predict(X_test)
    plot_result(y_pred, y_test)
    plot_joint_result(y_pred, y_test)

def plot_comparison_boxplot(y_pred, y_test, figsize=(10, 6)):
    df = pd.DataFrame({
        'Value': np.concatenate([y_pred, y_test]),
        'Type': ['Predictions'] * len(y_pred) + ['Actual'] * len(y_test)
    })

    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df,
        x='Value',
        y='Type',
        ax=ax,
        hue='Type',
        palette={'Predictions': 'green', 'Actual': 'teal'},
        flierprops={
            "marker": "o",
            "markersize": 4,
            "markerfacecolor": "red",
            "markeredgecolor": "none",
            "alpha": 0.5,
        },
        linewidth=1.5,
        legend=False,
    )

    metrics = {
        'Actual': {'val': np.array(y_test), 'color': 'teal'},
        'Predictions': {'val': np.array(y_pred), 'color': 'green'}
    }

    for i, (label, info) in enumerate(metrics.items()):
        d_mean = info['val'].mean()
        ax.axvline(d_mean, color=info['color'], linestyle="--", linewidth=2, 
                   label=f"{label} Mean: {d_mean:.2f}")

    ax.set_title("Actual vs Prediction Distribution Comparison", fontsize=14, pad=15)
    ax.set_xlabel("Value")
    ax.set_ylabel("")
    ax.legend(fontsize=9, loc="upper right")
    
    plt.tight_layout()
    plt.show()

def plot_distribution_compare(y_pred, y_test):
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(y_test, label='Actual (y_test)', fill=True, alpha=0.3, color='skyblue')
    sns.kdeplot(y_pred, label='Predicted (y_pred)', fill=True, alpha=0.3, color='green')
    
    plt.xlabel('Market Value')
    plt.title('Comparison of Distributions: Actual vs Predicted')
    plt.legend()
    plt.show()

def plot_xgb_importance(model):
    xgb.plot_importance(model, importance_type='gain', height=0.5, values_format="{v:.2f}", xlim=(0, 40), grid=False)
    plt.title('Feature Importance')
    plt.show()

def plot_rf_importance(model, X_train):
    feature_names = X_train.columns
    importances = model.feature_importances_

    df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    df = df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x='Importance', y='Feature', hue='Feature', legend=False, data=df, palette='crest')

    for p in ax.patches:
        width = p.get_width() 
        ax.text(width + 0.005,
                p.get_y() + p.get_height()/2,
                f'{width:.2f}',
                va='center')

    plt.title('Feature Importances', fontsize=15)
    plt.xlabel('Importance Score')
    plt.ylabel('Features')

    plt.xlim(0, df['Importance'].max() * 1.1) 
    plt.tight_layout()
    plt.show()

def plot_error_distribution(y_pred, y_test):
    errors = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(errors, kde=True, bins=30, color='teal')
    plt.xlim(-5, 5)
    plt.ylim(0, 120)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'Error Distribution (MAE: {mean_absolute_error(y_test, y_pred):.3f}, MAPE: {(mean_absolute_percentage_error(y_test, y_pred) * 100):.3f}%)')
    plt.xlabel('Prediction Error')
    plt.show()

def plot_residual(y_pred, y_test, sigma_threshold=1.5):
    residuals = y_test - y_pred
    res_std = residuals.std()
    res_mean = residuals.mean()
    
    z_scores = (residuals - res_mean) / res_std
    
    upper_limit = res_mean + (sigma_threshold * res_std)
    lower_limit = res_mean - (sigma_threshold * res_std)
    
    conditions = [
        (z_scores > sigma_threshold),
        (z_scores < -sigma_threshold)
    ]
    choices = ['Undervalued (Hidden Gem)', 'Overvalued (Premium/Hyped)']
    status_labels = np.select(conditions, choices, default='Normal Range')
    
    temp_df = pd.DataFrame({
        'Actual': y_test,
        'Residuals': residuals,
        'Status': status_labels
    })
    
    counts = temp_df['Status'].value_counts()
    label_map = {
        'Undervalued (Hidden Gem)': f'Undervalued (>{sigma_threshold}σ, n={counts.get("Undervalued (Hidden Gem)", 0)})',
        'Overvalued (Premium/Hyped)': f'Overvalued (< -{sigma_threshold}σ, n={counts.get("Overvalued (Premium/Hyped)", 0)})',
        'Normal Range': f'Normal (n={counts.get("Normal Range", 0)})'
    }
    temp_df['Status_Label'] = temp_df['Status'].map(label_map)
    
    plt.figure(figsize=(13, 8))
    
    palette = {
        label_map['Undervalued (Hidden Gem)']: '#2E86C1',
        label_map['Overvalued (Premium/Hyped)']: '#CB4335',
        label_map['Normal Range']: '#D5DBDB'
    }
    
    sns.scatterplot(data=temp_df, x='Actual', y='Residuals', hue='Status_Label', 
                    palette=palette, alpha=0.8, s=80, edgecolor='black', linewidth=0.5)
    
    plt.axhline(y=res_mean, color='black', linestyle='-', linewidth=1, label='Error Mean')
    plt.axhline(y=upper_limit, color='#2E86C1', linestyle='--', linewidth=1.5)
    plt.axhline(y=lower_limit, color='#CB4335', linestyle='--', linewidth=1.5)
    
    plt.axhspan(upper_limit, plt.ylim()[1], color='#2E86C1', alpha=0.05)
    plt.axhspan(plt.ylim()[0], lower_limit, color='#CB4335', alpha=0.05)
    
    plt.title(f'Market Valuation: Classification ({sigma_threshold}σ)', fontsize=16)
    plt.xlabel('Actual Market Value')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Statistical Status")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()