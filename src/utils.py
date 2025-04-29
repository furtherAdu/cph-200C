import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    for col in cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        del df[col]
    return df


# class to handle scaling of differemt feature types
class Columns(BaseEstimator, TransformerMixin):
    def __init__(self, names=None):
        self.names = names

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X):
        return X[self.names]

# class to transform sparse matrix to dense
class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.toarray()

class PreFittedEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoder):
        self.encoder = encoder

    def fit(self, X, y=None, **fit_params):
        # self.is_fitted_ = True
        # self._is_fitted = True
        return self

    def transform(self, X):
        return self.encoder.transform(X)

def plot_points_with_ci(df, x_col='x', y_col='y', se_col='se', title='Points with Confidence Intervals', x_label='X Axis', y_label='Y Axis', output_path='plot.png'):
    """
    Plots points with confidence intervals using Seaborn and Times New Roman font,
    with a grid and a gray dashed line for y = 0.

    Args:
        df (pd.DataFrame): DataFrame containing x, y, and standard error columns.
        x_col (str): Name of the x-value column.
        y_col (str): Name of the y-value column.
        se_col (str): Name of the standard error column.
        title (str): Title of the plot.
        x_label (str): Label for the x-axis.
        y_label (str): Label for the y-axis.
        output_path (str): Path to save the plot.
    """

    # Calculate confidence interval bounds (assuming 95% CI)
    df['lower'] = df[y_col] - 1.96 * df[se_col]
    df['upper'] = df[y_col] + 1.96 * df[se_col]

    # # Set Times New Roman font for all text
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']   

    # Create the plot
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    sns.scatterplot(x=x_col, y=y_col, data=df, color='black') # plot the points
    plt.errorbar(x=df[x_col], y=df[y_col], yerr=1.96 * df[se_col], fmt='none', color='black', capsize=5) # plot the error bars.

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.5)

    # Add gray dashed line for y = 0
    plt.axhline(y=0, color='gray', linestyle='--')

    # Set labels and title
    plt.title(title, fontsize=16)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()