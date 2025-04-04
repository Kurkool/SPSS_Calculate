import pyspssio
import pingouin as pg
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

def conbrach_alpha(data, columns):
    """
    Calculate Cronbach's alpha for the given columns in the DataFrame.

    :param data: DataFrame containing the data
    :param columns: List of column names to include in the calculation
    :return: Cronbach's alpha value
    """
    return pg.cronbach_alpha(data[columns])[0]

def regression_analysis(data, dependent_var, independent_vars):
    """
    Perform regression analysis on the given data.

    :param data: DataFrame containing the data
    :param dependent_var: Name of the dependent variable
    :param independent_vars: List of independent variable names
    :return: Regression model summary
    """
    # Drop rows with missing values
    data = data.dropna(subset=[dependent_var] + independent_vars)

    # Create the formula for the regression model
    formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"

    # Fit the regression model using the formula
    model = smf.ols(formula, data=data).fit()

    # Get the ANOVA table
    anova_results = anova_lm(model)

    return model.summary(), anova_results

if __name__ == '__main__':
    df, meta = pyspssio.read_sav("cow4cash_raw_mean.sav")
    print(df.columns)

    trust_columns = [col for col in df.columns if 'trust' in col]
    time_columns = [col for col in df.columns if 'time' in col]
    return_columns = [col for col in df.columns if 'return' in col]
    risk_columns = [col for col in df.columns if 'risk' in col]
    intent_columns = [col for col in df.columns if 'intent' in col]

    # Calculate Cronbach's alpha of trust columns
    print(f"Cronbach's alpha of trust: {conbrach_alpha(df, trust_columns)}")
    # Calculate Cronbach's alpha of time columns
    print(f"Cronbach's alpha of time: {conbrach_alpha(df, time_columns)}")
    # Calculate Cronbach's alpha of return columns
    print(f"Cronbach's alpha of return: {conbrach_alpha(df, return_columns)}")
    # Calculate Cronbach's alpha of risk columns
    print(f"Cronbach's alpha of return: {conbrach_alpha(df, risk_columns)}")

    print("###### Regression Analysis ###")

    regression_all, anova_results_all = regression_analysis(df, intent_columns[0], ['A_TRUST', 'A_TIME', 'A_RETURN', 'A_RISK'])
    print(f"regression_all:\n {regression_all}")
    print(f"anova_results_all:\n {anova_results_all}")

    regression_trust, anova_results_trust = regression_analysis(df, intent_columns[0], ['A_TRUST'])
    print(f"regression_trust:\n {regression_trust}")
    print(f"anova_results_trust:\n {anova_results_trust}")

    regression_time, anova_results_time = regression_analysis(df, intent_columns[0], ['A_TIME'])
    print(f"regression_time:\n {regression_time}")
    print(f"anova_results_time:\n {anova_results_time}")


