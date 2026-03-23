import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def fill_missing_values(dataset_column, decimals=2):
    """Fill missing values by interpolating between adjacent known values.

    Uses the mean change between consecutive values as the default step
    for edges, and averages neighbors for interior gaps.
    """
    dataset_column = dataset_column.copy()
    usual_change = round(dataset_column.diff().dropna().mean(), decimals)

    for i in range(len(dataset_column)):
        if not pd.isna(dataset_column[i]):
            continue

        if i == 0:
            first_valid = dataset_column.first_valid_index()
            dataset_column[i] = dataset_column[first_valid] - usual_change
        elif i == len(dataset_column) - 1:
            dataset_column[i] = dataset_column[i - 1] + usual_change
            return dataset_column
        else:
            previous_value = dataset_column[i - 1]
            next_value = dataset_column[i + 1]

            if pd.isna(next_value):
                next_valid = dataset_column[i + 1 :].first_valid_index()
                if next_valid is None or pd.isna(dataset_column[next_valid]):
                    dataset_column[i] = previous_value + usual_change
                    continue
                next_value = dataset_column[next_valid]

            dataset_column[i] = round((previous_value + next_value) / 2, decimals)

    return dataset_column


def sync_data_column(sub_data, train_data, test_data):
    """Align auxiliary dataset dates with training and test data.

    Adds rows for any dates present in train/test but missing from sub_data,
    filling new rows with NaN for non-date columns.
    """
    missing_train = train_data[~train_data["date"].isin(sub_data["date"])]["date"].unique()
    missing_test = test_data[~test_data["date"].isin(sub_data["date"])]["date"].unique()

    missing_train = pd.DataFrame(missing_train, columns=["date"])
    missing_test = pd.DataFrame(missing_test, columns=["date"])

    sub_data = pd.concat([sub_data, missing_train], ignore_index=True).sort_values(by="date").reset_index(drop=True)
    sub_data = pd.concat([sub_data, missing_test], ignore_index=True).sort_values(by="date").reset_index(drop=True)

    return sub_data


def get_categorical_features(df, threshold=35):
    """Return non-numeric columns with unique values <= threshold."""
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    return [col for col in categorical_columns if df[col].nunique() <= threshold]


def get_numerical_features(df, threshold=35):
    """Return numeric columns with unique values >= threshold."""
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    return [col for col in numerical_columns if df[col].nunique() >= threshold]


def add_wage_day(df):
    """Add a WageDay boolean feature (1 if 15th or last day of month)."""
    def is_last_day(date):
        next_day = date + pd.Timedelta(days=1)
        return next_day.month != date.month

    is_15th = df["date"].dt.day == 15
    is_last = df["date"].apply(is_last_day)
    df["WageDay"] = (is_15th | is_last).astype(int)
    return df


def extract_date(df, date_col):
    """Extract 10 temporal features from a datetime column."""
    df[f"year {date_col}"] = df[date_col].dt.year.astype(int)
    df[f"month {date_col}"] = df[date_col].dt.month.astype(int)
    df[f"day {date_col}"] = df[date_col].dt.day.astype(int)
    df[f"day of week {date_col}"] = df[date_col].dt.dayofweek.astype(int)
    df[f"week of year {date_col}"] = df[date_col].dt.isocalendar().week.astype(int)
    df[f"quarter {date_col}"] = df[date_col].dt.quarter.astype(int)
    df[f"is weekend {date_col}"] = df[f"day of week {date_col}"].isin([5, 6]).astype(int)
    df[f"is leap year {date_col}"] = df[date_col].dt.is_leap_year.astype(int)
    df[f"is_month_end {date_col}"] = df[date_col].dt.is_month_end.astype(int)
    df[f"is_month_start {date_col}"] = df[date_col].dt.is_month_start.astype(int)


def skewness_analyze(df, skew_threshold=0.5):
    """Return list of numeric columns with |skewness| above the threshold."""
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    numerical_columns = [
        col for col in numerical_columns
        if len(df[col].unique()) > 2 and sorted(df[col].unique()) != [0, 1]
    ]

    skewed = [col for col in numerical_columns if abs(df[col].skew()) > skew_threshold]
    print(f"Total Skewed Features: {len(skewed)}")
    return skewed


def fix_skewness(df, skewed_features):
    """Apply StandardScaler to non-skewed and log1p to skewed features."""
    non_skewed = list(set(df.columns) - set(skewed_features))
    scaler = StandardScaler()

    for col in non_skewed:
        df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    for col in skewed_features:
        negative_mask = df[col] < 0
        transformed = np.log1p(np.abs(df[col]))
        df[col] = np.where(negative_mask, -transformed, transformed)

    return df


def method_preprocessing(dataset, method, zero_frac=0.2, random_state=42):
    """Handle zero-inflated sales target.

    Methods:
        1: Do nothing
        2: Remove all zero sales
        3: Keep only zero_frac of zero-sales rows (default 20%)
    """
    if method == 1:
        return dataset
    elif method == 2:
        return dataset[dataset["sales"] > 0]
    elif method == 3:
        zero_sales = dataset[dataset["sales"] == 0]
        non_zero_sales = dataset[dataset["sales"] > 0]
        zero_sales = zero_sales.sample(frac=zero_frac, random_state=random_state)
        dataset = pd.concat([zero_sales, non_zero_sales])
        dataset = dataset.sample(frac=1, random_state=random_state).reset_index(drop=True)
        dataset = dataset.sort_values(by="date").reset_index(drop=True)
        return dataset
    else:
        raise ValueError(f"Method {method} is not implemented.")


def align_columns(train_dataset, test_data, fix_columns=False):
    """Align columns between train and test datasets."""
    train_align, test_align = train_dataset.align(test_data, join="outer", axis=1)
    test_align = test_align.fillna(0)

    if fix_columns:
        train_align = train_align[train_dataset.columns]
        test_align = test_align[train_dataset.columns]

    return train_align, test_align


def one_hot_encoding(df, columns=None, threshold=35):
    """One-hot encode categorical features, dropping the first category.

    Args:
        df: DataFrame to encode.
        columns: Explicit list of columns to encode. If None, auto-detects
                 non-numeric columns with unique values <= threshold.
        threshold: Max unique values for auto-detection.
    """
    cat_features = columns if columns is not None else get_categorical_features(df, threshold=threshold)
    if not cat_features:
        return df

    encoder = OneHotEncoder(drop="first", sparse_output=False)
    one_hot = encoder.fit_transform(df[cat_features])
    hot_df = pd.DataFrame(
        one_hot,
        columns=encoder.get_feature_names_out(cat_features),
        index=df.index,
    )
    df = df.drop(columns=cat_features)
    df = pd.concat([df, hot_df], axis=1)
    return df
