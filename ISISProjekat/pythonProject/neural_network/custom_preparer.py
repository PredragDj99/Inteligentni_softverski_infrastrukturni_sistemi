import numpy as np
import pandas as pd

class CustomPreparer:
    def __init__(
        self,
        df,
        feature_columns,
        target_column=None,
        min_values=None,
        max_values=None,
        datetime_column="TS"
    ):
        self.df = df.copy()
        self.feature_columns = feature_columns.copy()
        self.target_column = target_column
        self.datetime_column = datetime_column

        self.min_values = min_values
        self.max_values = max_values
        self.eps = 1e-8

        # Izuzeci zbog praznika
        self.excluded_dates = [
            "2018-01-01","2018-01-15","2018-02-14","2018-02-19","2018-03-30",
            "2018-04-01","2018-05-13","2018-05-28","2018-06-01","2018-06-17",
            "2018-07-04","2018-09-03","2018-10-08","2018-10-31","2018-11-11",
            "2018-11-22","2018-12-25",
            "2019-01-01","2019-01-21","2019-02-14","2019-02-18","2019-04-19",
            "2019-04-21","2019-05-12","2019-05-27","2019-06-16","2019-07-04",
            "2019-09-02","2019-10-14","2019-10-31","2019-11-11","2019-11-28",
            "2019-12-25",
            "2020-01-01","2020-01-20","2020-02-14","2020-02-17","2020-04-10",
            "2020-04-12","2020-05-10","2020-05-25","2020-06-21","2020-07-03",
            "2020-07-04","2020-09-07","2020-10-12","2020-10-31","2020-11-11",
            "2020-11-26","2020-12-25",
            "2021-01-01","2021-01-20","2021-02-14","2021-02-17","2021-04-10",
            "2021-04-12","2021-05-10","2021-05-25","2021-06-21","2021-07-03",
            "2021-07-04","2021-09-07","2021-10-12","2021-10-31","2021-11-11",
            "2021-11-26","2021-12-25"
        ]

        # Satna rezolucija
        if self.datetime_column in self.df.columns:
            self.df[self.datetime_column] = pd.to_datetime(self.df[self.datetime_column])

            # Izuzimanje praznika tj. specijalne dane uklanjam
            if self.excluded_dates and len(self.excluded_dates) > 0:
                excluded_dates_dt = pd.to_datetime(self.excluded_dates).date
                self.df = self.df[~self.df[self.datetime_column].dt.date.isin(excluded_dates_dt)]

            # Izbacivanje podataka iz 2020. godine zbog korone
            self.df = self.df[self.df[self.datetime_column].dt.year != 2020]

            self.df["hour"] = self.df[self.datetime_column].dt.hour
            self.df["day_of_week"] = self.df[self.datetime_column].dt.dayofweek
            self.df["month"] = self.df[self.datetime_column].dt.month
            self.df["is_weekend"] = (
                self.df["day_of_week"].isin([5, 6]).astype(int)
            )

            self.feature_columns.extend(
                ["hour", "day_of_week", "month", "is_weekend"]
            )

        # Clipovanje negativnih vrednosti
        self.df[self.feature_columns] = self.df[self.feature_columns].clip(lower=0)

        if self.target_column is not None:
            self.df[self.target_column] = self.df[self.target_column].clip(lower=0)

    # Trening faza

    def prepare_for_training(self):
        if self.target_column is None:
            raise ValueError("target_column mora biti definisan za trening")

        #X = self.df[self.feature_columns].values.astype(np.float32)
        #y = self.df[self.target_column].values.astype(np.float32).reshape(-1, 1)

        # Zamena NaN vrednosti
        # X = np.nan_to_num(X, nan=0.0) # matrica ulaznih podataka
        # y = np.nan_to_num(y, nan=0.0) # vektor ciljne vrednosti(temperatura, vlaga,...)

        # Interpolacija izmedjuu prve i poslednje vrednosti
        #for i in range(X.shape[1]):
        #    col = X[:, i]
        # Indeksi koji nisu NaN
        #    valid_idx = np.where(~np.isnan(col))[0]
        #    if len(valid_idx) == 0:
            # ako je cela kolona NaN, postavi na 0
        #        col[:] = 0.0
        #    else:
        #        first = col[valid_idx[0]]
        #        last = col[valid_idx[-1]]
        #        fill_value = (first + last) / 2
        #        col[np.isnan(col)] = fill_value
        #    X[:, i] = col


        # Isto za y
        #y_col = y[:, 0]
        #valid_idx = np.where(~np.isnan(y_col))[0]
        #if len(valid_idx) == 0:
        #    y_col[:] = 0.0
        #else:
        #    first = y_col[valid_idx[0]]
        #    last = y_col[valid_idx[-1]]
        #    fill_value = (first + last) / 2
        #    y_col[np.isnan(y_col)] = fill_value
        #y[:, 0] = y_col

        # Linearna interpolacija za feature kolone
        self.df[self.feature_columns] = (
            self.df[self.feature_columns]
            .interpolate(method='linear', limit_direction='both')
        )

        # Linearna interpolacija za target kolonu
        self.df[self.target_column] = (
            self.df[self.target_column]
            .interpolate(method='linear', limit_direction='both')
        )

        # Pretvaranje u numpy
        X = self.df[self.feature_columns].values.astype(np.float32)
        y = self.df[self.target_column].values.astype(np.float32).reshape(-1, 1)

        # Ako ostane neki NaN pretvaram ga u 0
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        # Min-Max normalizacija (samo trening)
        self.min_values = X.min(axis=0)
        self.max_values = X.max(axis=0)

        denom = self.max_values - self.min_values
        denom[denom < self.eps] = 1.0

        X_norm = (X - self.min_values) / denom

        return X_norm, y

    # Predikcija
    def prepare_for_prediction(self):
        if self.min_values is None or self.max_values is None:
            raise ValueError(
                "min_values i max_values moraju biti ucitani pre predikcije"
            )

        #X = self.df[self.feature_columns].values.astype(np.float32)

        # Zamena NaN vrednosti
        #X = np.nan_to_num(X, nan=0.0)

        # Interpolacija
        #for i in range(X.shape[1]):
        #    col = X[:, i]
        #    valid_idx = np.where(~np.isnan(col))[0]
        #    if len(valid_idx) == 0:
                # ako je cela kolona NaN, postavi na 0
        #        col[:] = 0.0
        #    else:
        #        first = col[valid_idx[0]]
        #        last = col[valid_idx[-1]]
        #        fill_value = (first + last) / 2
        #        col[np.isnan(col)] = fill_value
        #    X[:, i] = col

        # Linearna interpolacija feature kolona
        self.df[self.feature_columns] = (
            self.df[self.feature_columns]
            .interpolate(method='linear', limit_direction='both')
        )

        # Pretvaranje u numpy
        X = self.df[self.feature_columns].values.astype(np.float32)

        # Ako ostane neki NaN pretvaram ga u 0
        X = np.nan_to_num(X, nan=0.0)

        # min-max normalizacija
        denom = self.max_values - self.min_values
        denom[denom < self.eps] = 1.0

        X_norm = (X - self.min_values) / denom

        return X_norm
