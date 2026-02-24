import os
import threading
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import traceback

from neural_network.scorer import Scorer
from neural_network.ann_regression import AnnRegression
from neural_network.custom_preparer import CustomPreparer

# SQL SERVER
SERVER = r"PEDJA\SQLEXPRESS"
DATABASE = "ISIS"
engine = create_engine(
    f"mssql+pyodbc://{SERVER}/{DATABASE}?driver=ODBC+Driver+17+for+SQL+Server"
)

# Flask
app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = "uploads"
EXPORT_FOLDER = "exports"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)
os.makedirs("my_Models", exist_ok=True)
os.makedirs("my_Predictions", exist_ok=True)

# CSV -> BAZA
def create_table_if_not_exists(table_name, df):
    with engine.connect() as conn:
        result = conn.execute(text(f"""
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_NAME = '{table_name}'
        """))
        if result.scalar() == 0:
            cols = []
            for col, dtype in df.dtypes.items():
                if "float" in str(dtype):
                    sql_type = "FLOAT"
                elif "int" in str(dtype):
                    sql_type = "BIGINT"
                elif "datetime" in str(dtype):
                    sql_type = "DATETIME"
                else:
                    sql_type = "NVARCHAR(MAX)"
                cols.append(f"[{col}] {sql_type}")
            conn.execute(text(f"CREATE TABLE {table_name} ({','.join(cols)})"))

def insert_csv_to_db(file_path):
    df = pd.read_csv(file_path)

    # normalizacija naziva kolona
    df.columns = [c.strip().lower().replace(" ", "").replace(".", "") for c in df.columns]

    # Load data
    if "load" in df.columns:
        column_map = {
            "timestamp": "Time_Stamp",
            "timezone": "Time_Zone",
            "name": "Name",
            "ptid": "PTID",
            "load": "Load"
        }

        df = df.rename(columns=column_map)

        allowed = list(column_map.values())
        df = df[[c for c in allowed if c in df.columns]]

        # Konvertuj Time_Stamp u datetime
        df["Time_Stamp"] = pd.to_datetime(df["Time_Stamp"], errors="coerce")

        # Ako postoje NULL vrednosti u Time_Stamp, izbaci ih ili popuni
        df = df.dropna(subset=["Time_Stamp"])  # izbaci redove bez vremena

        df.to_sql("LoadData", engine, if_exists="append", index=False)

    # Weather data
    else:
        column_map = {
            "name": "City",
            "datetime": "Timestamp",
            "temp": "Temp",
            "feelslike": "FeelsLike",
            "dew": "Dew",
            "humidity": "Humidity",
            "precip": "Precip",
            "precipprob": "PrecipProb",
            "preciptype": "preciptype",
            "snow": "Snow",
            "snowdepth": "SnowDepth",
            "windgust": "WindGust",
            "windspeed": "WindSpeed",
            "winddir": "WindDir",
            "sealevelpressure": "Pressure",
            "cloudcover": "CloudCover",
            "visibility": "Visibility",
            "solarradiation": "SolarRadiation",
            "solarenergy": "solarenergy",
            "uvindex": "UVIndex",
            "severerisk": "severerisk",
            "conditions": "Conditions"
        }

        df = df.rename(columns=column_map)

        allowed = list(column_map.values())
        df = df[[c for c in allowed if c in df.columns]]

        df.to_sql("WeatherData", engine, if_exists="append", index=False)

# Trening
def train_model_in_background(layers, neurons, epochs, datum_od, datum_do, region):
    try:
        os.makedirs("my_Models", exist_ok=True)

        # Ucitavanje podataka
        query_weather = f"""
            SELECT *, TRY_CONVERT(datetime, Timestamp) AS TS
            FROM WeatherData
            WHERE TRY_CONVERT(datetime, Timestamp) BETWEEN '{datum_od}' AND '{datum_do}'
            ORDER BY TS
        """
        query_load = f"""
            SELECT TRY_CONVERT(datetime, Time_Stamp) AS TS, Load
            FROM LoadData
            WHERE Name='{region}'
              AND TRY_CONVERT(datetime, Time_Stamp) BETWEEN '{datum_od}' AND '{datum_do}'
        """

        df_w = pd.read_sql(query_weather, engine)
        df_l = pd.read_sql(query_load, engine)

        df = pd.merge(df_w, df_l, on="TS", how="left")
        df.interpolate(inplace=True)
        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)

        feature_columns = [
            'Temp','FeelsLike','Dew','Humidity','Precip','PrecipProb','Snow',
            'SnowDepth','WindGust','WindSpeed','WindDir','Pressure','CloudCover',
            'Visibility','SolarRadiation','solarenergy','UVIndex','severerisk'
        ]

        # Load mi je target column
        preparer = CustomPreparer(df, feature_columns, 'Load')
        X, y = preparer.prepare_for_training()

        ann = AnnRegression()
        ann.number_of_hidden_layers = layers
        ann.number_of_neurons_in_first_hidden_layer = neurons
        ann.number_of_neurons_in_other_hidden_layers = neurons
        ann.epoch_number = epochs

        # Split pre treninga
        split_idx = int(len(X) * 0.85)

        X_train = X[:split_idx]
        y_train = y[:split_idx]

        X_test = X[split_idx:]
        y_test = y[split_idx:]

        # Trening na train podacima
        ann.compile_and_fit(X_train, y_train)

        # MAPE/RMSE evaluacija

        # Predikcija
        y_train_pred = ann.predict(X_train)
        y_test_pred = ann.predict(X_test)

        # Evaluacija
        scorer = Scorer()
        train_rmse, test_rmse = scorer.get_score(y_train, y_train_pred, y_test, y_test_pred)
        train_mape, test_mape = scorer.get_absolute(y_train, y_train_pred, y_test, y_test_pred)

        print(f"Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")
        print(f"Train MAPE: {train_mape:.2f}%, Test MAPE: {test_mape:.2f}%")

        # Cuvanje modela i min/max
        model_path = f"my_Models/trained_model_{region}.keras"
        min_path = f"my_Models/min_values_{region}.npy"
        max_path = f"my_Models/max_values_{region}.npy"

        ann.save_model(model_path)
        np.save(min_path, preparer.min_values)
        np.save(max_path, preparer.max_values)

        print(f"Trening završen za regiju {region}. Model sacuvan u: {model_path}")

    except Exception as e:
        print("Greska tokom treninga:", str(e))
        print(traceback.format_exc())

# Ucitavanje fajlova

@app.route("/run-network", methods=["POST"])
def run_network():
    try:
        files = request.files.getlist("trening")

        if not files:
            return jsonify({"error": "Nema poslatih fajlova"}), 400

        for f in files:
            file_path = os.path.join(UPLOAD_FOLDER, f.filename)
            f.save(file_path)
            insert_csv_to_db(file_path)

        return jsonify({"message": "CSV fajlovi su uspesno upisani u bazu"}), 200

    except Exception as e:
        print("Greska:", str(e))
        return jsonify({"error": str(e)}), 500

# Trening

@app.route("/train-model", methods=["POST"])
def train_model():
    data = request.get_json()
    t = threading.Thread(
        target=train_model_in_background,
        args=(
            int(data["layers"]),
            int(data["neurons"]),
            int(data["epochs"]),
            data["datumOd"],
            data["datumDo"],
            data["region"]
        )
    )
    t.start()
    return jsonify({"result": "Trening je pokrenut i bice zavrsen u pozadini"})

# Predikcija
@app.route("/run-prediction", methods=["POST"])
def run_prediction():
    try:
        os.makedirs("my_Predictions", exist_ok=True)
        os.makedirs("my_Models", exist_ok=True)
        data = request.get_json()
        region = data.get("region", "N.Y.C.")

        # Datumi
        start_date = datetime.fromisoformat(data["datumOd"])
        end_date = datetime.fromisoformat(data["datumDo"])

        # Ucitavanje weather podataka
        query = f"""
            SELECT *, TRY_CONVERT(datetime, Timestamp) AS TS
            FROM WeatherData
            WHERE TRY_CONVERT(datetime, Timestamp)
            BETWEEN '{start_date}' AND '{end_date}'
            ORDER BY TS
        """
        df_weather = pd.read_sql(query, engine)
        if df_weather.empty:
            return jsonify({"error": "Nema vremenskih podataka za izabrani period"}), 400

        feature_columns = [
            'Temp','FeelsLike','Dew','Humidity','Precip','PrecipProb','Snow',
            'SnowDepth','WindGust','WindSpeed','WindDir','Pressure','CloudCover',
            'Visibility','SolarRadiation','solarenergy','UVIndex','severerisk'
        ]

        # Ucitavanje modela i min/max
        model_path = f"my_Models/trained_model_{region}.keras"
        min_path = f"my_Models/min_values_{region}.npy"
        max_path = f"my_Models/max_values_{region}.npy"

        if not os.path.exists(model_path) or not os.path.exists(min_path) or not os.path.exists(max_path):
            return jsonify({"error": "Model ili min/max fajlovi nisu pronadjeni u my_Models"}), 400

        min_v = np.load(min_path)
        max_v = np.load(max_path)

        preparer = CustomPreparer(df_weather, feature_columns, None, min_v, max_v)
        X = preparer.prepare_for_prediction()

        # Predikcija
        ann = AnnRegression()
        ann.load_model(model_path)


        # Dodato nakon baga koji se pojavio prilikom ucitavanja TEST fajla
        # Koliko feature kolona model zapravo očekuje
        expected_features = ann.model.input_shape[1]

        # Dodato nakon baga koji se pojavio prilikom ucitavanja TEST fajla
        # Ako imamo više kolona nego što model očekuje → skrati
        if X.shape[1] > expected_features:
            X = X[:, :expected_features]

        # Dodato nakon baga koji se pojavio prilikom ucitavanja TEST fajla
        # Ako imamo manje kolona → dopuni nulama
        elif X.shape[1] < expected_features:
            diff = expected_features - X.shape[1]
            padding = np.zeros((X.shape[0], diff), dtype=np.float32)
            X = np.hstack((X, padding))



        preds = ann.predict(X)

        # Dodato nakon baga koji se pojavio prilikom ucitavanja TEST fajla
        # Poravnanje duzine
        if len(df_weather) != len(preds):
            df_weather = df_weather.iloc[:len(preds)].reset_index(drop=True)



        # Prilagodjavanje imena kolone za bazu
        df_result = df_weather[['TS']].copy()
        df_result.rename(columns={'TS': 'Datetime'}, inplace=True)
        df_result['Predicted_Load'] = np.array(preds).reshape(-1)
        df_result['Region'] = region
        df_result['Created_At'] = datetime.now()

        # MAPE
        query_load = f"""
            SELECT TRY_CONVERT(datetime, Time_Stamp) AS TS, Load
            FROM LoadData
            WHERE Name='{region}'
              AND TRY_CONVERT(datetime, Time_Stamp) BETWEEN '{start_date}' AND '{end_date}'
        """
        df_load = pd.read_sql(query_load, engine)
        mape = None
        if not df_load.empty:
            df_load.rename(columns={'TS': 'Datetime'}, inplace=True)
            df_merged = pd.merge(df_result, df_load, on='Datetime', how='left')
            y_true = df_merged['Load'].values
            y_pred = df_merged['Predicted_Load'].values
            mask = y_true != 0
            if np.any(mask):
                mape = float(np.mean(np.abs((y_true[mask]-y_pred[mask])/y_true[mask]))*100)
                print(f"MAPE za {region} u periodu {start_date} - {end_date}: {mape:.2f}%")
            else:
                print(f"MAPE: svi stvarni Load podaci su 0, nije moguce izracunati MAPE")
        else:
            print("Nema stvarnih Load podataka za izabrani period, MAPE nije izracunata")

        # Upis u bazu
        create_table_if_not_exists("Predictions", df_result)
        df_result.to_sql("Predictions", engine, if_exists="append", index=False)

        # CSV export
        csv_file = f"my_Predictions/prediction_{region}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_result[['Datetime','Predicted_Load']].to_csv(csv_file, index=False)

        return jsonify({
            "message": "Prognoza uspješno izvrsena",
            "predictions": df_result[['Datetime','Predicted_Load']].to_dict(orient='records'),
            "csv": csv_file,
            "rows": int(len(df_result)),
            "mape": mape
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "trace": traceback.format_exc()
        }), 500

@app.route("/get-latest-predictions", methods=["POST"])
def get_latest_predictions():
    data = request.get_json()

    region = data["region"]
    datum_od = data["datumOd"]
    datum_do = data["datumDo"]

    query = f"""
    WITH LatestPredictions AS (
        SELECT *,
               ROW_NUMBER() OVER (
                   PARTITION BY Datetime, Region
                   ORDER BY Created_At DESC
               ) AS rn
        FROM Predictions
        WHERE Region = '{region}'
          AND Datetime BETWEEN '{datum_od}' AND '{datum_do}'
    )
    SELECT Datetime, Predicted_Load
    FROM LatestPredictions
    WHERE rn = 1
    ORDER BY Datetime
    """

    df = pd.read_sql(query, engine)

    return jsonify({
        "predictions": df.to_dict(orient="records"),
        "rows": int(len(df))
    })

if __name__ == "__main__":
    app.run(debug=True)
