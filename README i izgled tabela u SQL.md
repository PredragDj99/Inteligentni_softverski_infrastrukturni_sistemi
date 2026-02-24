# AIProject

Tabele za SQL Server

CREATE TABLE LoadData (
    Id INT IDENTITY PRIMARY KEY,
    Timestamp DATETIME NOT NULL,
    RegionName VARCHAR(50) NOT NULL,
    PTID INT,
    Load FLOAT,
    TimeZone VARCHAR(10)
);

CREATE TABLE WeatherData (
    Id INT IDENTITY PRIMARY KEY,
    Timestamp DATETIME NOT NULL,
    City VARCHAR(100),
    Temp FLOAT,
    FeelsLike FLOAT,
    Dew FLOAT,
    Humidity FLOAT,
    Precip FLOAT,
    PrecipProb FLOAT,
    Snow FLOAT,
    SnowDepth FLOAT,
    WindGust FLOAT,
    WindSpeed FLOAT,
    WindDir FLOAT,
    Pressure FLOAT,
    CloudCover FLOAT,
    Visibility FLOAT,
    SolarRadiation FLOAT,
    UVIndex FLOAT,
    Conditions VARCHAR(50)
);