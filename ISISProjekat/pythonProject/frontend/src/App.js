import React, { useState } from 'react';
import logo from './logo.svg';
import './App.css';

function App() {
  const [treningFiles, setTreningFiles] = useState([]);
  const [messageUpload, setMessageUpload] = useState('');
  const [messageTrain, setMessageTrain] = useState('');
  const [messagePredict, setMessagePredict] = useState('');
  const [hyperparams, setHyperparams] = useState({ layers: 2, neurons: 10, epochs: 50 });
  const [predictionResult, setPredictionResult] = useState([]);
  const [mape, setMape] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [region, setRegion] = useState('N.Y.C.');

  const handleShowLatestPrediction = async () => {
  const datumOd = document.getElementById("DatumPrognozeOd").value;
  const datumDo = document.getElementById("DatumPrognozeDo").value;

    if (!datumOd || !datumDo) {
      setMessagePredict("Morate izabrati oba datuma.");
      return;
    }

    try {
      const response = await fetch(
        "http://127.0.0.1:5000/get-latest-predictions",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            datumOd,
            datumDo,
            region,
          }),
        }
      );

      const data = await response.json();

      if (data.error) {
        setPredictionResult([]);
        setMessagePredict("Greska: " + data.error);
      } else {
        setPredictionResult(data.predictions || []);
        setMessagePredict(
        `  Prikazana poslednja prognoza (${data.rows} zapisa)`
        );
        setMape(null);
      }
    } catch (error) {
      console.error(error);
      setMessagePredict("Greska prilikom citanja prognoze iz baze.");
    }
  };

  const traverseFileTree = (item, path = '', filesArray = []) => {
    return new Promise((resolve) => {
      if (item.isFile) {
        item.file((file) => {
          file.relativePath = path + file.name;
          filesArray.push(file);
          resolve();
        });
      } else if (item.isDirectory) {
        const dirReader = item.createReader();
        const readAllEntries = () => {
          dirReader.readEntries(async (entries) => {
            if (entries.length === 0) {
              resolve();
            } else {
              for (const entry of entries) {
                await traverseFileTree(entry, path + item.name + '/', filesArray);
              }
              readAllEntries();
            }
          });
        };
        readAllEntries();
      }
    });
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setDragOver(false);
    const items = e.dataTransfer.items;
    let filesArray = [];
    for (let i = 0; i < items.length; i++) {
      const item = items[i].webkitGetAsEntry();
      if (item) await traverseFileTree(item, '', filesArray);
    }
    setTreningFiles((prev) => [...prev, ...filesArray]);
  };

  const handleDragOver = (e) => { e.preventDefault(); setDragOver(true); };
  const handleDragLeave = () => { setDragOver(false); };

  const handleUpload = async () => {
    if (!treningFiles.length) {
      setMessageUpload('Niste dodali fajlove.');
      return;
    }
    const formData = new FormData();
    treningFiles.forEach((f) => formData.append('trening', f));

    try {
      const response = await fetch('http://127.0.0.1:5000/run-network', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setMessageUpload(data.result || 'Podaci su poslati!');
    } catch (error) {
      console.error(error);
      setMessageUpload('Greska prilikom slanja fajlova.');
    }
  };

  const handleTrain = async () => {
    if (hyperparams.layers < 1 || hyperparams.neurons < 1 || hyperparams.epochs < 1) {
      setMessageTrain('Svi hiperparametri moraju biti pozitivni brojevi.');
      return;
    }
    const parametri = {
      layers: hyperparams.layers,
      neurons: hyperparams.neurons,
      epochs: hyperparams.epochs,
      datumOd: document.getElementById('DatumOd').value,
      datumDo: document.getElementById('DatumDo').value,
      region: region
    };
    try {
      const response = await fetch('http://127.0.0.1:5000/train-model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(parametri),
      });
      const data = await response.json();
      setMessageTrain(data.result || 'Trening je pokrenut');
    } catch (error) {
      console.error(error);
      setMessageTrain('Greska prilikom treninga.');
    }
  };

  const handlePredict = async () => {
    const datumOd = document.getElementById('DatumPrognozeOd').value;
    const datumDo = document.getElementById('DatumPrognozeDo').value;

    if (!datumOd || !datumDo) {
      setMessagePredict('Morate izabrati oba datuma.');
      return;
    }

    const start = new Date(datumOd);
    const end = new Date(datumDo);
    const diffDays = (end - start) / (1000 * 60 * 60 * 24);

    if (diffDays < 0) { setMessagePredict('Datum DO mora biti posle datuma OD.'); return; }
    if (diffDays > 7) { setMessagePredict('Prognoza ne može biti duža od 7 dana.'); return; }

    const parametri = { datumOd, datumDo, region };

    try {
      const response = await fetch('http://127.0.0.1:5000/run-prediction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(parametri),
      });
      const data = await response.json();
      if (data.error) {
        setMessagePredict("Greska: " + data.error);
        setPredictionResult([]);
        setMape(null);
      } else {
        setPredictionResult(data.predictions || []);
        setMape(data.mape);
        setMessagePredict(data.message || 'Prognoza uspesno odradjena');
      }
    } catch (error) {
      console.error(error);
      setMessagePredict('Greska prilikom prognoze.');
    }
  };

  const renderFileList = (files) => {
    return files.length ? (
      <ul style={{ maxHeight: '150px', overflowY: 'auto' }}>
        {files.map((f, i) => (<li key={i}>{f.relativePath}</li>))}
      </ul>
    ) : (<p>Nema fajlova</p>);
  };

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <h2>Prognoza potrošnje električne energije</h2>
      </header>

      <div className="container">
        <div className="unos">
          <h3>Učitaj podatke</h3>
          <label>Ovde prevuci folder sa podacima za trening</label>
          <div
            className={`drop-zone ${dragOver ? 'active' : ''}`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
          >
            {renderFileList(treningFiles)}
          </div>
          <p>Broj učitanih fajlova: {treningFiles.length}</p>
          <button className="button" onClick={handleUpload}>Pošalji fajlove</button>
          <div className="half status">
            <h4>Status za upload:</h4>
            <p>{messageUpload}</p>
          </div>
        </div>

        <div className="obrada">
          <h3>Trening modela</h3>
          <label>Datum od:</label>
          <input type="date" id="DatumOd" />
          <label>Datum do:</label>
          <input type="date" id="DatumDo" />
          <label>Regija:</label>
          <select value={region} onChange={(e) => setRegion(e.target.value)}>
            <option value="N.Y.C.">N.Y.C.</option>
            <option value="WEST">WEST</option>
            <option value="CAPITL">CAPITL</option>
            <option value="CENTRL">CENTRL</option>
            <option value="DUNWOD">DUNWOD</option>
            <option value="GENESE">GENESE</option>
            <option value="HUD VL">HUD VL</option>
            <option value="LONGIL">LONGIL</option>
            <option value="MHK VL">MHK VL</option>
            <option value="MILLWD">MILLWD</option>
            <option value="NORTH">NORTH</option>
          </select>
          <h4>Hiperparametri</h4>
          <label>Broj slojeva:</label>
          <input type="number" value={hyperparams.layers} min={1} onChange={(e) => setHyperparams({ ...hyperparams, layers: parseInt(e.target.value) })}/>
          <label>Broj neurona po sloju:</label>
          <input type="number" value={hyperparams.neurons} min={1} onChange={(e) => setHyperparams({ ...hyperparams, neurons: parseInt(e.target.value) })}/>
          <label>Broj epoha:</label>
          <input type="number" value={hyperparams.epochs} min={1} onChange={(e) => setHyperparams({ ...hyperparams, epochs: parseInt(e.target.value) })}/>
          <button className="button" onClick={handleTrain}>Pokreni trening</button>
          <div className="half status">
            <h4>Status treninga:</h4>
            <p>{messageTrain}</p>
          </div>
        </div>

        <div className="obrada">
          <h3>Prognoza</h3>
          <label>Datum od:</label>
          <input type="date" id="DatumPrognozeOd" />
          <label>Datum do:</label>
          <input type="date" id="DatumPrognozeDo" />
          <label>Regija:</label>
          <select value={region} onChange={(e) => setRegion(e.target.value)}>
            <option value="N.Y.C.">N.Y.C.</option>
            <option value="WEST">WEST</option>
            <option value="CAPITL">CAPITL</option>
            <option value="CENTRL">CENTRL</option>
            <option value="DUNWOD">DUNWOD</option>
            <option value="GENESE">GENESE</option>
            <option value="HUD VL">HUD VL</option>
            <option value="LONGIL">LONGIL</option>
            <option value="MHK VL">MHK VL</option>
            <option value="MILLWD">MILLWD</option>
            <option value="NORTH">NORTH</option>
          </select>
          <button className="button" onClick={handlePredict}>Predikcija</button>
          <div className="half status">
            <h4>Status prognoze:</h4>
            <p>{messagePredict}</p>
          </div>

          {mape !== null && (
                <p><strong>MAPE:</strong> {mape.toFixed(2)}%</p>
          )}

          <button className="button" onClick={handleShowLatestPrediction}>
            Prikazi poslednju prognozu
          </button>

          {predictionResult.length > 0 && (
            <div className="prediction-result">
              <table>
                <thead>
                  <tr>
                  <th>Timestamp</th>
                  <th>Predicted Load</th>
                  </tr>
                </thead>
                <tbody>
                  {predictionResult.map((row, idx) => (
                    <tr key={idx}>
                      <td>{row.Datetime}</td>
                      <td>{row.Predicted_Load.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
