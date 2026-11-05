"""
Penguin Species Classifier - web app for deployment on Vercel.

The model is the same one built in Penguin_Classification.ipynb:
median imputer -> StandardScaler -> KNN (k=5).
It is trained when the app starts (the dataset is tiny, so this takes
milliseconds), which avoids shipping a pickled model and any scikit-learn
version mismatch between training and deployment.
"""
import csv
import os

import numpy as np
from flask import Flask, jsonify, render_template_string, request
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "penguins_lter.csv")

FEATURES = [
    "Culmen Length (mm)",
    "Culmen Depth (mm)",
    "Flipper Length (mm)",
    "Body Mass (g)",
]

# Plausible input ranges (slightly wider than the observed data)
LIMITS = {
    "Culmen Length (mm)": (25, 70),
    "Culmen Depth (mm)": (10, 25),
    "Flipper Length (mm)": (150, 250),
    "Body Mass (g)": (2000, 7500),
}


def load_data():
    """Read the CSV with the standard library (keeps the deployment small)."""
    rows, labels = [], []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            species = (rec.get("Species") or "").strip()
            if not species:
                continue
            row = []
            for col in FEATURES:
                val = (rec.get(col) or "").strip()
                row.append(float(val) if val else np.nan)
            rows.append(row)
            labels.append(species.split(" ")[0])  # Adelie / Chinstrap / Gentoo
    return np.array(rows, dtype=float), np.array(labels)


def build_model():
    X, y = load_data()
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ])
    model.fit(X, y)
    return model, len(y)


MODEL, N_TRAIN = build_model()


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Penguin Species Classifier</title>
<style>
  :root { --bg:#f5f7fa; --card:#ffffff; --text:#1c2733; --muted:#5b6b7b;
          --accent:#1f6feb; --border:#d8dee6; --bar:#cfe0fb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#10151b; --card:#18202a; --text:#e8edf2; --muted:#9aa9b8;
            --accent:#5b9dff; --border:#2b3745; --bar:#24476f; }
  }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
         background:var(--bg); color:var(--text); line-height:1.5; }
  main { max-width:640px; margin:0 auto; padding:32px 16px 48px; }
  h1 { font-size:1.6rem; margin:0 0 4px; }
  p.sub { color:var(--muted); margin:0 0 24px; }
  .card { background:var(--card); border:1px solid var(--border);
          border-radius:12px; padding:20px; margin-bottom:20px; }
  label { display:block; font-weight:600; margin:14px 0 4px; }
  label small { font-weight:400; color:var(--muted); }
  input { width:100%; padding:10px 12px; font-size:1rem; border-radius:8px;
          border:1px solid var(--border); background:var(--bg); color:var(--text); }
  button { margin-top:20px; width:100%; padding:12px; font-size:1rem; font-weight:600;
           border:0; border-radius:8px; background:var(--accent); color:#fff; cursor:pointer; }
  button.ghost { background:transparent; color:var(--accent); border:1px solid var(--border);
                 margin-top:8px; font-weight:500; }
  #result h2 { margin:0 0 12px; font-size:1.25rem; }
  .row { display:flex; align-items:center; gap:10px; margin:8px 0; font-size:.95rem; }
  .row span.name { width:90px; }
  .track { flex:1; height:12px; background:var(--bar); border-radius:6px; overflow:hidden; }
  .fill { height:100%; background:var(--accent); }
  .err { color:#d1242f; margin-top:12px; }
  footer { color:var(--muted); font-size:.85rem; text-align:center; }
  a { color:var(--accent); }
</style>
</head>
<body>
<main>
  <h1>Penguin Species Classifier</h1>
  <p class="sub">Enter four body measurements and the model predicts whether the
  penguin is an Adelie, Chinstrap or Gentoo.</p>

  <div class="card">
    <label>Culmen length (mm) <small>typical 32 to 60</small>
      <input id="cl" type="number" step="0.1" value="44.5"></label>
    <label>Culmen depth (mm) <small>typical 13 to 22</small>
      <input id="cd" type="number" step="0.1" value="17.2"></label>
    <label>Flipper length (mm) <small>typical 172 to 231</small>
      <input id="fl" type="number" step="1" value="197"></label>
    <label>Body mass (g) <small>typical 2700 to 6300</small>
      <input id="bm" type="number" step="10" value="4050"></label>
    <button id="go">Predict species</button>
    <button class="ghost" id="ex1" type="button">Try an example: Gentoo-like</button>
    <button class="ghost" id="ex2" type="button">Try an example: Adelie-like</button>
    <div id="error" class="err"></div>
  </div>

  <div class="card" id="result" style="display:none"></div>

  <footer>
    KNN (k=5) trained on __N__ penguins from the Palmer Penguins dataset.
    <br>Source code on <a href="https://github.com/ashk-k10/penguin_classification">GitHub</a>.
  </footer>
</main>

<script>
const $ = id => document.getElementById(id);
function setValues(a,b,c,d){ $("cl").value=a; $("cd").value=b; $("fl").value=c; $("bm").value=d; }
$("ex1").onclick = () => setValues(47.5, 15.0, 217, 5000);
$("ex2").onclick = () => setValues(38.8, 18.3, 190, 3700);

$("go").onclick = async () => {
  $("error").textContent = "";
  const payload = {
    culmen_length: parseFloat($("cl").value),
    culmen_depth: parseFloat($("cd").value),
    flipper_length: parseFloat($("fl").value),
    body_mass: parseFloat($("bm").value)
  };
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!res.ok) { $("error").textContent = data.error || "Something went wrong."; return; }
    let html = "<h2>Predicted species: " + data.species + "</h2>";
    for (const [name, p] of Object.entries(data.probabilities)) {
      const pct = Math.round(p * 100);
      html += '<div class="row"><span class="name">' + name + '</span>' +
              '<div class="track"><div class="fill" style="width:' + pct + '%"></div></div>' +
              '<span>' + pct + '%</span></div>';
    }
    html += '<p class="sub" style="margin-top:12px">The percentages show how many of the 5 nearest ' +
            'penguins in the training data belong to each species.</p>';
    $("result").innerHTML = html;
    $("result").style.display = "block";
  } catch (e) {
    $("error").textContent = "Could not reach the server. Please try again.";
  }
};
</script>
</body>
</html>
"""


@app.get("/")
def home():
    return render_template_string(PAGE.replace("__N__", str(N_TRAIN)))


@app.get("/api/health")
def health():
    return jsonify(status="ok", training_samples=N_TRAIN)


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    keys = {
        "culmen_length": "Culmen Length (mm)",
        "culmen_depth": "Culmen Depth (mm)",
        "flipper_length": "Flipper Length (mm)",
        "body_mass": "Body Mass (g)",
    }
    values = []
    for key, name in keys.items():
        raw = data.get(key)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return jsonify(error=f"Please enter a valid number for {name}."), 400
        if not np.isfinite(val):
            return jsonify(error=f"Please enter a valid number for {name}."), 400
        low, high = LIMITS[name]
        if not (low <= val <= high):
            return jsonify(error=f"{name} should be between {low} and {high}."), 400
        values.append(val)

    X = np.array([values])
    species = MODEL.predict(X)[0]
    probs = MODEL.predict_proba(X)[0]
    probabilities = {cls: round(float(p), 3) for cls, p in zip(MODEL.classes_, probs)}
    return jsonify(species=species, probabilities=probabilities)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
