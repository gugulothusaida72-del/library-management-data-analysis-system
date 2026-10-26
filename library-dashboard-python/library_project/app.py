"""Library Management Data Analysis Dashboard.  Run: python app.py  ->  http://127.0.0.1:5000"""
import os, re, json, io
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file, session
import analysis as A

app = Flask(__name__); app.secret_key = "library-dashboard"; app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
D = os.path.join(A.BASE, "data"); os.makedirs(D, exist_ok=True)
SAMPLE = "library_records"; cache = {}

def names(): return sorted(f[:-4] for f in os.listdir(D) if f.endswith(".csv"))
def current():
    n = request.args.get("ds") or session.get("ds") or SAMPLE
    return n if n in names() else (SAMPLE if SAMPLE in names() else names()[0])
def load(n):
    if n not in cache:
        mp = os.path.join(D, n + ".map.json"); m = json.load(open(mp)) if os.path.exists(mp) else None
        cache[n] = A.normalize(pd.read_csv(os.path.join(D, n + ".csv")), m)
    return cache[n]
def view():
    df, rep = load(current()); a = request.args
    return A.apply_filters(df, a.get("dept", ""), a.get("cat", ""), a.get("year", "")), df, rep
def err(msg, code=400, **kw): return jsonify(error=msg, **kw), code

@app.route("/")
def index(): return render_template("dashboard.html")

@app.route("/api/datasets")
def datasets():
    out = []
    for n in names():
        try: df, rep = load(n); out.append(dict(name=n, rows=len(df), raw=rep["raw_rows"], protected=n == SAMPLE, active=n == current(), frm=str(df.Issue_Date.min().date()), to=str(df.Issue_Date.max().date())))
        except Exception as e: out.append(dict(name=n, rows=0, error=str(e), protected=n == SAMPLE, active=False))
    return jsonify(out)

@app.route("/api/select/<n>", methods=["POST"])
def select(n):
    if n not in names(): return err("Unknown dataset", 404)
    session["ds"] = n; return jsonify(ok=True)

@app.route("/api/delete/<n>", methods=["POST"])
def delete(n):
    if n == SAMPLE or n not in names(): return err("Cannot delete this dataset")
    for ext in (".csv", ".map.json"):
        p = os.path.join(D, n + ext); os.path.exists(p) and os.remove(p)
    cache.pop(n, None); session["ds"] = SAMPLE; return jsonify(ok=True)

@app.route("/api/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f or not f.filename: return err("Choose a file first.")
    stem, ext = os.path.splitext(f.filename); ext = ext.lower()
    try:
        if ext in (".xlsx", ".xlsm"): raw = pd.read_excel(f)
        elif ext in (".csv", ".txt", ".tsv"): raw = pd.read_csv(f, sep=None, engine="python", encoding_errors="replace")
        else: return err("Unsupported file type. Upload .csv or .xlsx")
    except Exception as e: return err(f"Could not read file: {e}")
    raw.columns = [str(c).strip() for c in raw.columns]
    mapping = json.loads(request.form["mapping"]) if request.form.get("mapping") else A.detect(list(raw.columns))
    mapping = {k: v for k, v in mapping.items() if v}
    if any(c not in mapping for c in A.REQUIRED) and not request.form.get("mapping"):
        return err("Some required columns could not be recognised. Please map them.", 422, columns=list(raw.columns), mapping=mapping, required=A.REQUIRED, fields=A.FIELDS)
    try: df, rep = A.normalize(raw, mapping)
    except ValueError as e: return err(str(e), 422, columns=list(raw.columns), mapping=mapping, required=A.REQUIRED, fields=A.FIELDS)
    if df.empty: return err("No valid rows found. Check the date columns and mapping.", 422, columns=list(raw.columns), mapping=mapping, required=A.REQUIRED, fields=A.FIELDS)
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", request.form.get("name") or stem).strip("_") or "dataset"; n, i = base, 2
    while n in names(): n = f"{base}_{i}"; i += 1
    raw.to_csv(os.path.join(D, n + ".csv"), index=False); json.dump(mapping, open(os.path.join(D, n + ".map.json"), "w"))
    cache[n] = (df, rep); session["ds"] = n
    return jsonify(ok=True, name=n, rows=len(df), report=dict(raw=rep["raw_rows"], duplicates=rep["duplicates"], invalid=rep["invalid"], notes=rep["notes"]))

@app.route("/api/filters")
def filters():
    df, _ = load(current())
    return jsonify(dept=sorted(df.Department.unique().tolist()), cat=sorted(df.Category.unique().tolist()), year=sorted(int(y) for y in df.Year.unique()))

@app.route("/api/data")
def data():
    fdf, df, rep = view(); r = A.analyse(fdf, float(request.args.get("fine", 2) or 2))
    r["report"] = dict(raw_rows=rep["raw_rows"], duplicates=rep["duplicates"], invalid=rep["invalid"], filled_category=rep["filled_category"], missing_return=rep["missing_return"], clean_rows=rep["clean_rows"], notes=rep["notes"], mapping=rep["mapping"], dataset=current())
    return jsonify(r)

@app.route("/api/recommend")
def rec(): return jsonify(A.recommend(view()[0], request.args.get("book", "")))

@app.route("/plot/<name>.png")
def plot(name):
    fdf = view()[0]; r = A.analyse(fdf) if not fdf.empty else None
    b = A.plot(r, name) if r and not r["empty"] else None
    return send_file(b, mimetype="image/png") if b else ("", 404)

@app.route("/download/clean.csv")
def dl_clean():
    b = io.BytesIO(view()[0].to_csv(index=False).encode()); return send_file(b, mimetype="text/csv", as_attachment=True, download_name="clean_library_records.csv")

@app.route("/download/template.csv")
def dl_template():
    t = ",".join(A.FIELDS) + "\nT0001,S001,CSE,B001,Clean Code,Computer Science,2025-01-10,2025-01-24,2025-01-20\nT0002,S002,ECE,B002,Thermodynamics,Engineering,2025-01-12,2025-01-26,\n"
    return send_file(io.BytesIO(t.encode()), mimetype="text/csv", as_attachment=True, download_name="library_template.csv")

if __name__ == "__main__": app.run(debug=False)
