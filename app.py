"""
BSF Monitor & Larvae Prediction System
Flask backend — All 3 models (LSTM, ANN, RandomForest)
Each model predicts separately, best one highlighted
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import serial, serial.tools.list_ports
import threading, time, json, os, pickle
import numpy as np
import hashlib
from datetime import datetime

app = Flask(__name__)
app.secret_key = "bsf_monitor_secret_2024"


# ── LOGIN SETUP ──────────────────────────────────────────────
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

USERS = {
    "bsfnexus05": hashlib.sha256("bsfnexus05".encode()).hexdigest(),
    "farmer":     hashlib.sha256("farmer123".encode()).hexdigest(),
}

class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(username):
    if username in USERS:
        return User(username)
    return None

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "cycle_data.json")
MODEL_DIR = os.path.join(BASE, "models")

# ── Sensor state ──────────────────────────────────────────────
sensor_data = {
    "larvae": {"temp": None, "hum": None, "timestamp": ""},
    "pupa":   {"temp": None, "hum": None, "timestamp": ""},
    "adult":  {"temp": None, "hum": None, "light": None, "timestamp": ""},
}
serial_status = {"connected": False, "port": None, "error": ""}
data_lock     = threading.Lock()

# ── Cycle data ────────────────────────────────────────────────
def load_cycle_data():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH) as f:
            return json.load(f)
    return {"current_cycle": 1, "cycles": {"1": new_cycle(1)}}

def save_cycle_data():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    with open(DATA_PATH, "w") as f:
        json.dump(cycle_data, f, indent=2)

def new_cycle(cycle_id):
    return {
        "cycle_id":      cycle_id,
        "started_at":    datetime.now().isoformat(),
        "finished":      False,
        "actual_yield_g": None,
        "weeks":         [],
        "days":          [],
    }

cycle_data = load_cycle_data()

# ── Model loading ─────────────────────────────────────────────
model_bundles = {"LSTM": None, "ANN": None, "RandomForest": None}
model_info    = {"type": "linear_fallback", "r2": None, "trained": False}

def find_latest_model(prefix):
    if not os.path.exists(MODEL_DIR):
        return None
    files = [f for f in os.listdir(MODEL_DIR)
             if f.startswith(prefix) and f.endswith(".pkl") and "cycle" in f]
    if not files:
        return None
    files.sort()
    return os.path.join(MODEL_DIR, files[-1])

def load_all_models():
    global model_bundles, model_info
    prefixes = {
        "LSTM":         "lstm_cycle",
        "ANN":          "ann_cycle",
        "RandomForest": "rf_cycle",
    }
    for name, prefix in prefixes.items():
        path = find_latest_model(prefix)
        if path and os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    bundle = pickle.load(f)
                model_bundles[name] = bundle
                print(f"[BSF] {name} loaded from {os.path.basename(path)}")
            except Exception as e:
                print(f"[BSF] Failed to load {name}: {e}")

    best_path = os.path.join(MODEL_DIR, "lstm_model.pkl")
    if os.path.exists(best_path) and all(v is None for v in model_bundles.values()):
        try:
            with open(best_path, "rb") as f:
                bundle = pickle.load(f)
            mtype = bundle.get("model_type", "LSTM")
            model_bundles[mtype] = bundle
            print(f"[BSF] Fallback: loaded lstm_model.pkl as {mtype}")
        except Exception as e:
            print(f"[BSF] Fallback load failed: {e}")

    for name in ["LSTM", "ANN", "RandomForest"]:
        if model_bundles[name]:
            model_info.update(model_bundles[name].get("info", {}))
            break

load_all_models()

# ── Prediction helpers ────────────────────────────────────────
def get_confidence(entries):
    stages     = set(w["stage"] for w in entries)
    days_count = len(entries)
    if "adult" in stages:
        return "high"
    elif "pupa" in stages and "larvae" in stages:
        return "medium-high"
    elif "pupa" in stages or days_count >= 10:
        return "medium"
    return "low"

def stage_from_week(week):
    if week <= 3:   return "larvae"
    elif week <= 5: return "pupa"
    else:           return "adult"

def stageFromDay(day):
    if day <= 24:   return "larvae"
    elif day <= 38: return "pupa"
    else:           return "adult"

def growth_sigmoid(day):
    day = int(day)
    if day < 5:
        return 0.02
    elif 5 <= day <= 24:
        progress = (day - 5) / (24 - 5)
        return float(1 / (1 + np.exp(-10 * (progress - 0.5))))
    else:
        return 1.0

def predict_single(bundle, col_values, all_days=None, current_day=None):
    if bundle is None:
        return None, "no_model"

    scaler_X     = bundle["scaler_X"]
    scaler_y     = bundle["scaler_y"]
    feature_cols = bundle["feature_cols"]
    seq_days     = bundle.get("sequence_days", 1)
    model_type   = bundle.get("model_type", "LSTM")

    feat_means = scaler_X.data_min_ + (scaler_X.data_max_ - scaler_X.data_min_) * 0.5

    row = []
    for i, col in enumerate(feature_cols):
        row.append(col_values.get(col, float(feat_means[i])))

    row_arr = np.array(row, dtype=float).reshape(1, -1)
    row_sc  = scaler_X.transform(row_arr)

    try:
        if model_type == "RandomForest":
            rf = bundle.get("rf_model")
            if rf is None:
                return None, "no_rf"
            pred_g = float(rf.predict(row_sc)[0])
            if current_day is not None:
                pred_g = pred_g * growth_sigmoid(current_day)
            return round(max(0.0, pred_g), 1), "rf"

        import tensorflow as tf

        if model_type == "ANN":
            model = tf.keras.models.model_from_config(bundle["model_config"])
            model.set_weights(bundle["model_weights"])
            pred_sc = model.predict(row_sc, verbose=0)
            pred_g  = float(scaler_y.inverse_transform(pred_sc)[0][0])
            if current_day is not None:
                pred_g = pred_g * growth_sigmoid(current_day)
            return round(max(0.0, pred_g), 1), "ann"

        if model_type == "LSTM":
            model = tf.keras.models.model_from_config(bundle["model_config"])
            model.set_weights(bundle["model_weights"])
            SEQUENCE_HOURS = bundle.get("sequence_hours", seq_days) or 24

            if all_days and len(all_days) >= 2:
                seq_rows    = []
                sorted_days = sorted(all_days, key=lambda x: x.get("day", x.get("week", 0)))
                for entry in sorted_days:
                    cv = build_col_values_from_weeks([entry])
                    r  = [cv.get(col, float(feat_means[i])) for i, col in enumerate(feature_cols)]
                    seq_rows.append(r)
                while len(seq_rows) < SEQUENCE_HOURS:
                    seq_rows.insert(0, seq_rows[0])
                seq_rows = seq_rows[-SEQUENCE_HOURS:]
                seq_arr   = np.array(seq_rows, dtype=float)
                seq_sc   = scaler_X.transform(seq_arr)
                seq_in   = seq_sc.reshape(1, SEQUENCE_HOURS, len(feature_cols))
            else:
                seq_in = np.repeat(row_sc, SEQUENCE_HOURS, axis=0).reshape(
                    1, SEQUENCE_HOURS, len(feature_cols))

            pred_sc = model.predict(seq_in, verbose=0)
            pred_g  = float(scaler_y.inverse_transform(pred_sc)[0][0])
            if current_day is not None:
                pred_g = pred_g * growth_sigmoid(current_day)
            return round(max(0.0, pred_g), 1), "lstm"

    except Exception as e:
        print(f"[predict_single] {model_type} error: {e}")
        base   = bundle.get("info", {}).get("predicted_yield_g", 190.0)
        pred_g = float(base)
        if current_day is not None:
            pred_g = pred_g * growth_sigmoid(current_day)
        return round(pred_g, 1), "cached"


def predict_all_models(col_values, all_days=None, current_day=None):
    results = {}
    for name, bundle in model_bundles.items():
        pred_g, method = predict_single(bundle, col_values,
                                        all_days=all_days,
                                        current_day=current_day)
        info = bundle.get("info", {}) if bundle else {}
        results[name] = {
            "predicted_g": pred_g,
            "method":      method,
            "r2":          info.get("r2_train") or info.get("r2") or 0,
            "mae_g":       info.get("mae_g_train") or info.get("mae_g"),
            "rmse_g":      info.get("rmse_g_train") or info.get("rmse_g"),
            "mape":        info.get("mape_train") or info.get("mape"),
            "mape_train":  info.get("mape_train"),
            "r2_train":    info.get("r2_train"),
        }

    available = {k: v for k, v in results.items() if v["predicted_g"] is not None}
    if not available:
        val = -250 + 0.42 * col_values.get("feed_g", 100) + \
               8.0 * col_values.get("larvae_temp", 28) + \
               3.0 * col_values.get("larvae_hum", 72)
        fb = round(max(0.0, val), 1)
        return {
            "LSTM":         {"predicted_g": fb, "method": "fallback", "r2": None},
            "ANN":          {"predicted_g": fb, "method": "fallback", "r2": None},
            "RandomForest": {"predicted_g": fb, "method": "fallback", "r2": None},
            "best_model":   "LSTM",
        }

    best_name = max(available, key=lambda m: available[m].get("r2") or 0)
    results["best_model"] = best_name
    return results


def build_col_values_from_weeks(weeks):
    stage_data = {}
    for w in weeks:
        s = w["stage"]
        if s not in stage_data:
            stage_data[s] = {"temps": [], "hums": [], "feeds": [], "lux": []}
        stage_data[s]["temps"].append(w["avg_temp"])
        stage_data[s]["hums"].append(w["avg_humidity"])
        if w.get("feed_g"):        stage_data[s]["feeds"].append(w["feed_g"])
        if w.get("avg_light_lux"): stage_data[s]["lux"].append(w["avg_light_lux"])

    def avg(lst): return round(sum(lst) / len(lst), 2) if lst else None

    col_values = {}
    lt = avg(stage_data.get("larvae", {}).get("temps", []))
    lh = avg(stage_data.get("larvae", {}).get("hums",  []))
    feeds = stage_data.get("larvae", {}).get("feeds", [])
    fg = feeds[-1] if feeds else None
    pt = avg(stage_data.get("pupa",   {}).get("temps", []))
    ph = avg(stage_data.get("pupa",   {}).get("hums",  []))
    at = avg(stage_data.get("adult",  {}).get("temps", []))
    ah = avg(stage_data.get("adult",  {}).get("hums",  []))
    lx = avg(stage_data.get("adult",  {}).get("lux",    []))

    if lt is not None: col_values["larvae_temp"] = lt
    if lh is not None: col_values["larvae_hum"]  = lh
    if fg is not None: col_values["feed_g"]       = fg
    if pt is not None: col_values["pupa_temp"]    = pt
    if ph is not None: col_values["pupa_hum"]     = ph
    if at is not None: col_values["adult_temp"]   = at
    if ah is not None: col_values["adult_hum"]    = ah
    if lx is not None: col_values["adult_lux"]    = lx
    return col_values


def predict_from_weeks(weeks, current_day=None):
    col_values = build_col_values_from_weeks(weeks)
    return predict_all_models(col_values, all_days=weeks, current_day=current_day)


def recompute_all_day_predictions(cycle):
    all_day_entries = sorted(
        cycle.get("days", []),
        key=lambda x: x.get("day", 0)
    )
    for i, entry in enumerate(all_day_entries):
        d              = entry.get("day", 1)
        entries_so_far = all_day_entries[:i + 1]
        day_preds      = predict_all_models(
            build_col_values_from_weeks(entries_so_far),
            all_days=entries_so_far,
            current_day=d
        )
        entry["all_predictions"] = {
            k: v for k, v in day_preds.items() if k != "best_model"
        }
        best = day_preds.get("best_model", "LSTM")
        entry["predicted_g"] = (day_preds.get(best) or {}).get("predicted_g")
        entry["best_model"]  = best
    return cycle


# ── Serial reader ─────────────────────────────────────────────
def find_port():
    for p in serial.tools.list_ports.comports():
        for kw in ["Arduino", "CH340", "CP210", "USB Serial", "ttyUSB", "ttyACM"]:
            if kw.lower() in p.description.lower() or kw.lower() in p.device.lower():
                return p.device
    return None

def parse_line(line):
    try:
        d     = json.loads(line.strip())
        stage = d.get("stage", "").lower()
        if stage not in sensor_data: return
        with data_lock:
            sensor_data[stage]["temp"]     = round(float(d["temp"]), 1)
            sensor_data[stage]["humidity"] = round(float(d["humidity"]), 1)
            if stage == "adult" and "light" in d:
                sensor_data[stage]["light"] = round(float(d["light"]), 1)
            sensor_data[stage]["timestamp"] = datetime.now().strftime("%H:%M:%S")
    except:
        pass

def serial_thread():
    while True:
        port = find_port()
        if port:
            try:
                # Siguraduhin na 9600 ang baud rate
                ser = serial.Serial(port, 9600, timeout=1)
                serial_status.update({"connected": True, "port": port, "error": None})
                print(f"[SERIAL] Connected to {port}")
                
                while True:
                    if ser.in_waiting > 0:
                        raw_line = ser.readline().decode('utf-8', errors='ignore').strip()
                        if raw_line:
                            try:
                                data = json.loads(raw_line)
                                with data_lock:
                                    # PAGWAWASTO NG MGA NAMES:
                                    # 1. Larvae
                                    if "larvae" in data:
                                        sensor_data["larvae"]["temp"] = data["larvae"].get("temp")
                                        sensor_data["larvae"]["humidity"] = data["larvae"].get("hum") # 'hum' sa Arduino
                                        sensor_data["larvae"]["timestamp"] = datetime.now().strftime("%H:%M:%S")
                                    
                                    # 2. Pupa (Match sa 'pupae' ng Arduino)
                                    if "pupae" in data:
                                        sensor_data["pupa"]["temp"] = data["pupae"].get("temp")
                                        sensor_data["pupa"]["humidity"] = data["pupae"].get("hum")
                                        sensor_data["pupa"]["timestamp"] = datetime.now().strftime("%H:%M:%S")
                                        
                                    # 3. Adult
                                    if "adult" in data:
                                        sensor_data["adult"]["temp"] = data["adult"].get("temp")
                                        sensor_data["adult"]["humidity"] = data["adult"].get("hum")
                                        sensor_data["adult"]["light"] = data["adult"].get("light")
                                        sensor_data["adult"]["timestamp"] = datetime.now().strftime("%H:%M:%S")
                                        
                            except json.JSONDecodeError as je:
                                print(f"[JSON] Error: {je} | Raw: {raw_line}")
                                
            except Exception as e:
                print(f"[SERIAL] Error: {e}")
                serial_status.update({"connected": False, "port": None, "error": str(e)})
                if 'ser' in locals(): ser.close()
        else:
            serial_status.update({"connected": False, "port": None, "error": "No Arduino found"})
        
        time.sleep(5)
# ── Routes ────────────────────────────────────────────────────
@app.route("/")
@login_required
def index():
    return render_template("index.html", model_info=model_info)

@app.route("/api/sensors")
def api_sensors():
    # Siguraduhin na ang sensor_data ay ang dictionary na may adult, larvae, pupae
    return jsonify({"sensors": sensor_data, "serial": serial_status})

@app.route("/api/model_info")
def api_model_info():
    return jsonify(model_info)

@app.route("/api/cycle", methods=["GET"])
def api_cycle_get():
    cid   = str(cycle_data["current_cycle"])
    cycle = cycle_data["cycles"].get(cid, new_cycle(int(cid)))
    cycle = recompute_all_day_predictions(cycle)

    all_entries = cycle.get("days", []) + cycle.get("weeks", [])
    latest      = None

    if all_entries:
        max_day   = max((w.get("day", w.get("week", 1)) for w in all_entries), default=1)
        all_preds = predict_all_models(
            build_col_values_from_weeks(all_entries),
            all_days=all_entries,
            current_day=max_day
        )
        best_name  = all_preds.get("best_model", "LSTM")
        best_pred  = all_preds.get(best_name, {})
        confidence = get_confidence(all_entries)
        latest = {
            "predicted_g":     best_pred.get("predicted_g"),
            "method":          best_pred.get("method"),
            "confidence":      confidence,
            "best_model":      best_name,
            "all_predictions": {k: v for k, v in all_preds.items() if k != "best_model"},
        }

    return jsonify({
        "current_cycle":     cycle_data["current_cycle"],
        "cycle":             cycle,
        "latest_prediction": latest,
    })

@app.route("/api/cycle/week", methods=["POST"])
def api_cycle_week_post():
    b = request.json or {}
    def sf(k):
        v = b.get(k)
        try:   return float(v) if v not in (None, "", "null") else None
        except: return None

    week = b.get("week")
    if week is None: return jsonify({"error": "week is required"}), 400
    try: week = int(week)
    except: return jsonify({"error": "week must be a number"}), 400

    avg_temp = sf("avg_temp")
    avg_hum  = sf("avg_humidity")
    if avg_temp is None or avg_hum is None:
        return jsonify({"error": "avg_temp and avg_humidity are required"}), 400

    cid   = str(cycle_data["current_cycle"])
    cycle = cycle_data["cycles"].setdefault(cid, new_cycle(cycle_data["current_cycle"]))
    if cycle["finished"]: return jsonify({"error": "Finished. Start a new cycle."}), 400

    stage = stage_from_week(week)
    entry = {
        "week": week, "stage": stage, "avg_temp": avg_temp,
        "avg_humidity": avg_hum, "feed_g": sf("feed_g"),
        "avg_light_lux": sf("avg_light_lux"),
        "recorded_at": datetime.now().isoformat(),
    }

    updated = False
    for i, w in enumerate(cycle["weeks"]):
        if w["week"] == week:
            cycle["weeks"][i] = entry
            updated = True
            break
    if not updated:
        cycle["weeks"].append(entry)
        cycle["weeks"].sort(key=lambda x: x["week"])

    all_preds  = predict_from_weeks(cycle["weeks"])
    best_name  = all_preds.get("best_model", "LSTM")
    best_pred  = all_preds.get(best_name, {})
    
    idx = next(i for i, w in enumerate(cycle["weeks"]) if w["week"] == week)
    cycle["weeks"][idx].update({
        "predicted_g": best_pred.get("predicted_g"),
        "confidence": get_confidence(cycle["weeks"]),
        "method": best_pred.get("method"),
        "best_model": best_name,
        "all_predictions": {k: v for k, v in all_preds.items() if k != "best_model"}
    })

    save_cycle_data()
    return jsonify({"status": "updated" if updated else "added", "week": week})

@app.route("/api/cycle/day", methods=["POST"])
def api_cycle_day_post():
    b = request.json or {}
    def sf(k):
        v = b.get(k)
        try:   return float(v) if v not in (None, "", "null") else None
        except: return None

    day = b.get("day")
    if day is None: return jsonify({"error": "day is required"}), 400
    try: day = int(day)
    except: return jsonify({"error": "day must be a number"}), 400

    avg_temp, avg_hum = sf("avg_temp"), sf("avg_humidity")
    if avg_temp is None or avg_hum is None:
        return jsonify({"error": "avg_temp and avg_humidity are required"}), 400

    cid   = str(cycle_data["current_cycle"])
    cycle = cycle_data["cycles"].setdefault(cid, new_cycle(cycle_data["current_cycle"]))
    if cycle["finished"]: return jsonify({"error": "Current cycle is finished."}), 400

    stage = stageFromDay(day)
    entry = {
        "day": day, "stage": stage, "avg_temp": avg_temp,
        "avg_humidity": avg_hum, "feed_g": sf("feed_g"),
        "avg_light_lux": sf("avg_light_lux"),
        "recorded_at": datetime.now().isoformat(),
    }

    if "days" not in cycle: cycle["days"] = []
    updated = False
    for i, d in enumerate(cycle["days"]):
        if d["day"] == day:
            cycle["days"][i] = entry
            updated = True
            break
    if not updated:
        cycle["days"].append(entry)
        cycle["days"].sort(key=lambda x: x["day"])

    cycle = recompute_all_day_predictions(cycle)
    save_cycle_data()
    return jsonify({"status": "updated" if updated else "added", "day": day})

@app.route("/api/cycle/day/<int:day>", methods=["DELETE"])
def api_cycle_day_delete(day):
    cid   = str(cycle_data["current_cycle"])
    cycle = cycle_data["cycles"].get(cid)
    if not cycle or "days" not in cycle: return jsonify({"error": "Not found"}), 404
    cycle["days"] = [d for d in cycle["days"] if d["day"] != day]
    save_cycle_data()
    return jsonify({"status": "deleted", "day": day})

@app.route("/api/cycle/finish", methods=["POST"])
def api_cycle_finish():
    b = request.json or {}
    try: actual_g = float(b["actual_yield_g"])
    except: return jsonify({"error": "actual_yield_g is required"}), 400

    cid   = str(cycle_data["current_cycle"])
    cycle = cycle_data["cycles"].get(cid)
    if not cycle: return jsonify({"error": "No active cycle"}), 404

    all_entries = cycle.get("days", []) + cycle.get("weeks", [])
    pred_g = 0
    if all_entries:
        max_day = max((w.get("day", w.get("week", 1)) for w in all_entries), default=1)
        all_preds = predict_all_models(build_col_values_from_weeks(all_entries), all_days=all_entries, current_day=max_day)
        best_name = all_preds.get("best_model", "LSTM")
        best_pred = all_preds.get(best_name, {})
        pred_g = best_pred.get("predicted_g", 0)

        cycle["finished"] = True
        cycle["actual_yield_g"] = actual_g
        cycle["finished_at"] = datetime.now().isoformat()
        cycle["final_prediction"] = {
            "predicted_g": pred_g,
            "confidence": get_confidence(all_entries),
            "best_model": best_name,
            "error_g": round(abs(actual_g - pred_g), 1),
            "error_pct": round(abs(actual_g - pred_g) / actual_g * 100, 1) if actual_g else 0,
        }

    new_id = cycle_data["current_cycle"] + 1
    cycle_data["current_cycle"] = new_id
    cycle_data["cycles"][str(new_id)] = new_cycle(new_id)
    save_cycle_data()
    return jsonify({"status": "cycle_finished", "new_cycle_id": new_id})

@app.route("/api/cycles/history")
def api_cycles_history():
    finished = [c for c in cycle_data["cycles"].values() if c.get("finished")]
    finished.sort(key=lambda x: x["cycle_id"])
    return jsonify(finished)

@app.route("/api/cycle/restart", methods=["POST"])
def api_cycle_restart():
    global cycle_data
    cycle_data = {"current_cycle": 1, "cycles": {"1": new_cycle(1)}}
    save_cycle_data()
    return jsonify({"status": "reset"})

# ── LOGIN / LOGOUT ────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        hashed   = hashlib.sha256(password.encode()).hexdigest()
        if username in USERS and USERS[username] == hashed:
            login_user(User(username))
            return redirect(url_for("index"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

if __name__ == "__main__":
    # Naka-bind sa 0.0.0.0 para ma-access mo sa phone mo via Local IP
    app.run(debug=True, host="0.0.0.0", port=5000)
