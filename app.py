import os
import json
import math
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import xgboost as xgb
from scipy.spatial import KDTree
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from shapely.geometry import Point

app = FastAPI(title="Web GIS Landslide Susceptibility Engine")

@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(self)"
    return response

# ==============================================================================
# 1. LOAD MODEL & DATA SPASIAL SAAT STARTUP
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

model_lr = joblib.load(os.path.join(MODEL_DIR, "landslide_model_lr.pkl"))
model_svm = joblib.load(os.path.join(MODEL_DIR, "landslide_model_svm.pkl"))
model_rf = joblib.load(os.path.join(MODEL_DIR, "landslide_model_rf.pkl"))
model_xgb = xgb.XGBClassifier()
model_xgb.load_model(os.path.join(MODEL_DIR, "landslide_model_xgb.json"))
scaler = joblib.load(os.path.join(MODEL_DIR, "spatial_scaler.pkl"))

df_clean = pd.read_csv(os.path.join(DATA_DIR, "dataset_landslide_corridor_clean.csv"))

# Deteksi CRS otomatis (WGS84 vs UTM 48S)
sample_x = float(df_clean["X"].dropna().iloc[0])
source_crs = "EPSG:4326" if -180.0 <= sample_x <= 180.0 else "EPSG:32748"

gdf_pts = gpd.GeoDataFrame(
    df_clean,
    geometry=gpd.points_from_xy(df_clean.X, df_clean.Y),
    crs=source_crs,
).to_crs("EPSG:4326")

gdf_pts["latitude"] = gdf_pts.geometry.y
gdf_pts["longitude"] = gdf_pts.geometry.x

# KD-Tree untuk Interpolasi IDW 5-Piksel
spatial_tree = KDTree(gdf_pts[["latitude", "longitude"]].values)

# Pre-compute probabilitas seluruh node & quantile cutoff
aspect_rad = np.radians(gdf_pts["aspect"])
X_pts_prep = pd.DataFrame({
    "elevation": gdf_pts["elevation"],
    "slope": gdf_pts["slope"],
    "rainfall": gdf_pts["rainfall"],
    "aspect_sin": np.sin(aspect_rad),
    "aspect_cos": np.cos(aspect_rad),
})
X_pts_scaled = pd.DataFrame(scaler.transform(X_pts_prep), columns=X_pts_prep.columns)

gdf_pts["prob_lr"] = model_lr.predict_proba(X_pts_scaled)[:, 1]
gdf_pts["prob_xgb"] = model_xgb.predict_proba(X_pts_prep)[:, 1]

q20, q40, q60, q80 = np.quantile(gdf_pts["prob_xgb"], [0.20, 0.40, 0.60, 0.80])

# Load Vektor Jalan Provinsi (mendukung .geojson maupun .shp)
road_file = os.path.join(DATA_DIR, "jalan_provinsi.geojson")
if not os.path.exists(road_file):
    road_file = os.path.join(DATA_DIR, "shapefile_igt_2021_jalan_daerah_v2/data_1793852851_7e866142.shp")

gdf_jalan = gpd.read_file(road_file).to_crs("EPSG:4326")
for col in gdf_jalan.columns:
    if col != "geometry":
        gdf_jalan[col] = gdf_jalan[col].astype(str)
gdf_jalan["geometry"] = gdf_jalan.geometry.simplify(0.001)
ROADS_GEOJSON = json.loads(gdf_jalan.to_json())

wilayah_file = os.path.join(
    DATA_DIR,
    "batas_wilayah_jawa_barat",
    "batas_wilayah_jawa_barat.shp",
)
gdf_wilayah = gpd.read_file(wilayah_file)
wilayah_bounds = gdf_wilayah.total_bounds
coordinates_are_wgs84 = (
    -180 <= wilayah_bounds[0] <= 180
    and -180 <= wilayah_bounds[2] <= 180
    and -90 <= wilayah_bounds[1] <= 90
    and -90 <= wilayah_bounds[3] <= 90
)
if coordinates_are_wgs84:
    gdf_wilayah = gdf_wilayah.set_crs("EPSG:4326", allow_override=True)
else:
    gdf_wilayah = gdf_wilayah.to_crs("EPSG:4326")
gdf_wilayah["geometry"] = gdf_wilayah.geometry.simplify(
    tolerance=0.002,
    preserve_topology=True,
)
WILAYAH_GEOJSON = json.loads(gdf_wilayah.to_json())
WILAYAH_UNION = gdf_wilayah.geometry.union_all()

MAP_NODES_CACHE = None
WILAYAH_CACHE = None


def classify_risk(prob: float):
    pct = round(float(prob) * 100, 2)
    if prob <= q20:
        return {"color": "#2ECC71", "kelas": "Sangat Rendah", "status": "AMAN", "prob_pct": pct}
    elif prob <= q40:
        return {"color": "#27AE60", "kelas": "Rendah", "status": "STABIL", "prob_pct": pct}
    elif prob <= q60:
        return {"color": "#F1C40F", "kelas": "Sedang", "status": "WASPADA", "prob_pct": pct}
    elif prob <= q80:
        return {"color": "#E67E22", "kelas": "Tinggi", "status": "SIAGA", "prob_pct": pct}
    else:
        return {"color": "#E74C3C", "kelas": "Sangat Tinggi", "status": "AWAS", "prob_pct": pct}


# ==============================================================================
# 2. API ENDPOINTS UNTUK FRONTEND STITCH
# ==============================================================================
class CoordRequest(BaseModel):
    lat: float
    lon: float

    class Config:
        extra = "forbid"

class PredictRequest(BaseModel):
    elevation: float
    slope: float
    rainfall: float
    aspect: float

    class Config:
        extra = "forbid"


@app.get("/api/layers")
def get_map_layers():
    """Mengirim 632 node LR dan vektor jalan provinsi ke Leaflet."""
    global MAP_NODES_CACHE
    if MAP_NODES_CACHE is None:
        nodes = []
        for idx, r in gdf_pts.iterrows():
            risk = classify_risk(r["prob_lr"])
            nodes.append({
                "idx": int(idx),
                "lat": round(float(r["latitude"]), 6),
                "lon": round(float(r["longitude"]), 6),
                "road": f"Koridor Jalan Provinsi (Grid #{idx})",
                "elev": round(float(r["elevation"]), 2),
                "slope": round(float(r["slope"]), 2),
                "rain": round(float(r["rainfall"]), 2),
                "aspect": round(float(r["aspect"]), 2),
                "lrProb": risk["prob_pct"],
                "status": risk["status"],
                "cls": risk["kelas"],
                "color": risk["color"],
            })
        MAP_NODES_CACHE = {"nodes": nodes, "roads": ROADS_GEOJSON}
    return JSONResponse(
        content=MAP_NODES_CACHE,
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/layers/wilayah-jawa-barat")
def get_wilayah_layer():
    """Mengirim batas wilayah Jawa Barat dalam koordinat WGS84."""
    global WILAYAH_CACHE
    if WILAYAH_CACHE is None:
        WILAYAH_CACHE = WILAYAH_GEOJSON
    return JSONResponse(
        content=WILAYAH_CACHE,
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/api/node/{point_id}")
def get_node_by_index(point_id: int):
    if point_id < 0 or point_id >= len(gdf_pts):
        raise HTTPException(status_code=404, detail="Index titik di luar jangkauan")
    r = gdf_pts.iloc[point_id]
    return {
        "idx": point_id,
        "lat": round(float(r["latitude"]), 5),
        "lon": round(float(r["longitude"]), 5),
        "elevation": round(float(r["elevation"]), 2),
        "slope": round(float(r["slope"]), 2),
        "rainfall": round(float(r["rainfall"]), 2),
        "aspect": round(float(r["aspect"]), 2),
        "status": f"Data terisi dari Index #{point_id} ({r['latitude']:.4f}, {r['longitude']:.4f})",
    }


@app.post("/api/idw")
def interpolate_location(req: CoordRequest):
    if not math.isfinite(req.lat) or not (-90 <= req.lat <= 90):
        raise HTTPException(
            status_code=422,
            detail="Koordinat tidak valid. Pastikan Latitude berada pada rentang -90 sampai 90.",
        )
    if not math.isfinite(req.lon) or not (-180 <= req.lon <= 180):
        raise HTTPException(
            status_code=422,
            detail="Koordinat tidak valid. Pastikan Longitude berada pada rentang -180 sampai 180.",
        )

    if not WILAYAH_UNION.covers(Point(req.lon, req.lat)):
        raise HTTPException(
            status_code=422,
            detail="Titik yang Anda pilih berada di luar batas administrasi Provinsi Jawa Barat. Silakan pilih lokasi lain di dalam wilayah Jawa Barat untuk melanjutkan ekstraksi data.",
        )

    dists, indices = spatial_tree.query([req.lat, req.lon], k=5)
    dists = np.maximum(dists, 1e-6)
    weights = (1.0 / (dists ** 2))
    weights /= np.sum(weights)

    sampled = gdf_pts.iloc[indices]
    elev = float(np.sum(sampled["elevation"].values * weights))
    slp = float(np.sum(sampled["slope"].values * weights))
    rain = float(np.sum(sampled["rainfall"].values * weights))

    asp_rad = np.radians(sampled["aspect"].values)
    asp = float(np.degrees(np.arctan2(
        np.sum(np.sin(asp_rad) * weights),
        np.sum(np.cos(asp_rad) * weights)
    )) % 360)

    dist_km = dists[0] * 111.0
    return {
        "elevation": round(elev, 2),
        "slope": round(slp, 2),
        "rainfall": round(rain, 2),
        "aspect": round(asp, 2),
        "status": f"Ter-interpolasi IDW 5-Piksel (Jarak acuan terdekat: {dist_km:.2f} km)",
    }


@app.post("/api/predict")
def predict_multimodel(req: PredictRequest):
    values = {
        "elevation": (req.elevation, -500.0, 10000.0),
        "slope": (req.slope, 0.0, 90.0),
        "rainfall": (req.rainfall, 0.0, 100000.0),
        "aspect": (req.aspect, 0.0, 360.0),
    }
    for name, (value, minimum, maximum) in values.items():
        if not math.isfinite(value) or not (minimum <= value <= maximum):
            raise HTTPException(status_code=422, detail=f"Nilai {name} di luar batas valid")
    asp_rad = np.radians(req.aspect)
    input_raw = pd.DataFrame([{
        "elevation": req.elevation,
        "slope": req.slope,
        "rainfall": req.rainfall,
        "aspect_sin": np.sin(asp_rad),
        "aspect_cos": np.cos(asp_rad),
    }])
    input_scaled = pd.DataFrame(scaler.transform(input_raw), columns=input_raw.columns)

    models = [
        ("Logistic Regression", model_lr.predict_proba(input_scaled)[0][1]),
        ("Support Vector Machine", model_svm.predict_proba(input_scaled)[0][1]),
        ("Random Forest", model_rf.predict_proba(input_raw)[0][1]),
        ("XGBoost", model_xgb.predict_proba(input_raw)[0][1]),
    ]

    comparison = []
    for name, prob in models:
        info = classify_risk(prob)
        comparison.append({
            "model": name,
            "prob_pct": info["prob_pct"],
            "kelas": info["kelas"],
            "status": info["status"],
        })

    lr_info = comparison[0]
    dss_status = lr_info["status"]
    dss_color = classify_risk(models[0][1])["color"]

    if dss_status == "AMAN" or dss_status == "STABIL":
        dss = "Kondisi lereng tergolong stabil berdasarkan Logistic Regression. Pemeliharaan rutin dan pemantauan drainase tetap diperlukan."
    elif dss_status == "WASPADA":
        dss = "Potensi pergerakan tanah sedang berdasarkan Logistic Regression. Diperlukan penataan vegetasi akar dalam dan perbaikan lereng minor."
    elif dss_status == "SIAGA":
        dss = "Prioritas Tinggi berdasarkan Logistic Regression! Rekomendasi perkuatan lereng (shotcrete / soil nailing) & pembersihan drainase atas lereng."
    else:
        dss = "PRIORITAS UTAMA DINDING PENAHAN TANAH (RETAINING WALL)! Berdasarkan Logistic Regression, risiko kelongsoran sangat tinggi dan mengancam bahu jalan."

    return {"models": comparison, "dss": dss, "dss_status": dss_status, "dss_color": dss_color}


# Serve Frontend Stitch di root "/"
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


@app.get("/docs/terms")
def get_terms_document():
    document_path = os.path.join(BASE_DIR, "docs", "toc.txt")
    with open(document_path, "r", encoding="utf-8") as document:
        content = document.read()
    return Response(content=content, media_type="text/plain")


@app.get("/docs/privacy")
def get_privacy_document():
    document_path = os.path.join(BASE_DIR, "docs", "privacy.txt")
    with open(document_path, "r", encoding="utf-8") as document:
        content = document.read()
    return Response(content=content, media_type="text/plain")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(
        os.path.join(BASE_DIR, "static", "favicon.ico"),
        media_type="image/x-icon",
    )


@app.get("/")
def serve_ui():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))