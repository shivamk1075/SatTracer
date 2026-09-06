# VERSION 2
from pathlib import Path
import os
from dotenv import load_dotenv
import json
import requests
import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import folium
from streamlit_folium import st_folium
from geopy.geocoders import ArcGIS
from folium.plugins import MousePosition
from streamlit_searchbox import st_searchbox
import base64  # <-- ADD THIS

# --- 1. PAGE SETUP & STYLING ---
st.set_page_config(page_title="SatTracer | Viability Engine", layout="wide")

# --- NEW: BACKGROUND IMAGE FUNCTION ---
def set_background(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        
        /* 1. Darker overlay for better contrast */
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background-color: rgba(15, 18, 22, 0.85); /* Darkened from 0.7 to 0.85 */
            z-index: -1;
        }}

        /* 2. Larger base font and text-shadow for readability */
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
            font-size: 18px !important; /* Increased base size */
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8); /* Helps text pop off the background */
        }}

        /* 3. Make Main Headers Massive and Clear */
        .main-header {{
            font-weight: 800;
            font-size: 3.5rem !important;
            margin-bottom: -10px;
            color: #ffffff;
            text-shadow: 2px 2px 6px rgba(0,0,0,0.9);
        }}
        .sub-header {{
            color: #e0e0e0;
            font-size: 1.5rem !important;
            font-weight: 500;
            margin-bottom: 25px;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.8);
        }}
        
        /* Force buttons to be completely solid/opaque */
        .stButton > button {{
            background-color: #1E56D0 !important;
            color: #FFFFFF !important;
            opacity: 1 !important;
            border: none !important;
            font-weight: 600 !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.5) !important;
            text-shadow: none !important; /* Remove shadow inside button */
        }}
        
        .stButton > button:hover {{
            background-color: #1542A6 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.warning(f"⚠️ Background image not found at: {image_path}")

# CALL THE FUNCTION HERE:
# Replace the string below with the actual path to your image



# Recommended: resolves relative to current file's directory
BASE_DIR = Path(__file__).resolve().parent.parent
BG_IMAGE_PATH = BASE_DIR / "data" / "bgr.png"

# Or simply use forward slashes:
# BG_IMAGE_PATH = "data/bgr.png"
set_background(BG_IMAGE_PATH)

# --- LOAD ENVIRONMENT VARIABLES ---
load_dotenv()

# Modal API Endpoints
SPATIAL_API_BULK = os.getenv("SPATIAL_API_BULK")
SPATIAL_API_SINGLE = os.getenv("SPATIAL_API_SINGLE")
MODEL_API_BULK = os.getenv("MODEL_API_BULK")
MODEL_API_SINGLE = os.getenv("MODEL_API_SINGLE")

# Optional: Add a quick safety check so the app warns you if the .env file is missing
if not SPATIAL_API_BULK:
    st.error("⚠️ API endpoints are missing! Please check your .env file.")
    st.stop()

BASE_DIR = Path(__file__).resolve().parent.parent
ORDERED_CATEGORIES_PATH = BASE_DIR / "data/ordered_categories.json"

@st.cache_data
def load_categories():
    if ORDERED_CATEGORIES_PATH.exists():
        with ORDERED_CATEGORIES_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    with open("ordered_categories.json", "r", encoding="utf-8") as file:
        return json.load(file)

ordered_categories = load_categories()

@st.cache_data
def load_benchmark_sites():
    benchmark_path = BASE_DIR / "data/benchmark_sites.parquet"
    if benchmark_path.exists():
        df = pd.read_parquet(benchmark_path)
        idx = df.groupby(['lat', 'long'])['viability_score_0_1'].idxmax()
        best_sites = df.loc[idx]
        locs = best_sites[['lat', 'long', 'target_category', 'viability_score_0_1']].to_dict('records')
        return df, locs
    return None, []

benchmark_df, benchmark_locations = load_benchmark_sites()

# --- 2. CORE FUNCTIONS ---
def calibrate_score(raw_score: float) -> float:
    """Scales raw XGBoost predictions to intuitive business ratings."""
    raw_anchors =    [0.05, 0.12, 0.216, 0.269, 0.321, 0.420, 0.550]
    scaled_anchors = [0.05, 0.20, 0.400, 0.600, 0.800, 0.950, 1.000]
    scaled = float(np.interp(raw_score, raw_anchors, scaled_anchors))
    return round(float(np.clip(scaled, 0.0, 1.0)), 4)

def suggest_locations(searchterm: str):
    if not searchterm or len(searchterm) < 3:
        return []
    try:
        geolocator = ArcGIS(timeout=10) 
        locations = geolocator.geocode(searchterm + ", India", exactly_one=False)
        if locations:
            results = []
            for loc in locations[:5]:
                if loc and hasattr(loc, 'address'):
                    results.append((loc.address, (loc.latitude, loc.longitude)))
            return results
        return [("No results found. Try a different spelling.", None)]
    except Exception as e:
        return [(f"⚠️ Error: {str(e)}", None)]

@st.cache_data(show_spinner=False)
def get_address_from_coords(lat, lon):
    """Converts coordinates to a human-readable address."""
    try:
        geolocator = ArcGIS(timeout=5)
        # Reverse geocoding takes a string "lat, lon"
        location = geolocator.reverse(f"{lat}, {lon}")
        if location and hasattr(location, 'address'):
            return location.address
        return "Unknown Location"
    except Exception:
        return "Address unavailable"

# --- 3. SESSION STATE MANAGEMENT ---
default_lat, default_lon = 19.0760, 72.8774 # Default: Delhi

if "map_lat" not in st.session_state: st.session_state.map_lat = default_lat
if "map_lon" not in st.session_state: st.session_state.map_lon = default_lon
if "map_center_lat" not in st.session_state: st.session_state.map_center_lat = default_lat
if "map_center_lon" not in st.session_state: st.session_state.map_center_lon = default_lon
if "zoom_level" not in st.session_state: st.session_state.zoom_level = 16
if "analysis_results" not in st.session_state: st.session_state.analysis_results = None
if "last_processed_click" not in st.session_state: st.session_state.last_processed_click = None
if "last_searched_coords" not in st.session_state: st.session_state.last_searched_coords = None
if "trigger_analysis" not in st.session_state: st.session_state.trigger_analysis = False
if "analysis_params" not in st.session_state: st.session_state.analysis_params = None


# --- 4. HEADER ---
st.markdown("<div class='main-header'>SatTracer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Geospatial Business Viability Engine</div>", unsafe_allow_html=True)


# --- 5. TOP DASHBOARD (MAP & CONTROLS) ---
col_map, col_control = st.columns([1.6, 1], gap="large") # 62% / 38% split looks very professional

with col_control:
    # --- CARD 1: Location Settings ---
    with st.container(border=True):
        st.subheader("Site Selection")
        
        selected_coords = st_searchbox(
            suggest_locations,
            key="location_search",
            placeholder="Search city, area, or landmark..."
        )

        if selected_coords is not None:
            if selected_coords != st.session_state.last_searched_coords:
                lat, lon = selected_coords
                st.session_state.map_lat, st.session_state.map_lon = lat, lon
                st.session_state.map_center_lat, st.session_state.map_center_lon = lat, lon
                st.session_state.zoom_level = 16
                st.session_state.last_searched_coords = selected_coords
                st.rerun()

        # show_examples = st.toggle("Show Example Offline Sites", value=False)
        
        map_style = st.selectbox(
            "Map Style",
            options=["Hybrid (Satellite + Labels)", "Satellite Only", "Standard Roadmap", "Terrain"],
            index=0,
            label_visibility="collapsed" # Cleaner look
        )
        
        style_mapping = {
            "Hybrid (Satellite + Labels)": ("https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", "Google Hybrid"),
            "Satellite Only": ("https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}", "Google Satellite"),
            "Standard Roadmap": ("https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", "Google Maps"),
            "Terrain": ("https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}", "Google Terrain")
        }
        selected_tiles, selected_attr = style_mapping[map_style]

    # --- CARD 2: Scan Parameters ---
    with st.container(border=True):
        st.subheader("Analysis Parameters")
        
        analysis_mode = st.radio(
            "Scope",
            ["Scan (All 1,591 Categories)", "Single Target Category"],
            horizontal=False
        )

        selected_category = None
        if analysis_mode == "Single Target Category":
            # Curated list of 39 viable business types (removed 'none' and 'other')
            TOP_CATEGORIES = [
                "hotel", "school", "hospital", "hindu_temple", "financial_service", "banks", 
                "bank_credit_union", "college_university", "restaurant", "shopping", "gas_station", 
                "atms", "education", "landmark_and_historical_building", "resort", "clothing_store", 
                "mobile_phone_store", "party_and_event_planning", "pharmacy", "car_dealer", 
                "electronics", "credit_union", "beauty_salon", "central_government_office", "gym", 
                "professional_services", "high_school", "indian_restaurant", "jewelry_store", 
                "real_estate_service", "dentist", "preschool", "cafe", "travel_services", 
                "motorcycle_dealer", "hardware_store", "tutoring_center", "diagnostic_services", 
                "fast_food_restaurant"
            ]
            
            selected_category = st.selectbox(
                "Target Category:",
                options=TOP_CATEGORIES,
                index=TOP_CATEGORIES.index("cafe") if "cafe" in TOP_CATEGORIES else 0
            )

    # --- EXECUTION BUTTON ---
    target_lat = st.session_state.map_lat
    target_lon = st.session_state.map_lon
     
    st.info(f"Selected Coordinates: {target_lat:.6f}, {target_lon:.6f}")
    
    if st.button("Execute Viability Analysis", type="primary", use_container_width=True):
        st.session_state.trigger_analysis = True
        st.session_state.analysis_params = {
            "lat": target_lat, "lon": target_lon, 
            "mode": analysis_mode, "category": selected_category
        }

with col_map:
    # --- NEW: Dynamic Address Header ---
    current_address = get_address_from_coords(st.session_state.map_lat, st.session_state.map_lon)
    st.markdown(f"### {current_address}")

    # Wrap map in a container for a subtle border frame
    with st.container(border=True):
        m = folium.Map(
            location=[st.session_state.map_center_lat, st.session_state.map_center_lon],
            zoom_start=st.session_state.zoom_level,
            tiles=selected_tiles,
            attr=selected_attr,
            max_zoom=22 
        )

        m.get_root().header.add_child(folium.Element("""
        <style> .leaflet-container { background: #111418 !important; } </style>
        """))

        folium.Marker(
            [st.session_state.map_lat, st.session_state.map_lon],
            tooltip="Target Location",
            popup=f"Lat: {st.session_state.map_lat:.6f}, Lon: {st.session_state.map_lon:.6f}",
            icon=folium.Icon(color="red", icon="crosshairs", prefix="fa")
        ).add_to(m)

        MousePosition(
            position="bottomright", separator=" | ", empty_string="NaN",
            lng_first=False, prefix="Coordinates:"
        ).add_to(m)

        # if show_examples:
        #     for site in benchmark_locations:
        #         cat = str(site.get('target_category', 'Unknown'))
        #         raw_score = site.get('viability_score_0_1', 0)
        #         calibrated_perc = calibrate_score(raw_score) * 100.0
        #         display_cat = cat.replace("_", " ").title()
                
        #         tooltip_html = f"<b>{display_cat}</b><br>Viability: {calibrated_perc:.2f}%"
        #         popup_html = f"<b>Offline Benchmark Site</b><br>Category: {display_cat}<br>Score: {calibrated_perc:.2f}%"

        #         folium.CircleMarker(
        #             location=[site['lat'], site['long']],
        #             radius=6, color="#0066cc", fill=True, fill_color="#0066cc",
        #             fill_opacity=0.7, tooltip=tooltip_html, popup=popup_html
        #         ).add_to(m)
                
        map_data = st_folium(
            m,
            height=540, 
            use_container_width=True,
            key="sattracer_map_view",
            returned_objects=["last_clicked", "zoom", "center"]
        )

        if map_data:
            clicked = map_data.get("last_clicked")
            if clicked and clicked != st.session_state.last_processed_click:
                st.session_state.map_lat = clicked["lat"]
                st.session_state.map_lon = clicked["lng"]
                
                if map_data.get("zoom"):
                    st.session_state.zoom_level = map_data["zoom"]
                if map_data.get("center"):
                    st.session_state.map_center_lat = map_data["center"]["lat"]
                    st.session_state.map_center_lon = map_data["center"]["lng"]
                
                st.session_state.last_processed_click = clicked
                st.rerun()


# --- 6. EXECUTION PIPELINE (BACKGROUND LOGIC) ---
if st.session_state.get("trigger_analysis") and st.session_state.get("analysis_params"):
    params = st.session_state.analysis_params
    run_lat, run_lon = params["lat"], params["lon"]
    run_mode, run_cat = params["mode"], params["category"]
    
    # Render progress below the main columns
    st.markdown("---")
    
    matched_data = None
    if benchmark_df is not None:
        threshold = 0.0002 
        match = benchmark_df[
            (np.abs(benchmark_df['lat'] - run_lat) < threshold) &
            (np.abs(benchmark_df['long'] - run_lon) < threshold)
        ]
        if not match.empty: matched_data = match

    if matched_data is not None:
        st.toast("Fetched instantly from local dataset.")
        
        if run_mode.startswith("Scan (All"):
            matched_data = matched_data.set_index('target_category').reindex(ordered_categories).reset_index()
            raw_scores = matched_data['viability_score_0_1'].fillna(0).tolist()
            
            spatial_data = {
                "shared_environment": matched_data.iloc[0].to_dict(),
                "competitors_array": matched_data[['comp_count_100m', 'comp_count_300m', 'comp_count_1000m', 'comp_count_5000m', 'nearest_comp_dist', 'comp_gravity_score']].fillna(0).values.tolist()
            }
            
            st.session_state.analysis_results = {
                "mode": "bulk", "spatial_data": spatial_data,
                "calibrated_percentages": [calibrate_score(s) * 100.0 for s in raw_scores]
            }
        else:
            row = matched_data[matched_data['target_category'] == run_cat]
            if not row.empty:
                feat_dict = row.iloc[0].to_dict()
                st.session_state.analysis_results = {
                    "mode": "single", "category": run_cat, "features": feat_dict,
                    "calibrated_percentage": calibrate_score(feat_dict.get('viability_score_0_1', 0)) * 100.0
                }
        st.session_state.trigger_analysis = False

    else:
        if run_mode.startswith("Scan (All"):
            with st.status("Executing Geospatial Analysis Pipeline...", expanded=True) as status:
                st.write("Extracting spatial features and competitor distribution...")
                try:
                    spatial_res = requests.post(SPATIAL_API_BULK, json={"lat": run_lat, "lon": run_lon}, timeout=180)
                    spatial_res.raise_for_status()
                    spatial_data = spatial_res.json()
                except Exception as e:
                    status.update(label="Spatial Service Error", state="error")
                    st.error(f"Error: {e}")
                    st.session_state.trigger_analysis = False
                    st.stop()
    
                st.write("Evaluating category suitability via ML model...")
                try:
                    model_res = requests.post(
                        MODEL_API_BULK,
                        json={"shared_environment": spatial_data["shared_environment"], "competitors_array": spatial_data["competitors_array"]},
                        timeout=180
                    )
                    model_res.raise_for_status()
                    raw_scores = model_res.json()["scores"]
                except Exception as e:
                    status.update(label="Model Service Error", state="error")
                    st.error(f"Error: {e}")
                    st.session_state.trigger_analysis = False
                    st.stop()
    
                status.update(label="Analysis Complete", state="complete", expanded=False)
    
            st.session_state.analysis_results = {
                "mode": "bulk", "spatial_data": spatial_data,
                "calibrated_percentages": [calibrate_score(s) * 100.0 for s in raw_scores]
            }
            st.session_state.trigger_analysis = False
    
        else:
            with st.status(f"Evaluating Viability for '{run_cat}'...", expanded=True) as status:
                st.write("Extracting localized spatial context...")
                try:
                    spatial_res = requests.post(SPATIAL_API_SINGLE, json={"lat": run_lat, "lon": run_lon, "category": run_cat}, timeout=180)
                    spatial_res.raise_for_status()
                    feat_dict = spatial_res.json()["features"]
                except Exception as e:
                    status.update(label="Spatial Service Error", state="error")
                    st.error(f"Error: {e}")
                    st.session_state.trigger_analysis = False
                    st.stop()
    
                st.write("Computing viability index...")
                try:
                    model_res = requests.post(MODEL_API_SINGLE, json={"features": feat_dict}, timeout=180)
                    model_res.raise_for_status()
                    raw_score = model_res.json()["viability_score"]
                except Exception as e:
                    status.update(label="Model Service Error", state="error")
                    st.error(f"Error: {e}")
                    st.session_state.trigger_analysis = False
                    st.stop()
    
                status.update(label="Analysis Complete", state="complete", expanded=False)
    
            st.session_state.analysis_results = {
                "mode": "single", "category": run_cat, "features": feat_dict,
                "calibrated_percentage": calibrate_score(raw_score) * 100.0
            }
            st.session_state.trigger_analysis = False

# --- 7. FULL WIDTH RESULTS DASHBOARD ---
if st.session_state.analysis_results:
    st.markdown("---")
    st.header("Viability Analysis Results")
    
    res = st.session_state.analysis_results

    if res["mode"] == "bulk":
        comp_cols = ["comp_count_100m", "comp_count_300m", "comp_count_1000m", "comp_count_5000m", "nearest_comp_dist", "comp_gravity_score"]
        comp_df = pd.DataFrame(res["spatial_data"]["competitors_array"], columns=comp_cols)
        
        df_results = pd.DataFrame({
            "Category": ordered_categories,
            "Display Name": [c.replace("_", " ").title() for c in ordered_categories],
            "Viability Score (%)": res["calibrated_percentages"]
        })

        for col in comp_cols: df_results[col] = comp_df[col]
        # for key, val in res["spatial_data"]["shared_environment"].items(): df_results[key] = val
        for key, val in res["spatial_data"]["shared_environment"].items(): 
            if key != "nearest_road_surface":
                df_results[key] = val

        df_ranked = df_results.sort_values(by="Viability Score (%)", ascending=False).reset_index(drop=True)
        df_ranked.insert(0, "Rank", df_ranked.index + 1)
        top_row = df_ranked.iloc[0]

        # Callout card for the #1 result
        with st.container(border=True):
            st.metric(
                label=f"Top Recommended Business Concept",
                value=f"{top_row['Display Name']}",
                delta=f"{top_row['Viability Score (%)']:.1f}% Match Score",
                delta_color="normal"
            )

        tab_prominent, tab_catalog, tab_environment = st.tabs(["Top Opportunities", "Full Catalog", "Base Environment Context"])

        with tab_prominent:
            top_15 = df_ranked.head(15)[["Rank", "Display Name", "Viability Score (%)", "nearest_comp_dist", "comp_count_300m"]].copy()
            top_15.rename(columns={"Display Name": "Business Category", "nearest_comp_dist": "Nearest Competitor (m)", "comp_count_300m": "Competitors (300m)"}, inplace=True)
            
            st.dataframe(
                top_15,
                column_config={
                    "Viability Score (%)": st.column_config.ProgressColumn("Viability Score", format="%.2f%%", min_value=0, max_value=100)
                },
                use_container_width=True, hide_index=True, height=550
            )

        with tab_catalog:
            catalog_export_cols = [c for c in df_ranked.columns if c != "Display Name"]
            st.dataframe(
                df_ranked[catalog_export_cols],
                column_config={"Viability Score (%)": st.column_config.NumberColumn("Viability Score (%)", format="%.2f%%")},
                use_container_width=True, hide_index=True
            )
            
        # with tab_environment:
        #     env_dict = res["spatial_data"]["shared_environment"]
        #     features_summary_df = pd.DataFrame([
        #         {"Feature": k.replace("_", " ").title(), "Value": str(v)} for k, v in env_dict.items()
        #     ])
        #     st.dataframe(features_summary_df, use_container_width=True, hide_index=True)
        with tab_environment:
            env_dict = res["spatial_data"]["shared_environment"]
            features_summary_df = pd.DataFrame([
                {"Feature": k.replace("_", " ").title(), "Value": str(v)} 
                for k, v in env_dict.items() if k != "nearest_road_surface"
            ])
            st.dataframe(features_summary_df, use_container_width=True, hide_index=True)

    elif res["mode"] == "single":
        st.subheader(f"Concept: {res['category'].replace('_', ' ').title()}")

        # Metric Cards Layout
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            with st.container(border=True):
                st.metric("Viability Index", f"{res['calibrated_percentage']:.1f}%")
        with mc2:
            with st.container(border=True):
                st.metric("Competitors (300m Radius)", res["features"].get("comp_count_300m", 0))
        with mc3:
            with st.container(border=True):
                st.metric("Nearest Competitor", f"{res['features'].get('nearest_comp_dist', 10000.0):.1f} m")

        # # Dynamic Insights
        # val = res["calibrated_percentage"]
        # if val >= 75.0:
        #     st.success("**High Viability:** Strong synergy indicators and favorable competitive spacing detected.")
        # elif val >= 45.0:
        #     st.warning("**Moderate Viability:** Balanced competition. Requires strategic differentiation.")
        # else:
        #     st.error("**Low Viability:** Dense competitor gravity or insufficient synergy anchors detected.")

        # Dynamic Insights
        f = res["features"]
        def get_f(snake, title, default=0):
            return f.get(snake, f.get(title, default))

        # Extract key drivers
        c_100m = get_f("comp_count_100m", "Comp Count 100M")
        c_300m = get_f("comp_count_300m", "Comp Count 300M")
        nearest = get_f("nearest_comp_dist", "Nearest Comp Dist", 1000)
        corner = get_f("corner_lot_indicator", "Corner Lot Indicator")
        junctions = get_f("junction_density_300m", "Junction Density 300M")
        shopping = get_f("synergy_shopping_300m", "Synergy Shopping 300M")

        val = res["calibrated_percentage"]
        
        if val >= 75.0:
            if corner == 1:
                st.success("**High Viability:** Strong synergy indicators, heavily aided by prime corner lot visibility and multi-directional access.")
            elif nearest > 400:
                st.success(f"**High Viability:** Strong localized demand bolstered by a first-mover advantage (nearest competitor is {int(nearest)}m away).")
            elif junctions > 50 or shopping > 20:
                st.success("**High Viability:** Exceptionally high transit density and retail synergy anchors offset local competition.")
            else:
                st.success("**High Viability:** Strong synergy indicators and favorable competitive spacing detected.")
                
        elif val >= 45.0:
            if c_300m >= 4:
                st.warning(f"**Moderate Viability:** Solid baseline demand, but heavily contested by {int(c_300m)} immediate competitors. Strategic differentiation is required.")
            elif junctions < 30 and corner == 0:
                st.warning("**Moderate Viability:** Manageable competition, but lower transit density means relying heavily on destination marketing rather than organic footfall.")
            else:
                st.warning("**Moderate Viability:** Balanced competition and standard synergy. Requires solid execution and strategic positioning.")
                
        else:
            if c_100m >= 3:
                st.error(f"**Low Viability:** Severe hyper-local saturation. {int(c_100m)} direct competitors within just 100m create intense gravity.")
            elif shopping < 5 and junctions < 20:
                st.error("**Low Viability:** Insufficient retail synergy and low transit density make organic customer acquisition highly difficult.")
            else:
                st.error("**Low Viability:** Dense competitor gravity or insufficient synergy anchors detected.")

        # with st.expander("View Complete Feature Vector (ML Inputs)"):
        #     single_feat_df = pd.DataFrame([
        #         {"Feature": k.replace("_", " ").title(), "Value": str(v)} 
        #         for k, v in res["features"].items() if k != "target_category"
        #     ])
        #     st.dataframe(single_feat_df, use_container_width=True, hide_index=True)
        with st.expander("View Complete Feature Vector (ML Inputs)"):
            single_feat_df = pd.DataFrame([
                {"Feature": k.replace("_", " ").title(), "Value": str(v)} 
                for k, v in res["features"].items() if k not in ["target_category", "nearest_road_surface"]
            ])
            st.dataframe(single_feat_df, use_container_width=True, hide_index=True)