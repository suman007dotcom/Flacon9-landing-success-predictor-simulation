import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Falcon 9 Landing Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; color: #e6edf3; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22; }
    [data-testid="stSidebar"] * { color: #e6edf3 !important; }

    /* Section headers */
    h1 { color: #58a6ff !important; }
    h2 { color: #79c0ff !important; }
    h3 { color: #a5d6ff !important; }
    
    /* Cards */
    .card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
    .card-success {
        border-left: 4px solid #3fb950;
    }
    .card-fail {
        border-left: 4px solid #f85149;
    }
    .card-info {
        border-left: 4px solid #58a6ff;
    }
    
    /* Result banner */
    .result-success {
        background: linear-gradient(135deg, #0d2818, #1a4a2e);
        border: 2px solid #3fb950;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        color: #3fb950;
    }
    .result-fail {
        background: linear-gradient(135deg, #2d0d0d, #4a1a1a);
        border: 2px solid #f85149;
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        font-size: 1.4rem;
        font-weight: bold;
        color: #f85149;
    }

    /* Feature tag chips */
    .chip {
        display: inline-block;
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 0.85rem;
        color: #79c0ff;
    }

    /* Metric number */
    .big-number {
        font-size: 2.5rem;
        font-weight: bold;
        color: #58a6ff;
    }
    
    /* Input labels */
    label { color: #c9d1d9 !important; }
    
    /* Streamlit widget overrides */
    .stSlider > div > div > div > div { background: #58a6ff; }
    .stSelectbox > div > div { background: #21262d; color: #e6edf3; border-color: #30363d; }
    .stRadio > div { background: transparent; }

    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  DATA & MODEL (FINAL – CLEAN + CONSISTENT)
#@st.cache_resource
@st.cache_data
def load_data():
    df = pd.read_csv("SpaceX_Falcon9.csv")  # or your actual file name
    
    # Convert Date column
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["LandingSuccess"] = df["Outcome"].apply(
    lambda x: 1 if str(x).startswith("True") else 0
)
    
    return df
@st.cache_resource
def train_models(df):

    # ── Copy ───────────────────────────────
    df2 = df.copy()
    df2.columns = df2.columns.str.strip()

    # ── Target creation (Outcome → LandingSuccess) ──
    df2["LandingSuccess"] = df2["Outcome"].apply(
        lambda x: 1 if str(x).startswith("True") else 0
    )

    # ── Handle missing values ───────────────
    df2["PayloadMass"] = df2["PayloadMass"].fillna(df2["PayloadMass"].median())
    df2["Flights"] = df2["Flights"].fillna(df2["Flights"].median())

    # ── Encoding ───────────────────────────
    df2 = pd.get_dummies(
        df2,
        columns=["BoosterVersion", "LaunchSite", "Orbit"],
        drop_first=True
    )

    # ── Drop unused columns ────────────────
    df2 = df2.drop(
        ["Unnamed: 0", "Outcome", "LandingPad", "Serial", "Date"],
        axis=1,
        errors="ignore"
    )

    # ── Features / Target ──────────────────
    X = df2.drop("LandingSuccess", axis=1)
    y = df2["LandingSuccess"]

    # ── Train-test split ───────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── Scaling (ONLY after split → no leakage) ──
    scaler = StandardScaler()

    num_cols = ["PayloadMass", "Flights"]
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

    # ── Final NaN safety ───────────────────
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_test.median())

    # ── Feature importance (for selection) ─

    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_temp.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_temp.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    top_features = importance_df.head(7)["Feature"].tolist()

    # ── Reduce features ────────────────────
    X_train = X_train[top_features]
    X_test = X_test[top_features]

    # ── Models ─────────────────────────────

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # ── Predictions ─────────────────────────

    y_pred_lr = lr.predict(X_test)
    y_pred_rf = rf.predict(X_test)

    lr_acc = accuracy_score(y_test, y_pred_lr)
    rf_acc = accuracy_score(y_test, y_pred_rf)

    # ── Best model selection ───────────────
    if rf_acc >= lr_acc:
        final_model = rf
        model_name = "Random Forest"
    else:
        final_model = lr
        model_name = "Logistic Regression"

    return {
        "lr": lr,
        "rf": rf,
        "final": final_model,
        "model_name": model_name,
        "lr_acc": lr_acc,
        "rf_acc": rf_acc,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_lr": y_pred_lr,
        "y_pred_rf": y_pred_rf,
        "feature_cols": top_features,
        "scaler": scaler,
        "importance_df": importance_df
    }
# ── RUN ─────────────────────────
df = load_data()
models = train_models(df)
# ─────────────────────────────────────────────
#  SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Falcon 9 Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate to:",
        ["Home", "Feature Guide", "Simulate a Launch"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(f"**Dataset:** 90 launches")
    st.markdown(f"**Best Model:** {models['model_name']}")
    st.markdown(f"**Accuracy:** {max(models['lr_acc'], models['rf_acc']):.0%}")
    st.markdown("---")
    st.caption("Built with Streamlit · SpaceX Dataset")


# ═══════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════
if page == "Home":

    st.title("Will the Falcon 9 Booster Land?")
    st.markdown(
        """
        SpaceX's **Falcon 9** rocket is special, it tries to land its booster back after launch 
        so it can be reused. This saves millions of dollars! 
        
        This app uses machine learning to predict: will the booster land safely or crash?
        """
    )

    st.markdown("---")

    # ── Model accuracy strip ────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Logistic Regression", f"{models['lr_acc']:.0%}", "accuracy")
    with c2:
        st.metric("Random Forest", f"{models['rf_acc']:.0%}", "accuracy")
    with c3:
        winner = models["model_name"]
        st.metric("Best Model", winner, "used for predictions")

    st.markdown("---")

    # ── 3 Real Launch Examples ──────────────────────────────
    st.subheader("3 Real Launches from the Dataset")
    st.markdown("Here are three actual missions — let's see what the rocket did!")

    examples = [
        {
            "title": "🛰️ Dragon Demo (2012)",
            "flight": 2, "date": "May 22, 2012",
            "payload": "525 kg", "orbit": "LEO (Low Earth Orbit)",
            "grid_fins": "No", "legs": "No", "flights": 1,
            "outcome": "Failure ❌",
            "why": "Early days — SpaceX hadn't added landing legs yet. No controlled landing attempt.",
            "type": "fail",
        },
        {
            "title": "📦 CRS-3 Supply Mission (2014)",
            "flight": 7, "date": "April 18, 2014",
            "payload": "2,296 kg", "orbit": "ISS (Space Station)",
            "grid_fins": "No", "legs": "Yes", "flights": 1,
            "outcome": "Success ✅",
            "why": "First time legs were used! Landed on ocean — an experimental but successful controlled descent.",
            "type": "success",
        },
        {
            "title": "🛸 GPS Mission (2020)",
            "flight": 90, "date": "Nov 5, 2020",
            "payload": "3,681 kg", "orbit": "MEO (Medium Earth Orbit)",
            "grid_fins": "Yes", "legs": "Yes", "flights": 1,
            "outcome": "Success ✅",
            "why": "Grid fins + legs + experienced team = textbook landing on drone ship.",
            "type": "success",
        },
    ]

    cols = st.columns(3)
    for col, ex in zip(cols, examples):
        card_class = "card-success" if ex["type"] == "success" else "card-fail"
        with col:
            st.markdown(
                f"""
                <div class="card {card_class}">
                  <strong style="font-size:1.05rem">{ex['title']}</strong><br>
                  <span style="color:#8b949e">📅 {ex['date']}</span><br><br>
                  Payload: <b>{ex['payload']}</b><br>
                  Orbit: <b>{ex['orbit']}</b><br>
                  Grid Fins: <b>{ex['grid_fins']}</b><br>
                  Landing Legs: <b>{ex['legs']}</b><br>
                  Times flown: <b>{ex['flights']}</b><br><br>
                  <b>Result: {ex['outcome']}</b><br>
                  <span style="color:#8b949e;font-size:0.85rem">💡 {ex['why']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Actual vs Predicted Chart ───────────────────────────
    st.subheader("How Well Does the Model Predict?")
    st.markdown(
        "The chart below shows the **test set** (launches the model never trained on). "
        "Blue dots = actual results. Orange dots = what our model predicted."
    )

    y_test_vals = models["y_test"].values
    y_pred_vals = models["y_pred_rf"]
    x_axis = list(range(len(y_test_vals)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_axis, y=y_test_vals + 0.05,
        mode="markers", name="Actual",
        marker=dict(symbol="circle", size=12, color="#58a6ff"),
    ))
    fig.add_trace(go.Scatter(
        x=x_axis, y=y_pred_vals - 0.05,
        mode="markers", name="Predicted",
        marker=dict(symbol="x", size=12, color="#f0883e"),
    ))
    fig.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#e6edf3",
        xaxis_title="Launch Sample Index",
        yaxis=dict(
            tickvals=[0, 1], ticktext=["💥 Fail", "✅ Success"],
            color="#e6edf3",
        ),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
        height=320,
        margin=dict(l=20, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "When the dots line up (blue ≈ orange at the same position), the model got it right. "
        "Any mismatch = a prediction error."
    )

    # ── Success rate over years ─────────────────────────────
    st.markdown("---")
    st.subheader("📅 Success Rate Over the Years")
    df_year = df.copy()
    df_year["Year"] = df_year["Date"].dt.year
    yearly = df_year.groupby("Year")["LandingSuccess"].agg(["mean", "count"]).reset_index()
    yearly.columns = ["Year", "SuccessRate", "Launches"]
    yearly["SuccessRate"] = (yearly["SuccessRate"] * 100).round(1)

    fig2 = px.bar(
        yearly, x="Year", y="SuccessRate",
        text="SuccessRate",
        color="SuccessRate",
        color_continuous_scale=["#f85149", "#e3b341", "#3fb950"],
        labels={"SuccessRate": "Success Rate (%)"},
    )
    fig2.update_traces(texttemplate="%{text}%", textposition="outside")
    fig2.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#e6edf3",
        coloraxis_showscale=False,
        height=300,
        margin=dict(l=20, r=20, t=20, b=40),
    )
    st.plotly_chart(fig2, width="stretch")
    st.caption("SpaceX got dramatically better at landing over time — from 0% in 2010 to near 100% by 2020.")


# ═══════════════════════════════════════════════════════════
#  PAGE 2 — FEATURE GUIDE
# ═══════════════════════════════════════════════════════════
elif page == "Feature Guide":

    st.title("What Do All These Terms Mean?")
    st.markdown(
        "Before you simulate a launch, it helps to know *what each input actually means* "
        "and *why it matters* for landing. Here's a plain-English breakdown."
    )

    features = [
        {
            "emoji": "🪨", "name": "Payload Mass",
            "simple": "How heavy is the cargo being launched?",
            "why": "Heavier cargo means the rocket uses more fuel going up. "
                   "That leaves less fuel for the booster to slow itself down during landing. "
                   "So heavier payloads → harder landing.",
            "range": "525 kg – 15,600 kg in this dataset",
            "type": "number",
        },
        {
            "emoji": "🔁", "name": "Number of Flights (Reuse Count)",
            "simple": "How many times has this booster flown before?",
            "why": "The more times a booster has flown, the more wear and tear it has. "
                   "But SpaceX designs boosters to be reused many times, so experienced boosters "
                   "have also been through the refurbishment process. This is an interesting trade-off.",
            "range": "1 (brand new) to 7+ (battle-hardened veteran)",
            "type": "number",
        },
        {
            "emoji": "🪁", "name": "Grid Fins",
            "simple": "Does the rocket have those X-shaped fins near the top?",
            "why": "Grid fins act like steering wheels during descent. Without them, "
                   "the booster can't control its direction while falling back to Earth. "
                   "Almost all successful landings use grid fins.",
            "range": "Yes or No",
            "type": "bool",
        },
        {
            "emoji": "🦵", "name": "Landing Legs",
            "simple": "Does the booster have legs that pop out before landing?",
            "why": "Obviously — no legs, no landing! Legs were added starting in 2014. "
                   "Before that, there was no way to land upright.",
            "range": "Yes or No",
            "type": "bool",
        },
        {
            "emoji": "🌍", "name": "Orbit Type",
            "simple": "Where is the rocket delivering the cargo?",
            "why": "Different orbits require different amounts of fuel. "
                   "A GTO (satellite orbit far away) needs so much fuel that little is left for landing. "
                   "LEO (low orbit, like ISS) is closer, leaving more fuel for a safe return.",
            "range": "LEO, ISS, GTO, MEO, SSO, VLEO, etc.",
            "type": "category",
        },
        {
            "emoji": "🏗️", "name": "Block Version",
            "simple": "Which generation/version of the Falcon 9 is this?",
            "why": "SpaceX upgrades their rocket in 'Blocks' — think iPhone generations. "
                   "Block 5 (the latest) is specifically designed for easy reuse and has the "
                   "best landing success rate.",
            "range": "Block 1 (oldest) to Block 5 (current)",
            "type": "number",
        },
        {
            "emoji": "🚀", "name": "Launch Site",
            "simple": "Where is the rocket launching from?",
            "why": "The launch site affects the flight path and which landing zones are reachable. "
                   "KSC (Kennedy Space Center) and CCSFS in Florida are most common for drone ship landings.",
            "range": "CCSFS SLC 40 | KSC LC 39A | VAFB SLC 4E",
            "type": "category",
        },
    ]

    for f in features:
        color = "#1a3a4a" if f["type"] in ("number",) else "#1a4a2a" if f["type"] == "bool" else "#2a1a4a"
        border = "#58a6ff" if f["type"] == "number" else "#3fb950" if f["type"] == "bool" else "#bc8cff"
        st.markdown(
            f"""
            <div class="card" style="border-left: 4px solid {border}; background:{color}20;">
              <h3 style="margin:0 0 4px 0">{f['emoji']} {f['name']}</h3>
              <p style="color:#8b949e;margin:0 0 8px 0">
                <span class="chip">📐 Range: {f['range']}</span>
              </p>
              <p style="margin:0 0 6px 0"><b>In simple terms:</b> {f['simple']}</p>
              <p style="margin:0;color:#c9d1d9"><b>Why it matters:</b> {f['why']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("🔥 Feature Importance (What Does the Model Think?)")
    st.markdown("Which features does the Random Forest model consider most important?")

    feat_names = models["feature_cols"]
    importances = models["rf"].feature_importances_
    feat_df = (
        pd.DataFrame({"Feature": feat_names, "Importance": importances})
        .sort_values("Importance", ascending=True)
        .tail(12)
    )

    readable = {
        "PayloadMass": "🪨 Payload Mass", "Flights": "🔁 No. of Flights",
        "GridFins": "🪁 Grid Fins", "Legs": "🦵 Landing Legs",
        "Block": "🏗️ Block Version", "FlightNumber": "✈️ Flight Number",
        "ReusedCount": "♻️ Reuse Count", "Longitude": "📍 Longitude",
        "Latitude": "📍 Latitude", "Reused": "♻️ Reused?",
        "Orbit_LEO": "🌍 Orbit: LEO", "Orbit_GTO": "🌍 Orbit: GTO",
        "Orbit_ISS": "🌍 Orbit: ISS", "Orbit_SSO": "🌍 Orbit: SSO",
        "Orbit_VLEO": "🌍 Orbit: VLEO", "Orbit_PO": "🌍 Orbit: PO",
        "LaunchSite_KSC LC 39A": "🚀 Site: KSC",
        "LaunchSite_VAFB SLC 4E": "🚀 Site: VAFB",
    }
    feat_df["Label"] = feat_df["Feature"].map(lambda x: readable.get(x, x))

    fig3 = px.bar(
        feat_df, x="Importance", y="Label", orientation="h",
        color="Importance", color_continuous_scale=["#58a6ff", "#3fb950"],
    )
    fig3.update_layout(
        paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        font_color="#e6edf3", yaxis_title="",
        coloraxis_showscale=False, height=380,
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig3, width="stretch")


# ═══════════════════════════════════════════════════════════
#  PAGE 3 — SIMULATE A LAUNCH
# ═══════════════════════════════════════════════════════════
elif page == "Simulate a Launch":

    st.title("Simulate Your Own Launch")
    st.markdown(
        "Set the mission parameters below. The model will predict whether the booster "
        "will land safely — and how confident it is."
    )

    # ── Input form ──────────────────────────────────────────
    with st.form("launch_form"):

        st.subheader("📦 Mission Cargo")
        payload_mass = st.slider(
            "🪨 How heavy is the payload? (kg)",
            min_value=500, max_value=16000, value=4000, step=100,
            help="Heavier = less fuel left for landing",
        )

        st.subheader("🌍 Where is it going?")
        orbit_friendly = st.selectbox(
            "Select the orbit (destination)",
            options=[
                "LEO – Low Earth Orbit (like weather satellites)",
                "ISS – Space Station (~400 km up)",
                "GTO – Geostationary Transfer (TV satellites, far away)",
                "SSO – Sun-Synchronous (Earth observation)",
                "VLEO – Very Low Earth Orbit (very close)",
                "MEO – Medium Earth Orbit (GPS satellites)",
                "PO – Polar Orbit",
                "GEO – Geostationary",
                "HEO – Highly Elliptical Orbit",
                "SO – Suborbital",
                "ES-L1 – Lagrange Point (deep space)",
            ],
        )
        orbit_map = {
            "LEO – Low Earth Orbit (like weather satellites)": "LEO",
            "ISS – Space Station (~400 km up)": "ISS",
            "GTO – Geostationary Transfer (TV satellites, far away)": "GTO",
            "SSO – Sun-Synchronous (Earth observation)": "SSO",
            "VLEO – Very Low Earth Orbit (very close)": "VLEO",
            "MEO – Medium Earth Orbit (GPS satellites)": "MEO",
            "PO – Polar Orbit": "PO",
            "GEO – Geostationary": "GEO",
            "HEO – Highly Elliptical Orbit": "HEO",
            "SO – Suborbital": "SO",
            "ES-L1 – Lagrange Point (deep space)": "ES-L1",
        }
        orbit = orbit_map[orbit_friendly]

        st.subheader("🚀 Launch Site")
        site_friendly = st.selectbox(
            "Where is the rocket launching from?",
            options=[
                "Florida – CCSFS SLC 40 (most common)",
                "Florida – Kennedy Space Center LC 39A",
                "California – Vandenberg VAFB SLC 4E",
            ],
        )
        site_map = {
            "Florida – CCSFS SLC 40 (most common)": "CCSFS SLC 40",
            "Florida – Kennedy Space Center LC 39A": "KSC LC 39A",
            "California – Vandenberg VAFB SLC 4E": "VAFB SLC 4E",
        }
        site = site_map[site_friendly]

        st.subheader("🔧 Booster Hardware")
        c1, c2 = st.columns(2)
        with c1:
            has_grid_fins = st.radio("🪁 Grid Fins installed?", ["Yes", "No"], horizontal=True)
            has_legs = st.radio("🦵 Landing Legs installed?", ["Yes", "No"], horizontal=True)
        with c2:
            block_version = st.slider("🏗️ Block version of Falcon 9", 1, 5, 5)
            num_flights = st.slider("🔁 How many times has this booster flown?", 1, 10, 1)

        st.subheader("✈️ Mission Number")
        flight_number = st.slider("Flight number in SpaceX history", 1, 100, 50)
        reused_count = num_flights - 1

        submitted = st.form_submit_button("🚀 Launch & Predict!", use_container_width=True)

    # ── Prediction ──────────────────────────────────────────
    if submitted:

        # Build the feature vector matching training columns
        feature_cols = models["feature_cols"]
        input_dict = {col: 0 for col in feature_cols}

        # Set numeric features only if they are part of the selected feature columns
        if "FlightNumber" in input_dict:
            input_dict["FlightNumber"] = flight_number
        if "Block" in input_dict:
            input_dict["Block"] = float(block_version)
        if "Flights" in input_dict:
            input_dict["Flights"] = float(num_flights)
        if "ReusedCount" in input_dict:
            input_dict["ReusedCount"] = float(reused_count)
        if "GridFins" in input_dict:
            input_dict["GridFins"] = 1 if has_grid_fins == "Yes" else 0
        if "Reused" in input_dict:
            input_dict["Reused"] = 1 if num_flights > 1 else 0
        if "Legs" in input_dict:
            input_dict["Legs"] = 1 if has_legs == "Yes" else 0

        # Longitude/Latitude approximate defaults
        if site == "CCSFS SLC 40":
            if "Longitude" in input_dict:
                input_dict["Longitude"] = -80.577
            if "Latitude" in input_dict:
                input_dict["Latitude"] = 28.562
        elif site == "KSC LC 39A":
            if "Longitude" in input_dict:
                input_dict["Longitude"] = -80.604
            if "Latitude" in input_dict:
                input_dict["Latitude"] = 28.608
        else:
            if "Longitude" in input_dict:
                input_dict["Longitude"] = -120.611
            if "Latitude" in input_dict:
                input_dict["Latitude"] = 34.632

        # Orbit one-hot
        orbit_col = f"Orbit_{orbit}"
        if orbit_col in input_dict:
            input_dict[orbit_col] = 1

        # Launch site one-hot
        if site == "KSC LC 39A" and "LaunchSite_KSC LC 39A" in input_dict:
            input_dict["LaunchSite_KSC LC 39A"] = 1
        elif site == "VAFB SLC 4E" and "LaunchSite_VAFB SLC 4E" in input_dict:
            input_dict["LaunchSite_VAFB SLC 4E"] = 1

        # Scale PayloadMass and Flights using same scaler
        scaler = models["scaler"]
        scaled = scaler.transform(
            pd.DataFrame([[payload_mass, num_flights]], columns=["PayloadMass", "Flights"])
        )
        pm_scaled, fl_scaled = scaled[0]
        if "PayloadMass" in input_dict:
            input_dict["PayloadMass"] = pm_scaled
        if "Flights" in input_dict:
            input_dict["Flights"] = fl_scaled

        input_df = pd.DataFrame([input_dict])[feature_cols]

        model = models["final"]
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        confidence = proba[1] if prediction == 1 else proba[0]

        st.markdown("---")
        st.subheader("🎯 Prediction Result")

        if prediction == 1:
            st.markdown(
                f"""<div class="result-success">
                    ✅ SUCCESS — Booster will LAND safely!<br>
                    <span style="font-size:1rem;font-weight:normal">
                    Confidence: {confidence*100:.1f}%
                    </span>
                </div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""<div class="result-fail">
                    💥 FAILURE — Booster will NOT land safely.<br>
                    <span style="font-size:1rem;font-weight:normal">
                    Confidence: {confidence*100:.1f}%
                    </span>
                </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"color": "#e6edf3", "size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                "bar": {"color": "#3fb950" if prediction == 1 else "#f85149"},
                "bgcolor": "#21262d",
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [0, 50], "color": "#2d0d0d"},
                    {"range": [50, 75], "color": "#2d2200"},
                    {"range": [75, 100], "color": "#0d2818"},
                ],
                "threshold": {
                    "line": {"color": "#e3b341", "width": 3},
                    "thickness": 0.75,
                    "value": 75,
                },
            },
            title={"text": "Model Confidence", "font": {"color": "#8b949e"}},
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0d1117",
            font_color="#e6edf3",
            height=280,
            margin=dict(l=40, r=40, t=20, b=10),
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.plotly_chart(fig_gauge, width="stretch")
        with c2:
            st.markdown("#### 📋 Your Mission Summary")
            st.markdown(
                f"""
                | Parameter | Value |
                |-----------|-------|
                | 🪨 Payload Mass | {payload_mass:,} kg |
                | 🌍 Orbit | {orbit} |
                | 🚀 Launch Site | {site} |
                | 🪁 Grid Fins | {has_grid_fins} |
                | 🦵 Landing Legs | {has_legs} |
                | 🏗️ Block Version | {block_version} |
                | 🔁 Previous Flights | {num_flights - 1} |
                | 🤖 Model Used | {models['model_name']} |
                """
            )

        # Both model comparison
        lr_pred = models["lr"].predict(input_df)[0]
        rf_pred = models["rf"].predict(input_df)[0]
        lr_prob = models["lr"].predict_proba(input_df)[0]
        rf_prob = models["rf"].predict_proba(input_df)[0]

        st.markdown("---")
        st.subheader("🔬 What Do Both Models Say?")
        c1, c2 = st.columns(2)
        with c1:
            lr_result = "✅ Success" if lr_pred == 1 else "💥 Failure"
            lr_conf = lr_prob[1] if lr_pred == 1 else lr_prob[0]
            st.markdown(
                f"""<div class="card card-info">
                <b>🔵 Logistic Regression</b><br>
                Result: <b>{lr_result}</b><br>
                Confidence: <b>{lr_conf*100:.1f}%</b>
                </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            rf_result = "✅ Success" if rf_pred == 1 else "💥 Failure"
            rf_conf = rf_prob[1] if rf_pred == 1 else rf_prob[0]
            st.markdown(
                f"""<div class="card card-info">
                <b>🟢 Random Forest</b><br>
                Result: <b>{rf_result}</b><br>
                Confidence: <b>{rf_conf*100:.1f}%</b>
                </div>""",
                unsafe_allow_html=True,
            )

        if lr_pred != rf_pred:
            st.warning(
                "⚠️ The two models disagree! This usually happens when the mission "
                "parameters are in a 'grey zone' — borderline cases where landing is uncertain."
            )
        else:
            st.success("✅ Both models agree on the outcome — prediction is more reliable.")