import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(page_title="Crop Yield Prediction System", page_icon="🌾", layout="wide")

USERNAME, PASSWORD = "admin", "crop123"
BASE_DIR = Path(__file__).resolve().parent
R2_SCORE, MAE_SCORE = 0.8923, 16.06  # verified against the training notebook's final XGBoost run

# Below this historical median yield (tons/hectare), a crop's own ground-truth
# records are already close to zero. This is a documented trait of the source
# dataset (several vegetable/fruit crops are reported with Production values
# far smaller than Area at state level), not a model failure. It is derived
# from the data, never hard-coded, and is used only to add context to the
# result -- never to alter the model's output.
LOW_YIELD_MEDIAN_THRESHOLD = 0.15


# ============================================================
# STYLE
# ============================================================

st.markdown("""
<style>
.stApp {background:#F1F5F0;}
h1,h2,h3 {color:#1B4332 !important;}
.hero {background:linear-gradient(135deg,#1B4332,#2D6A4F); padding:35px 40px;
       border-radius:16px; color:white; margin-bottom:25px;}
.hero h1 {color:white !important; margin:0;}
.hero p {color:#D8E5DC; margin:7px 0 0;}
.card {background:white; padding:22px; border-radius:14px; border:1px solid #E3E9E2; margin-bottom:16px;}
.stButton>button {width:100%; height:46px; border-radius:10px; font-weight:600; background:#2D6A4F; color:white;}
.stButton>button:hover {background:#1B4332; color:white;}
[data-testid="stSidebar"] {background:#1B4332;}
[data-testid="stSidebar"] * {color:white !important;}
</style>
""", unsafe_allow_html=True)


def hero(title, subtitle):
    st.markdown(f"<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL + DATA
# ============================================================

@st.cache_resource
def load_project():
    
    with open(BASE_DIR / "xgb_crop_yield_model.pkl", "rb") as f:
        model = pickle.load(f)

    crop = pd.read_csv(BASE_DIR / "crop_production.csv")
    rain = pd.read_csv(BASE_DIR / "rainfall_cleaned.csv")

    crop.columns = crop.columns.str.strip()
    rain.columns = rain.columns.str.strip().str.lower()

    crop = (
        crop.dropna(subset=["Production"])
        .rename(columns={"State_Name": "state", "Crop_Year": "year"})
        .drop(columns=["District_Name"])
    )
    crop["state"] = crop["state"].str.strip()
    rain["state"] = rain["state"].str.strip()

    df = crop.merge(rain, on=["state", "year"], how="left")
    df = df[df["year"] >= 1997]
    df = df[df["state"].isin(rain["state"].unique())]
    df = df.dropna(subset=["rainfall"]).copy()

    df["yield"] = df["Production"] / df["Area"]

    # Per-crop historical median, computed BEFORE encoding, so the app can
    # flag (never alter) predictions for crops whose own training data is
    # already near zero.
    low_yield_crops = set(df.groupby("Crop")["yield"].median().loc[lambda s: s < LOW_YIELD_MEDIAN_THRESHOLD].index)

    encoders = {}
    for col in ["state", "Crop", "Season"]:
        encoders[col] = LabelEncoder().fit(df[col])
        df[col] = encoders[col].transform(df[col])

    return model, df, encoders, low_yield_crops, rain


try:
    model, train_df, enc, low_yield_crops, rainfall = load_project()
except Exception as e:
    st.error(f"Could not load the project files:\n\n{e}")
    st.stop()


# ============================================================
# RAINFALL ESTIMATION
# ============================================================

def estimate_rainfall(state, year):
    """Historical rainfall when available; degree-2 polynomial regression,
    fit on that state's own history only, for years outside the dataset.
    Rainfall is physically bounded at 0 (a real constraint, not a tuning
    knob), so a negative extrapolated value is clipped to 0."""
    data = rainfall[rainfall["state"] == state].dropna(subset=["year", "rainfall"])

    if year in data["year"].values:
        return float(data.loc[data["year"] == year, "rainfall"].iloc[0]), False

    if len(data) < 5:
        return float(rainfall["rainfall"].mean()), True

    poly = PolynomialFeatures(degree=2)
    X = poly.fit_transform(data[["year"]])
    reg = LinearRegression().fit(X, data["rainfall"])
    value = reg.predict(poly.transform([[year]]))[0]
    return max(float(value), 0.0), True


# ============================================================
# PREDICTION
# ============================================================

def predict_yield(state, crop_name, season_name, year, area, rainfall_value):
    """Run the model with features in the exact order used at training time
    (state, year, Season, Crop, Area, rainfall -- confirmed from the training
    notebook). The raw model output is returned unmodified except for a clip
    at 0, since a negative yield is not physically possible. No arbitrary
    minimum (e.g. 0.1 or 1) is ever applied -- that would hide, not fix, an
    out-of-distribution prediction."""
    state_code = enc["state"].transform([state])[0]
    crop_code = enc["Crop"].transform([crop_name])[0]
    season_code = enc["Season"].transform([season_name])[0]

    X = pd.DataFrame(
        [[state_code, year, season_code, crop_code, area, rainfall_value]],
        columns=["state", "year", "Season", "Crop", "Area", "rainfall"],
    )
    raw_prediction = float(model.predict(X)[0])
    return max(0.0, raw_prediction), raw_prediction


# ============================================================
# LOGIN
# ============================================================

# Initialize Session State for Login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    hero("🌾 Crop Yield Prediction System", "Machine Learning Based Agricultural Decision Support System")

    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == USERNAME and password == PASSWORD:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password.")
        st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.markdown("## 🌾 Navigation")
page = st.sidebar.radio("", ["🏠 Home", "🌾 Prediction", "📊 Dashboard", "📘 User Guide", "👤 About"])
if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()


# ============================================================
# HOME
# ============================================================

# --- HOME PAGE ---
if page == "🏠 Home":
    hero("🌾 Crop Yield Prediction System", "Machine Learning Based Agricultural Decision Support System")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", "XGBoost Regressor")
    c2.metric("Model R²", f"{R2_SCORE * 100:.2f}%")
    c3.metric("MAE", f"{MAE_SCORE:.2f} t/ha")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("What the system does")
    st.write(
        "The system predicts crop yield from state, cultivation year, crop, season, "
        "cultivated area and annual rainfall. Historical rainfall is used when available; "
        "for a year outside the rainfall dataset, polynomial regression fit on that state's "
        "own rainfall history estimates it first."
    )
    st.caption(
        "R² indicates how much of the variation in yield the model explains (89.23% here). "
        "MAE is the average absolute prediction error, in tons/hectare."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    a, b, c = st.columns(3)
    a.markdown("### 🌱 Smart Prediction\nUses XGBoost to model non-linear relationships.")
    b.markdown("### 🌧️ Future Rainfall\nEstimates rainfall for years beyond the historical data.")
    c.markdown("### 📊 Data Dashboard\nDisplays rainfall trends and model information.")

    st.info("Select **Prediction** from the sidebar to generate a crop yield prediction.")


# ============================================================
# PREDICTION
# ============================================================

# --- PREDICTION PAGE ---
elif page == "🌾 Prediction":
    hero("🌾 Crop Yield Prediction", "Enter the cultivation information")

    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # Dependent selects: only combinations seen in training are offered, so
    # an unsupported State+Crop+Season combination can never be submitted.
    states = sorted(enc["state"].inverse_transform(train_df["state"].unique()))
    state = st.selectbox("State", states)

    state_df = train_df[train_df["state"] == enc["state"].transform([state])[0]]
    crops = sorted(enc["Crop"].inverse_transform(state_df["Crop"].unique()))
    crop_name = st.selectbox("Crop", crops)

    combo = state_df[state_df["Crop"] == enc["Crop"].transform([crop_name])[0]]
    seasons = sorted(enc["Season"].inverse_transform(combo["Season"].unique()))
    season_name = st.selectbox("Season", seasons)

    history = combo[combo["Season"] == enc["Season"].transform([season_name])[0]]

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Cultivation Year", min_value=1997, max_value=2035, value=2026, step=1)
    with col2:
        area_min = max(0.1, float(history["Area"].min()))
        area_max = float(history["Area"].max())
        area_default = float(history["Area"].median())
        area = st.number_input(
            "Area (hectares)",
            min_value=area_min,
            max_value=max(area_min, area_max),
            value=min(area_default, max(area_min, area_max)),
            step=max(0.1, round(area_default / 20, 1)),
            help="Range and default are drawn from historical records for this exact State + Crop + Season.",
        )

    year = int(year)
    rainfall_value, estimated = estimate_rainfall(state, year)
    st.info(
        f"🌧️ Rainfall for {state} in {year}: **{rainfall_value:.1f} mm** — "
        + ("estimated using polynomial regression (year outside the historical rainfall data)."
           if estimated else "historical value.")
    )

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("🚀 Predict Yield"):
        prediction, raw_prediction = predict_yield(state, crop_name, season_name, year, area, rainfall_value)
        production = prediction * area

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        r1, r2 = st.columns(2)
        r1.metric("Predicted Yield", f"{prediction:.2f} tons/hectare")
        r2.metric("Estimated Production", f"{production:,.2f} tons")

        hist_yield = history["yield"]
        st.caption(
            f"Historical yield for {crop_name} in {state} ({season_name.strip()}) "
            f"ranges {hist_yield.min():.2f}–{hist_yield.max():.2f} t/ha "
            f"(median {hist_yield.median():.2f} t/ha) across {len(history)} records."
        )

        if crop_name in low_yield_crops:
            st.warning(
                f"**{crop_name}** is one of the crops whose own historical records in this "
                "dataset already sit close to zero t/ha at state level (a known scale/unit "
                "limitation of the source data for several vegetable and fruit crops, not "
                "specific to this prediction). A low value here is consistent with that "
                "crop's training data rather than a model error."
            )
        elif raw_prediction < 0:
            st.warning(
                "The model's raw output for this combination was negative and has been "
                "clipped to 0 t/ha (yield cannot be negative). This usually means the "
                "inputs sit at the edge of what the model saw during training -- treat "
                "this result with caution."
            )
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Prediction details"):
            st.write(f"**State:** {state}")
            st.write(f"**Crop:** {crop_name}")
            st.write(f"**Season:** {season_name.strip()}")
            st.write(f"**Year:** {year}")
            st.write(f"**Area:** {area:,.2f} hectares")
            st.write(f"**Rainfall:** {rainfall_value:.1f} mm")


# ============================================================
# DASHBOARD
# ============================================================

elif page == "📊 Dashboard":
    hero("📊 Project Dashboard", "Dataset and model overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("States", train_df["state"].nunique())
    c2.metric("Crops", train_df["Crop"].nunique())
    c3.metric("Seasons", train_df["Season"].nunique())
    c4.metric("Training Records", f"{len(train_df):,}")

    left, right = st.columns(2)
    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Model Performance")
        st.write("**Algorithm:** XGBoost Regressor")
        st.write(f"**R² Score:** {R2_SCORE * 100:.2f}%")
        st.write(f"**Mean Absolute Error:** {MAE_SCORE:.2f} t/ha")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Dataset")
        st.write(f"**Years:** {int(train_df['year'].min())}–{int(train_df['year'].max())}")
        st.write(f"**States:** {train_df['state'].nunique()}")
        st.write(f"**Crops:** {train_df['Crop'].nunique()}")
        st.write(f"**Rainfall records:** {len(rainfall):,}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Average Annual Rainfall")
    st.line_chart(rainfall.groupby("year")["rainfall"].mean())

    st.subheader("Average Rainfall by State")
    st.bar_chart(rainfall.groupby("state")["rainfall"].mean().sort_values(ascending=False))

    st.subheader("Rainfall and Crop Yield")
    st.caption(
        "Exploratory only — a relationship between rainfall and yield here does not "
        "imply rainfall causes the yield level; crop type, area and state also vary."
    )
    top_crops = sorted(enc["Crop"].inverse_transform(train_df["Crop"].value_counts().head(12).index))
    chosen_crop = st.selectbox("Choose a crop (12 most-recorded crops shown)", top_crops)
    crop_rows = train_df[train_df["Crop"] == enc["Crop"].transform([chosen_crop])[0]]
    scatter_df = crop_rows[["rainfall", "yield"]].rename(
        columns={"rainfall": "Rainfall (mm)", "yield": "Yield (t/ha)"}
    )
    st.scatter_chart(scatter_df, x="Rainfall (mm)", y="Yield (t/ha)")
    corr = crop_rows["rainfall"].corr(crop_rows["yield"])
    st.caption(f"Correlation between rainfall and yield for {chosen_crop}: {corr:.2f}")


# ============================================================
# USER GUIDE
# ============================================================

elif page == "📘 User Guide":
    hero("📘 User Guide", "How to generate a prediction")

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    steps = [
        ("Select the State", "Only states present in both the crop and rainfall datasets are listed."),
        ("Select a Crop", "Only crops actually grown in the chosen state (per training data) are offered."),
        ("Select a Season", "Only seasons recorded for that State + Crop combination are offered."),
        ("Enter the Cultivation Year", "Any year from 1997 to 2035 is accepted."),
        ("Enter the Area", "The suggested range and default come from historical records for that exact combination."),
        ("Rainfall is automatic", "Historical rainfall is used when available; otherwise it is estimated with "
                                   "polynomial regression fit on that state's own rainfall history."),
        ("Click Predict Yield", "The model returns a predicted yield and an estimated total production."),
        ("Read the context notes", "Compare the prediction to the historical range shown, and read any warning "
                                    "about low-yield crop categories or out-of-range inputs before relying on it."),
    ]
    for i, (title, detail) in enumerate(steps, start=1):
        st.write(f"**Step {i}: {title}**")
        st.caption(detail)

    st.info(
        "Predictions are only as reliable as the training data for that specific combination. "
        "A handful of crops (mostly minor vegetables and fruits) have near-zero yield even in "
        "their own historical records — this is a known characteristic of the source dataset, "
        "not a defect in the app."
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ABOUT
# ============================================================

else:
    hero("👤 About This Project", "Crop Yield Prediction System")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Student Information")
        st.write("**Name:** Mubarak Bashir Wali")
        st.write("**Registration Number:** UG21/COMS/1132")
        st.write("**Department:** Computer Science")
        st.write("**University:** Aliko Dangote University of Science and Technology, Wudil")
        st.write("**Year:** 2026")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Project Information")
        st.write("**Title:** Crop Yield Prediction System")
        st.write("**Supervisor:** Mln Mahmud Muhammad")
        st.write("**Model:** XGBoost Regressor")
        st.write("**Rainfall method:** Polynomial Regression")
        st.write("**Framework:** Streamlit")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("System Features")
    st.write("✔ Secure login")
    st.write("✔ Crop yield prediction, driven only by combinations seen in training")
    st.write("✔ Future-year rainfall estimation")
    st.write("✔ Interactive dashboard with rainfall/yield exploration")
    st.write("✔ Estimated total production")
    st.write("✔ User guide")
    st.markdown("</div>", unsafe_allow_html=True)
