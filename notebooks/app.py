import io
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from scipy import stats
except Exception:
    stats = None


# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Video Games Analytics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main {background-color: #0b1020;}
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #111827 0%, #0b1020 100%);}
    h1, h2, h3 {letter-spacing: -0.03em;}
    .hero {
        padding: 1.5rem 1.7rem;
        border-radius: 24px;
        background: radial-gradient(circle at top left, rgba(99,102,241,0.35), transparent 35%),
                    linear-gradient(135deg, #111827 0%, #172554 55%, #312e81 100%);
        border: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 1rem;
    }
    .hero h1 {color: #f8fafc; font-size: 2.3rem; margin-bottom: 0.25rem;}
    .hero p {color: #cbd5e1; font-size: 1.02rem; margin-bottom: 0;}
    .metric-card {
        padding: 1rem;
        border-radius: 20px;
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(148,163,184,0.22);
        box-shadow: 0 16px 40px rgba(0,0,0,0.24);
    }
    .metric-label {color: #94a3b8; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em;}
    .metric-value {color: #f8fafc; font-size: 1.65rem; font-weight: 800; margin-top: .15rem;}
    .metric-note {color: #cbd5e1; font-size: .82rem; margin-top: .25rem;}
    .insight-box {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(30,41,59,0.74);
        border-left: 5px solid #818cf8;
        color: #e2e8f0;
        margin: 0.4rem 0 1rem 0;
    }
    .small-muted {color: #94a3b8; font-size: .9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FUNCIONES DE DATOS
# ============================================================
@st.cache_data(show_spinner=False)
def clean_games(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    required = {
        "name", "platform", "year_of_release", "genre", "na_sales", "eu_sales",
        "jp_sales", "other_sales", "critic_score", "user_score", "rating"
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {', '.join(missing)}")

    df["name"] = df["name"].fillna("unknown")
    df["genre"] = df["genre"].fillna("unknown")
    df["rating"] = df["rating"].fillna("unknown")

    df = df.replace("tbd", np.nan)
    numeric_cols = [
        "year_of_release", "na_sales", "eu_sales", "jp_sales", "other_sales",
        "critic_score", "user_score"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["total_sales"] = df[["na_sales", "eu_sales", "jp_sales", "other_sales"]].sum(axis=1)
    df["user_score_100"] = df["user_score"] * 10
    df["platform"] = df["platform"].astype(str)
    df["genre"] = df["genre"].astype(str)
    df["rating"] = df["rating"].astype(str)

    return df


@st.cache_data(show_spinner=False)
def load_default_data() -> pd.DataFrame | None:
    possible_paths = ["games.csv", "../games.csv", "/mnt/data/games.csv"]
    for path in possible_paths:
        try:
            return pd.read_csv(path)
        except Exception:
            continue
    return None


def fmt_millions(value: float) -> str:
    if pd.isna(value):
        return "—"
    return f"{value:,.1f}M"


def metric_card(label: str, value: str, note: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top_5_region(df: pd.DataFrame, regional: str, search: str) -> pd.DataFrame:
    out = df.groupby(search, observed=False)[[regional]].sum().sort_values(regional, ascending=False).head(5)
    total = out[regional].sum()
    out["market_share"] = np.where(total > 0, out[regional] / total, 0)
    return out.reset_index()


def platform_lifecycle(df: pd.DataFrame, selected_platforms: List[str]) -> pd.DataFrame:
    sales_by_year = df.groupby(["platform", "year_of_release"], observed=False)["total_sales"].sum()
    rows = []
    for platform in selected_platforms:
        if platform not in sales_by_year.index.get_level_values(0):
            continue
        s = sales_by_year.loc[platform]
        s = s[s > 0]
        if platform == "DS" and len(s) > 1:
            # El dataset clásico trae una venta aislada de DS en 1985; se elimina para no distorsionar.
            s = s[s.index >= 2004]
        if s.empty:
            continue
        first = int(s.index.min())
        last = int(s.index.max())
        peak_year = int(s.idxmax())
        rows.append(
            {
                "platform": platform,
                "first_year": first,
                "peak_year": peak_year,
                "last_year": last,
                "years_to_peak": peak_year - first,
                "commercial_life_years": last - first + 1,
                "peak_sales": s.max(),
                "total_sales": s.sum(),
            }
        )
    return pd.DataFrame(rows)


def hypothesis_test(df: pd.DataFrame, field: str, a: str, b: str, alpha: float) -> Dict[str, object]:
    x = df.loc[df[field] == a, "user_score"].dropna()
    y = df.loc[df[field] == b, "user_score"].dropna()

    if len(x) < 2 or len(y) < 2:
        return {
            "ok": False,
            "message": "No hay suficientes datos de user_score para hacer la prueba.",
        }

    if stats is None:
        return {
            "ok": False,
            "message": "SciPy no está instalado. Instala scipy para ejecutar la prueba estadística.",
        }

    levene = stats.levene(x, y)
    equal_var = bool(levene.pvalue >= 0.05)
    ttest = stats.ttest_ind(x, y, equal_var=equal_var)
    decision = "Rechazar H0" if ttest.pvalue < alpha else "No rechazar H0"
    interpretation = (
        "hay evidencia estadística de que las calificaciones promedio son diferentes"
        if ttest.pvalue < alpha
        else "no hay evidencia suficiente para afirmar que las calificaciones promedio son diferentes"
    )
    return {
        "ok": True,
        "n_a": len(x),
        "n_b": len(y),
        "mean_a": x.mean(),
        "mean_b": y.mean(),
        "levene_p": levene.pvalue,
        "equal_var": equal_var,
        "ttest_p": ttest.pvalue,
        "decision": decision,
        "interpretation": interpretation,
    }


# ============================================================
# SIDEBAR / CARGA DE DATOS
# ============================================================
st.sidebar.title("🎮 Video Games Dashboard")
st.sidebar.caption("Carga el archivo `games.csv` o usa uno disponible en la carpeta del proyecto.")

uploaded = st.sidebar.file_uploader("Subir games.csv", type=["csv"])

try:
    if uploaded is not None:
        raw_games = pd.read_csv(uploaded)
    else:
        raw_games = load_default_data()

    if raw_games is None:
        st.warning("Sube tu archivo `games.csv` para activar el dashboard.")
        st.stop()

    games = clean_games(raw_games)
except Exception as exc:
    st.error(f"No se pudo cargar o limpiar el dataset: {exc}")
    st.stop()

min_year = int(games["year_of_release"].dropna().min())
max_year = int(games["year_of_release"].dropna().max())

year_range = st.sidebar.slider(
    "Periodo de análisis",
    min_value=min_year,
    max_value=max_year,
    value=(max(2002, min_year), max_year),
)

platform_options = sorted(games["platform"].dropna().unique().tolist())
genre_options = sorted(games["genre"].dropna().unique().tolist())
rating_options = sorted(games["rating"].dropna().unique().tolist())

platform_filter = st.sidebar.multiselect("Plataformas", platform_options, default=[])
genre_filter = st.sidebar.multiselect("Géneros", genre_options, default=[])
rating_filter = st.sidebar.multiselect("Clasificación ESRB", rating_options, default=[])

filtered = games[
    games["year_of_release"].between(year_range[0], year_range[1], inclusive="both")
].copy()
if platform_filter:
    filtered = filtered[filtered["platform"].isin(platform_filter)]
if genre_filter:
    filtered = filtered[filtered["genre"].isin(genre_filter)]
if rating_filter:
    filtered = filtered[filtered["rating"].isin(rating_filter)]

st.sidebar.divider()
st.sidebar.caption("Tip: para replicar el enfoque del notebook, usa 2002–2016.")


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
        <h1>Video Games Market Analytics</h1>
        <p>Dashboard interactivo para analizar ventas, plataformas, géneros, regiones, reseñas y pruebas de hipótesis del mercado de videojuegos.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# KPIS
c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Juegos analizados", f"{filtered['name'].nunique():,}", "Títulos únicos")
with c2:
    metric_card("Ventas globales", fmt_millions(filtered["total_sales"].sum()), "NA + EU + JP + Other")
with c3:
    metric_card("Plataformas", f"{filtered['platform'].nunique():,}", "Con ventas en el periodo")
with c4:
    top_platform = filtered.groupby("platform")["total_sales"].sum().sort_values(ascending=False)
    metric_card("Plataforma líder", top_platform.index[0] if len(top_platform) else "—", fmt_millions(top_platform.iloc[0]) if len(top_platform) else "")

if filtered.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()


# ============================================================
# TABS PRINCIPALES
# ============================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "📈 Panorama",
        "🕹️ Plataformas",
        "🌎 Regiones",
        "⭐ Reseñas",
        "🧪 Hipótesis",
        "📋 Datos",
    ]
)

with tab1:
    st.subheader("Panorama general del mercado")
    yearly = filtered.groupby("year_of_release", as_index=False).agg(
        games_released=("name", "count"),
        total_sales=("total_sales", "sum"),
        avg_sales=("total_sales", "mean"),
    )

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        fig = px.bar(
            yearly,
            x="year_of_release",
            y="games_released",
            title="Juegos lanzados por año",
            labels={"year_of_release": "Año", "games_released": "Juegos lanzados"},
            text_auto=False,
        )
        fig.update_layout(height=420, template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig = px.line(
            yearly,
            x="year_of_release",
            y="total_sales",
            markers=True,
            title="Ventas globales por año",
            labels={"year_of_release": "Año", "total_sales": "Ventas globales (millones)"},
        )
        fig.update_layout(height=420, template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    genre_sales = filtered.groupby("genre", as_index=False)["total_sales"].mean().sort_values("total_sales", ascending=False)
    fig = px.bar(
        genre_sales,
        x="genre",
        y="total_sales",
        title="Rentabilidad promedio por género",
        labels={"genre": "Género", "total_sales": "Venta promedio por juego (millones)"},
    )
    fig.update_layout(height=430, template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

    best_genre = genre_sales.iloc[0]
    worst_genre = genre_sales.iloc[-1]
    st.markdown(
        f"""
        <div class="insight-box">
        <b>Insight:</b> En el periodo filtrado, el género con mayor venta promedio es <b>{best_genre['genre']}</b> 
        con {fmt_millions(best_genre['total_sales'])} por juego. El menor promedio aparece en <b>{worst_genre['genre']}</b>, 
        con {fmt_millions(worst_genre['total_sales'])} por juego.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab2:
    st.subheader("Análisis de plataformas")

    sales_by_platform = filtered.groupby("platform", as_index=False)["total_sales"].sum().sort_values("total_sales", ascending=False)
    top_n = st.slider("Número de plataformas a mostrar", 5, min(20, len(sales_by_platform)), min(10, len(sales_by_platform)))
    top_platforms = sales_by_platform.head(top_n)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        fig = px.bar(
            top_platforms,
            x="platform",
            y="total_sales",
            title=f"Top {top_n} plataformas por ventas totales",
            labels={"platform": "Plataforma", "total_sales": "Ventas (millones)"},
        )
        fig.update_layout(height=430, template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        annual_platform = filtered[filtered["platform"].isin(top_platforms["platform"])].groupby(
            ["year_of_release", "platform"], as_index=False
        )["total_sales"].sum()
        fig = px.line(
            annual_platform,
            x="year_of_release",
            y="total_sales",
            color="platform",
            markers=True,
            title="Evolución anual de plataformas líderes",
            labels={"year_of_release": "Año", "total_sales": "Ventas (millones)", "platform": "Plataforma"},
        )
        fig.update_layout(height=430, template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    box_data = filtered[filtered["platform"].isin(top_platforms["platform"])]
    fig = px.box(
        box_data,
        x="platform",
        y="total_sales",
        points="outliers",
        title="Distribución de ventas por juego en plataformas líderes",
        labels={"platform": "Plataforma", "total_sales": "Ventas por juego (millones)"},
    )
    fig.update_layout(height=460, template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

    lifecycle = platform_lifecycle(games, top_platforms["platform"].tolist())
    if not lifecycle.empty:
        st.markdown("### Ciclo de vida de plataformas")
        col_l1, col_l2 = st.columns([1, 1])
        with col_l1:
            fig = px.bar(
                lifecycle.sort_values("commercial_life_years", ascending=False),
                x="platform",
                y=["years_to_peak", "commercial_life_years"],
                barmode="group",
                title="Años hasta pico y vida comercial",
                labels={"value": "Años", "platform": "Plataforma", "variable": "Métrica"},
            )
            fig.update_layout(height=420, template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, use_container_width=True)
        with col_l2:
            st.dataframe(
                lifecycle.sort_values("total_sales", ascending=False),
                use_container_width=True,
                hide_index=True,
            )

with tab3:
    st.subheader("Perfil de usuario por región")
    region_labels = {
        "na_sales": "Norteamérica",
        "eu_sales": "Europa",
        "jp_sales": "Japón",
    }
    region = st.radio("Región", list(region_labels.keys()), format_func=lambda x: region_labels[x], horizontal=True)

    col_a, col_b, col_c = st.columns(3)
    for col, field, title in [
        (col_a, "platform", "Top plataformas"),
        (col_b, "genre", "Top géneros"),
        (col_c, "rating", "Top ESRB"),
    ]:
        with col:
            d = top_5_region(filtered, region, field)
            fig = px.pie(
                d,
                values=region,
                names=field,
                hole=0.55,
                title=title,
            )
            fig.update_layout(height=390, template="plotly_dark", margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)

    regional_sales = filtered[["na_sales", "eu_sales", "jp_sales", "other_sales"]].sum().reset_index()
    regional_sales.columns = ["region", "sales"]
    regional_sales["region"] = regional_sales["region"].replace(
        {
            "na_sales": "Norteamérica",
            "eu_sales": "Europa",
            "jp_sales": "Japón",
            "other_sales": "Otros",
        }
    )
    fig = px.bar(
        regional_sales,
        x="region",
        y="sales",
        title="Comparación de ventas por región",
        labels={"region": "Región", "sales": "Ventas (millones)"},
    )
    fig.update_layout(height=420, template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("Impacto de reseñas profesionales y de usuarios")

    default_platform = "PS4" if "PS4" in platform_options else platform_options[0]
    score_platform = st.selectbox("Selecciona plataforma", platform_options, index=platform_options.index(default_platform))
    score_df = filtered[filtered["platform"] == score_platform].copy()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        fig = px.scatter(
            score_df,
            x="critic_score",
            y="total_sales",
            hover_data=["name", "genre", "year_of_release"],
            title=f"Críticos vs ventas — {score_platform}",
            labels={"critic_score": "Calificación de críticos", "total_sales": "Ventas por juego (millones)"},
            trendline="ols" if len(score_df.dropna(subset=["critic_score", "total_sales"])) >= 3 else None,
        )
        fig.update_layout(height=430, template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig = px.scatter(
            score_df,
            x="user_score_100",
            y="total_sales",
            hover_data=["name", "genre", "year_of_release"],
            title=f"Usuarios vs ventas — {score_platform}",
            labels={"user_score_100": "Calificación de usuarios (0-100)", "total_sales": "Ventas por juego (millones)"},
            trendline="ols" if len(score_df.dropna(subset=["user_score_100", "total_sales"])) >= 3 else None,
        )
        fig.update_layout(height=430, template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

    corr_critic = score_df["critic_score"].corr(score_df["total_sales"])
    corr_user = score_df["user_score_100"].corr(score_df["total_sales"])
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        metric_card("Correlación críticos vs ventas", f"{corr_critic:.2f}" if pd.notna(corr_critic) else "—", "Pearson")
    with col_m2:
        metric_card("Correlación usuarios vs ventas", f"{corr_user:.2f}" if pd.notna(corr_user) else "—", "Pearson")

    st.markdown(
        """
        <div class="insight-box">
        <b>Lectura rápida:</b> una correlación cercana a 1 indica relación positiva fuerte; cercana a 0 indica relación débil. 
        En este dataset, normalmente las reseñas ayudan a explicar parte del desempeño, pero no sustituyen factores como franquicia, distribución, marketing, exclusividad o momento del ciclo de vida de la consola.
        </div>
        """,
        unsafe_allow_html=True,
    )

with tab5:
    st.subheader("Pruebas de hipótesis sobre calificaciones de usuarios")
    st.caption("Se comparan medias de `user_score` usando Levene para varianzas y t-test independiente.")

    col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 0.8])
    with col_a:
        field = st.selectbox("Comparar por", ["platform", "genre"], format_func=lambda x: "Plataforma" if x == "platform" else "Género")
    options = sorted(filtered[field].dropna().unique().tolist())
    default_a = "XOne" if field == "platform" and "XOne" in options else options[0]
    default_b = "PC" if field == "platform" and "PC" in options else options[min(1, len(options)-1)]
    if field == "genre":
        default_a = "Action" if "Action" in options else options[0]
        default_b = "Sports" if "Sports" in options else options[min(1, len(options)-1)]
    with col_b:
        a = st.selectbox("Grupo A", options, index=options.index(default_a))
    with col_c:
        b = st.selectbox("Grupo B", options, index=options.index(default_b))
    with col_d:
        alpha = st.selectbox("Alpha", [0.01, 0.05, 0.10], index=1)

    res = hypothesis_test(filtered, field, a, b, alpha)
    if not res["ok"]:
        st.warning(res["message"])
    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card(f"Media {a}", f"{res['mean_a']:.2f}", f"n = {res['n_a']}")
        with m2:
            metric_card(f"Media {b}", f"{res['mean_b']:.2f}", f"n = {res['n_b']}")
        with m3:
            metric_card("p-value Levene", f"{res['levene_p']:.4f}", "Varianzas iguales" if res["equal_var"] else "Varianzas diferentes")
        with m4:
            metric_card("p-value t-test", f"{res['ttest_p']:.4f}", res["decision"])

        st.markdown(
            f"""
            <div class="insight-box">
            <b>Resultado:</b> con α = {alpha}, se decide <b>{res['decision']}</b>. 
            Interpretación: {res['interpretation']} entre <b>{a}</b> y <b>{b}</b>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        dist = filtered[filtered[field].isin([a, b])].dropna(subset=["user_score"])
        fig = px.histogram(
            dist,
            x="user_score",
            color=field,
            barmode="overlay",
            marginal="box",
            opacity=0.65,
            title="Distribución de calificaciones de usuario",
            labels={"user_score": "Calificación usuario", field: "Grupo"},
        )
        fig.update_layout(height=460, template="plotly_dark", margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("Datos limpios y resumen")
    st.caption("El dataset se limpió: columnas en snake_case, `tbd` convertido a nulo, `user_score` numérico y ventas totales calculadas.")

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### Valores nulos por columna")
        nulls = filtered.isna().sum().reset_index()
        nulls.columns = ["column", "missing_values"]
        st.dataframe(nulls, use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("#### Estadística descriptiva")
        st.dataframe(filtered.describe(include="all").T, use_container_width=True)

    st.markdown("#### Vista del dataset filtrado")
    st.dataframe(filtered, use_container_width=True, hide_index=True)

    csv_buffer = io.StringIO()
    filtered.to_csv(csv_buffer, index=False)
    st.download_button(
        "⬇️ Descargar datos filtrados",
        data=csv_buffer.getvalue(),
        file_name="games_clean_filtered.csv",
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    """
    <p class="small-muted">
    Dashboard creado en un solo archivo Streamlit. Para ejecutarlo: <code>streamlit run app.py</code>. 
    El archivo espera un CSV con la estructura del dataset de videojuegos usado en el proyecto.
    </p>
    """,
    unsafe_allow_html=True,
)
