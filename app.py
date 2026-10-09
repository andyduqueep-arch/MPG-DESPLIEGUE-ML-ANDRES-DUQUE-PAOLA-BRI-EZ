"""
App Streamlit: predicción del consumo de combustible (mpg) con el modelo de bagging optimizado.
Archivo único: no depende de otros .py. Debe estar junto a:
  one_hot_columns.joblib, min_max_scaler.joblib, bagging_optimizado.joblib y requirements.txt
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================== preprocesamiento y predicción
TARGET = 'mpg'
CATEGORICAS = {'origin': ['USA', 'Europe', 'Japan']}
ARCHIVOS = ('one_hot_columns.joblib', 'min_max_scaler.joblib', 'bagging_optimizado.joblib')

# Rango y valor por defecto de cada variable (Dataset_limpio.xlsx y reglas de calidad del punto 2.3)
RANGOS = {
    'cylinders':    dict(min=3,      max=8,      default=4,      step=1,    label='Cilindros'),
    'displacement': dict(min=50.0,   max=500.0,  default=148.8,  step=1.0,  label='Cilindrada (pulg³)'),
    'horsepower':   dict(min=40.0,   max=500.0,  default=102.8,  step=1.0,  label='Potencia (HP)'),
    'weight':       dict(min=1500.0, max=6000.0, default=2680.0, step=10.0, label='Peso (lb)'),
    'acceleration': dict(min=5.0,    max=25.0,   default=15.6,   step=0.1,  label='Aceleración 0-60 mph (s)'),
    'model_year':   dict(min=1970,   max=2026,   default=2010,   step=1,    label='Año del modelo'),
    'vehicle_age':  dict(min=0,      max=56,     default=16,     step=1,    label='Antigüedad (años)'),
}


def buscar_archivo(carpeta, nombre):
    """Encuentra el archivo aunque el navegador lo haya renombrado (p. ej. 'min_max_scaler (1).joblib')."""
    exacto = carpeta / nombre
    if exacto.exists():
        return exacto
    base = nombre.rsplit('.', 1)[0]
    candidatos = sorted(carpeta.glob(base + '*.joblib'))
    if candidatos:
        return candidatos[0]
    raise FileNotFoundError(2, 'No encontrado', nombre)


def cargar_artefactos(carpeta='.'):
    """Carga los 3 archivos .joblib: one-hot, scaler y modelo."""
    c = Path(carpeta)
    return tuple(joblib.load(buscar_archivo(c, f)) for f in ARCHIVOS)


def nombres(obj):
    """Columnas con las que se ajustó un objeto de sklearn (None si se ajustó con arrays)."""
    n = getattr(obj, 'feature_names_in_', None)
    return list(n) if n is not None else None


def es_encoder(obj):
    """True si one_hot_columns.joblib guarda un OneHotEncoder (y no una lista de columnas)."""
    return hasattr(obj, 'transform') and hasattr(obj, 'categories_')


def columnas_modelo(one_hot, modelo):
    """Orden final de columnas que espera el modelo."""
    if nombres(modelo):
        return nombres(modelo)
    return None if es_encoder(one_hot) else list(one_hot)


def variables_de_entrada(one_hot, scaler, modelo):
    """Deduce de los artefactos qué variables originales hay que pedirle al usuario."""
    cols = columnas_modelo(one_hot, modelo) or []
    if es_encoder(one_hot):
        cats = list(nombres(one_hot) or CATEGORICAS)
        base = nombres(scaler) or cols
    else:
        cats = [k for k in CATEGORICAS if any(c.startswith(k + '_') for c in cols)]
        base = cols
    nums = [c for c in base if c != TARGET and not any(c.startswith(k + '_') for k in cats)]
    return nums, cats


def opciones_categoricas(var, one_hot):
    if es_encoder(one_hot):
        j = (nombres(one_hot) or list(CATEGORICAS)).index(var)
        return [str(x) for x in one_hot.categories_[j]]
    return CATEGORICAS.get(var, [])


def aplicar_one_hot(df, one_hot):
    """Paso 1: One-Hot igual que en el entrenamiento (mismas columnas y mismo orden)."""
    if es_encoder(one_hot):
        cat_cols = nombres(one_hot) or list(CATEGORICAS)
        arr = one_hot.transform(df[cat_cols])
        arr = arr.toarray() if hasattr(arr, 'toarray') else arr
        dummies = pd.DataFrame(arr, columns=one_hot.get_feature_names_out(cat_cols), index=df.index)
        return pd.concat([df.drop(columns=cat_cols), dummies], axis=1)
    cat_cols = [c for c in CATEGORICAS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, dtype=int)
    return df.reindex(columns=list(one_hot), fill_value=0)   # agrega las dummies faltantes con 0


def aplicar_scaler(df, scaler):
    """Paso 2: MinMaxScaler sólo sobre las columnas con las que se ajustó."""
    df = df.copy()
    cols = nombres(scaler)
    if cols is None:  # ajustado con array sin nombres
        n = scaler.n_features_in_
        cols = list(df.columns) if n == df.shape[1] else [c for c in df.columns if c in RANGOS][:n]
    tmp = df.reindex(columns=cols, fill_value=0).astype(float)
    esc = pd.DataFrame(scaler.transform(tmp.values), columns=cols, index=df.index)
    for c in cols:
        if c in df.columns:
            df[c] = esc[c]
    return df


def preparar(entrada, one_hot, scaler, modelo):
    """Entrada original (dict o DataFrame) -> matriz lista para el modelo."""
    df = pd.DataFrame([entrada]) if isinstance(entrada, dict) else entrada.copy()
    df = aplicar_scaler(aplicar_one_hot(df, one_hot), scaler)
    orden = columnas_modelo(one_hot, modelo) or list(df.columns)
    return df.reindex(columns=orden, fill_value=0)


def desescalar_objetivo(pred, scaler):
    """Si el scaler incluyó mpg, la predicción sale entre 0 y 1: se devuelve a mpg."""
    cols = nombres(scaler)
    if cols and TARGET in cols:
        i = cols.index(TARGET)
        return np.asarray(pred) * scaler.data_range_[i] + scaler.data_min_[i]
    return np.asarray(pred)


def predecir(entrada, one_hot, scaler, modelo):
    """Devuelve las predicciones de mpg (array) para una o varias filas."""
    X = preparar(entrada, one_hot, scaler, modelo)
    pred = modelo.predict(X if nombres(modelo) else X.values)
    return desescalar_objetivo(pred, scaler)


def predicciones_individuales(modelo, X, scaler=None):
    """Predicción de cada estimador del ensamble para una fila (rango de incertidumbre)."""
    try:
        Xv = X.values
        feats = getattr(modelo, 'estimators_features_', None)
        if feats is not None:  # BaggingRegressor
            p = [e.predict(Xv[:, f])[0] for e, f in zip(modelo.estimators_, feats)]
        else:                  # RandomForest / ExtraTrees
            p = [e.predict(Xv)[0] for e in modelo.estimators_]
        return desescalar_objetivo(p, scaler) if scaler is not None else np.array(p)
    except Exception:
        return None


# =============================================================== interfaz
import io

st.set_page_config(page_title='Predicción de consumo (mpg)', page_icon='⛽', layout='wide',
                   initial_sidebar_state='expanded')
try:  # color principal (sin necesitar .streamlit/config.toml)
    st._config.set_option('theme.primaryColor', '#0F766E')
    st._config.set_option('theme.base', 'light')
except Exception:
    pass

# ---------------------------------------------------------------- iconos SVG (dibujos propios, genéricos)
CARROCERIAS = {
    'compacto': dict(cuerpo='M18 40 L18 32 Q20 26 30 25 L40 16 Q44 13 52 13 L72 13 Q80 13 86 22 L96 26 Q102 28 102 34 L102 40 Z',
                     ventana='M43 18 L50 15.5 L70 15.5 L78 24 L41 24 Z', ruedas=[(34, 41, 7), (86, 41, 7)]),
    'sedan':    dict(cuerpo='M8 40 L8 33 Q9 28 20 27 L36 25 L48 15 Q51 13 58 13 L76 13 Q82 13 88 20 L94 25 L106 27 Q112 28 112 34 L112 40 Z',
                     ventana='M50 17 L57 15.5 L75 15.5 L86 24 L44 24 Z', ruedas=[(28, 41, 7.5), (92, 41, 7.5)]),
    'suv':      dict(cuerpo='M10 41 L10 22 Q10 12 20 11 L84 11 Q92 11 98 20 L104 26 Q110 27 110 33 L110 41 Z',
                     ventana='M18 15 L80 15 Q86 15 90 22 L18 22 Z', ruedas=[(30, 42, 9), (90, 42, 9)]),
    'pickup':   dict(cuerpo='M8 41 L8 26 L56 26 L58 12 L80 12 Q86 12 92 20 L98 25 Q110 26 110 32 L110 41 Z',
                     ventana='M62 15 L79 15 Q84 15 88 22 L60 22 Z', ruedas=[(28, 42, 8.5), (92, 42, 8.5)]),
    'deportivo': dict(cuerpo='M6 40 L6 34 Q8 29 22 28 L44 22 Q56 15 68 15 Q80 15 92 24 L108 28 Q114 30 114 36 L114 40 Z',
                      ventana='M50 23 Q58 18.5 67 18.5 Q76 18.5 86 25 Z', ruedas=[(28, 40, 7), (94, 40, 7)]),
    'clasico':  dict(cuerpo='M4 38 L4 31 Q6 27 18 27 L32 26 L42 17 Q45 15 52 15 L74 15 Q80 15 86 22 L92 26 L110 27 L116 23 L116 38 Z',
                     ventana='M45 19 L52 17 L73 17 L82 24.5 L40 25 Z', ruedas=[(26, 39, 7), (90, 39, 7)]),
}


def carro_svg(tipo, color, ancho=120):
    c = CARROCERIAS[tipo]
    ruedas = ''.join(f'<circle cx="{x}" cy="{y}" r="{r}" fill="#1E293B"/><circle cx="{x}" cy="{y}" r="{r * .42:.1f}" fill="#CBD5E1"/>'
                     for x, y, r in c['ruedas'])
    return (f'<svg viewBox="0 0 120 52" width="{ancho}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
            f'<ellipse cx="60" cy="49" rx="54" ry="2.5" fill="#0F172A" opacity=".08"/>'
            f'<path d="{c["cuerpo"]}" fill="{color}"/><path d="{c["ventana"]}" fill="#E0F2FE" opacity=".95"/>'
            f'<rect x="{c["ruedas"][1][0] + 10}" y="29" width="5" height="2.5" rx="1" fill="#FDE68A"/>{ruedas}</svg>')


SURTIDOR = ('<svg viewBox="0 0 24 24" width="{w}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
            '<rect x="3" y="3" width="11" height="18" rx="2" fill="{c}"/><rect x="5.2" y="5.5" width="6.6" height="5" rx="1" fill="#fff"/>'
            '<rect x="2" y="20" width="13" height="2" rx="1" fill="{c}"/>'
            '<path d="M14 8 h2.5 a2 2 0 0 1 2 2 v7 a1.5 1.5 0 0 0 3 0 V8.5 L19 6" stroke="{c}" stroke-width="1.6" fill="none" stroke-linecap="round"/></svg>')
GOTA = ('<svg viewBox="0 0 24 24" width="{w}" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        '<path d="M12 2 C12 2 5 10.5 5 15 a7 7 0 0 0 14 0 C19 10.5 12 2 12 2 Z" fill="{c}"/>'
        '<path d="M9 15.5 a3 3 0 0 0 3 3" stroke="#fff" stroke-width="1.6" fill="none" stroke-linecap="round"/></svg>')


def surtidor(w=22, c='#0F766E'):
    return SURTIDOR.format(w=w, c=c)


def gota(w=18, c='#0F766E'):
    return GOTA.format(w=w, c=c)


# ---------------------------------------------------------------- tipos de vehículo de ejemplo
PRESETS = {
    'Compacto urbano':   dict(tipo='compacto', color='#38BDF8', origin='Japan',  cylinders=4, horsepower=70,  acceleration=17.5, vehicle_age=5,
                              displacement=98,  weight=2100, model_year=2021, nota='Ciudad, bajo peso'),
    'Sedán familiar':    dict(tipo='sedan',    color='#818CF8', origin='Europe', cylinders=4, horsepower=110, acceleration=15.0, vehicle_age=10,
                              displacement=140, weight=2800, model_year=2016, nota='Uso diario, equilibrio'),
    'SUV mediana':       dict(tipo='suv',      color='#34D399', origin='USA',    cylinders=6, horsepower=150, acceleration=13.5, vehicle_age=8,
                              displacement=230, weight=3600, model_year=2018, nota='Espacio y altura'),
    'Pickup V8':         dict(tipo='pickup',   color='#F59E0B', origin='USA',    cylinders=8, horsepower=200, acceleration=11.0, vehicle_age=15,
                              displacement=350, weight=4500, model_year=2011, nota='Carga y potencia'),
    'Deportivo':         dict(tipo='deportivo', color='#F43F5E', origin='Europe', cylinders=6, horsepower=220, acceleration=8.5, vehicle_age=6,
                              displacement=200, weight=3000, model_year=2020, nota='Alto desempeño'),
    'Clásico americano': dict(tipo='clasico',  color='#A78BFA', origin='USA',    cylinders=8, horsepower=160, acceleration=12.0, vehicle_age=50,
                              displacement=400, weight=4300, model_year=1976, nota='Motor grande, 1970s'),
}

AYUDA = {
    'cylinders': 'Número de cilindros del motor. Más cilindros = motor más grande y, en general, más consumo.',
    'displacement': 'Volumen total del motor en pulgadas cúbicas (1 litro ≈ 61 pulg³).',
    'horsepower': 'Potencia máxima del motor en caballos de fuerza (HP).',
    'weight': 'Peso del vehículo en libras (1 kg ≈ 2,2 lb).',
    'acceleration': 'Segundos que tarda en pasar de 0 a 60 mph (0 a 97 km/h). Menos segundos = más deportivo.',
    'model_year': 'Año del modelo del vehículo.',
    'vehicle_age': 'Años de antigüedad del vehículo (2026 − año del modelo). Los más recientes suelen ser más eficientes.',
    'origin': 'Región del fabricante.',
}
COLOR_NIVEL = {'bajo': '#E11D48', 'medio': '#F59E0B', 'alto': '#10B981'}
ICONO_NIVEL = {'bajo': '🔴', 'medio': '🟡', 'alto': '🟢'}

# ---------------------------------------------------------------- estilos
st.markdown("""
<style>
:root{--ink:#0F172A; --muted:#64748B; --line:#E2E8F0; --bg:#F1F5F9; --card:#FFFFFF;
      --accent:#0F766E; --accent2:#14B8A6; --soft:#E6F4F1;}
.stApp{background:var(--bg);}
.block-container{padding-top:1.6rem; max-width:1180px;}
.stMarkdown, label{color:var(--ink);}

/* ---------- encabezado con carretera animada ---------- */
.hero{position:relative; overflow:hidden; border-radius:22px; padding:28px 32px 92px;
  background:linear-gradient(135deg,#0F766E 0%,#115E59 55%,#134E4A 100%); color:#fff; margin-bottom:18px;
  box-shadow:0 10px 30px rgba(15,118,110,.25);}
.hero h1{color:#fff; font-size:2.25rem; margin:0; font-weight:800; display:flex; gap:12px; align-items:center;}
.hero .autores{color:#99F6E4; font-weight:700; font-size:1.08rem; margin:6px 0 6px;}
.hero .desc{color:#CCFBF1; font-size:.95rem; margin:0; max-width:720px;}
.hero .gotas{position:absolute; right:30px; top:24px; display:flex; gap:10px; opacity:.9;}
.hero .gotas span{animation:flota 3s ease-in-out infinite;}
.hero .gotas span:nth-child(2){animation-delay:.6s;} .hero .gotas span:nth-child(3){animation-delay:1.2s;}
@keyframes flota{0%,100%{transform:translateY(0)} 50%{transform:translateY(-7px)}}
.carretera{position:absolute; left:0; right:0; bottom:0; height:62px; background:#1F2937;}
.carretera:before{content:""; position:absolute; left:0; right:0; top:29px; height:4px;
  background:repeating-linear-gradient(90deg,#FDE68A 0 34px,transparent 34px 70px); animation:linea 1.2s linear infinite;}
@keyframes linea{from{background-position:0 0} to{background-position:-70px 0}}
.auto{position:absolute; bottom:30px; animation:avanza 9s linear infinite;}
.auto.a2{animation-duration:13s; animation-delay:-5s;} .auto.a3{animation-duration:11s; animation-delay:-8s;}
@keyframes avanza{from{left:-160px} to{left:105%}}
.estacion{position:absolute; right:42px; bottom:34px;}

/* ---------- tarjetas y pestañas ---------- */
.stTabs [data-baseweb="tab-list"]{gap:8px; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{border-radius:12px 12px 0 0; padding:10px 20px; color:var(--muted); font-weight:700;}
.stTabs [aria-selected="true"]{color:var(--accent)!important; background:var(--soft);}
[data-baseweb="tab-highlight"]{background-color:var(--accent)!important;}
div[data-testid="stForm"]{background:var(--card); border:1px solid var(--line); border-radius:18px; padding:20px 22px;}
div[data-testid="stFileUploader"] section{background:var(--card); border:1px dashed #94A3B8; border-radius:14px;}

.stButton>button, .stFormSubmitButton>button, .stDownloadButton>button{border-radius:12px; font-weight:700;}
.stButton>button[kind="primary"], .stFormSubmitButton>button{
  background:linear-gradient(90deg,var(--accent),var(--accent2))!important; border:0!important; color:#fff!important;
  box-shadow:0 6px 16px rgba(15,118,110,.25); transition:transform .15s ease;}
.stButton>button[kind="primary"]:hover, .stFormSubmitButton>button:hover{transform:translateY(-1px);}
.stButton>button[kind="primary"] p, .stFormSubmitButton>button p{color:#fff!important;}
.stButton>button:disabled{background:#CBD5E1!important; box-shadow:none;}
.stButton>button[kind="secondary"]{border:1px solid var(--line); background:#fff; color:var(--ink);}
.stButton>button[kind="secondary"]:hover{border-color:var(--accent); color:var(--accent);}
.stDownloadButton>button{background:#fff; color:var(--accent); border:1px solid var(--accent);}
[data-baseweb="tag"]{background-color:var(--soft)!important;}
[data-baseweb="tag"] span{color:var(--accent)!important;} [data-baseweb="tag"] svg{fill:var(--accent)!important;}

.paso{display:flex; gap:10px; align-items:center; font-weight:800; color:var(--ink); font-size:1.08rem; margin:6px 0 10px;}
.paso .n{background:var(--accent); color:#fff; border-radius:99px; width:28px; height:28px; display:inline-flex;
  align-items:center; justify-content:center; font-size:.9rem;}

.modelo{background:var(--card); border:1px solid var(--line); border-radius:16px; padding:10px 8px 6px; text-align:center; min-height:158px;
  transition:all .2s ease; margin-bottom:6px;}
.modelo:hover{transform:translateY(-3px); box-shadow:0 10px 22px rgba(15,23,42,.08); border-color:#99F6E4;}
.modelo.activo{border:2px solid var(--accent); background:var(--soft);}
.modelo svg{transition:transform .3s ease;} .modelo:hover svg{transform:translateX(4px);}
.modelo .nom{font-weight:800; font-size:.9rem; color:var(--ink); margin-top:2px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.modelo .det{color:var(--muted); font-size:.74rem;}

.kpis{display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:6px 0 12px;}
.kpi{background:var(--card); border:1px solid var(--line); border-radius:16px; padding:14px 16px; animation:aparece .5s ease both;}
.kpi:nth-child(2){animation-delay:.08s} .kpi:nth-child(3){animation-delay:.16s}
@keyframes aparece{from{opacity:0; transform:translateY(8px)} to{opacity:1; transform:none}}
.kpi .lbl{color:var(--muted); font-size:.74rem; text-transform:uppercase; letter-spacing:.05em; display:flex; gap:6px; align-items:center;}
.kpi .val{font-size:1.75rem; font-weight:800; color:var(--ink); line-height:1.2;}
.kpi .sub{color:var(--muted); font-size:.8rem;}

.panel{background:var(--card); border:1px solid var(--line); border-radius:18px; padding:16px 18px; margin-bottom:12px; animation:aparece .5s ease both;}
.panel h4{margin:0 0 8px; font-size:1rem;}
.chip{display:inline-block; padding:3px 12px; border-radius:99px; font-weight:800; font-size:.8rem; color:#fff;}

.viaje{position:relative; height:70px; background:#E2E8F0; border-radius:14px; overflow:hidden; margin-top:8px;}
.viaje .via{position:absolute; left:0; right:0; bottom:16px; height:3px;
  background:repeating-linear-gradient(90deg,#94A3B8 0 16px,transparent 16px 30px);}
.viaje .recorrido{position:absolute; left:0; bottom:0; height:100%; background:linear-gradient(90deg,#CCFBF1,#99F6E4);}
.viaje .movil{position:absolute; bottom:14px; animation:recorre 2.2s cubic-bezier(.2,.7,.2,1) both;}
.viaje .meta{position:absolute; right:10px; top:8px; font-size:.75rem; color:var(--muted);}

.comp-fila{display:grid; grid-template-columns:70px 150px 1fr 70px; align-items:center; gap:10px; padding:4px 0;}
.comp-fila .nm{font-size:.86rem; font-weight:700; color:var(--ink);}
.comp-fila .barra-c{height:14px; background:#F1F5F9; border-radius:99px; overflow:hidden;}
.comp-fila .barra-c div{height:100%; border-radius:99px; animation:crece 1s ease both;}
@keyframes crece{from{width:0}}
.comp-fila .vl{font-weight:800; font-size:.86rem; text-align:right;}
.comp-fila.tuyo{background:var(--soft); border-radius:12px; padding:4px 6px;}

.consejo{border-left:4px solid var(--accent); background:#F0FDFA; padding:10px 14px; border-radius:10px; margin:6px 0; font-size:.9rem;}
.footer{color:var(--muted); font-size:.8rem; text-align:center; margin-top:28px; display:flex; gap:8px; justify-content:center; align-items:center;}
section[data-testid="stSidebar"]{background:#FFFFFF; border-right:1px solid var(--line);}
@media (max-width:760px){.hero .gotas{display:none;} .modelo{min-height:0;} .kpis{grid-template-columns:1fr;} .hero h1{font-size:1.55rem;} .comp-fila{grid-template-columns:56px 1fr 60px;} .comp-fila .nm{display:none;}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- encabezado
st.markdown(f"""
<div class="hero">
  <div class="gotas"><span>{gota(26, '#5EEAD4')}</span><span>{gota(20, '#99F6E4')}</span><span>{gota(16, '#CCFBF1')}</span></div>
  <h1>{surtidor(40, '#FFFFFF')} Predicción del consumo de combustible</h1>
  <p class="autores">Por: Andres Duque &amp; Paola Briñez</p>
  <p class="desc">Elige un tipo de vehículo o ingresa sus características y descubre cuántas millas por galón rendirá,
     cuánto consume cada 100 km y cuánto podrías gastar en combustible al año.</p>
  <div class="carretera"></div>
  <div class="auto">{carro_svg('sedan', '#FDE68A', 92)}</div>
  <div class="auto a2">{carro_svg('suv', '#5EEAD4', 100)}</div>
  <div class="auto a3">{carro_svg('deportivo', '#FCA5A5', 96)}</div>
  <div class="estacion">{surtidor(44, '#F8FAFC')}</div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def artefactos():
    return cargar_artefactos(Path(__file__).parent)   # busca los .joblib junto a app.py


try:
    one_hot, scaler, modelo = artefactos()
except FileNotFoundError as e:
    st.error(f'No se encontró {e.filename}. Copia los 3 archivos .joblib en la misma carpeta que app.py.')
    st.stop()

nums, cats = variables_de_entrada(one_hot, scaler, modelo)
ORIGENES = opciones_categoricas('origin', one_hot) if 'origin' in cats else []


def nivel_eficiencia(mpg):
    return 'alto' if mpg > 35 else 'medio' if mpg >= 25 else 'bajo'


# ---------------------------------------------------------------- estado de los campos
def valor_tipado(var, v):
    r = RANGOS.get(var, {})
    if r:
        v = min(max(v, r['min']), r['max'])
        return int(round(v)) if isinstance(r['default'], int) else float(v)
    return float(v)


for var in nums:
    st.session_state.setdefault(f'in_{var}', valor_tipado(var, RANGOS.get(var, {}).get('default', 0.0)))
for var in cats:
    st.session_state.setdefault(f'in_{var}', opciones_categoricas(var, one_hot)[0])
st.session_state.setdefault('preset', None)


def aplicar_preset(nombre):
    p = PRESETS[nombre]
    for var in nums:
        if var in p:
            st.session_state[f'in_{var}'] = valor_tipado(var, p[var])
    for var in cats:
        if var in p and p[var] in opciones_categoricas(var, one_hot):
            st.session_state[f'in_{var}'] = p[var]
    st.session_state['preset'] = nombre
    st.session_state.pop('ultimo', None)


def entrada_de_preset(nombre):
    p = PRESETS[nombre]
    e = {v: valor_tipado(v, p.get(v, RANGOS.get(v, {}).get('default', 0))) for v in nums}
    e.update({c: p.get(c, opciones_categoricas(c, one_hot)[0]) for c in cats})
    return e


@st.cache_data(show_spinner=False)
def mpg_presets():
    filas = pd.DataFrame([entrada_de_preset(n) for n in PRESETS])
    return dict(zip(PRESETS, predecir(filas, one_hot, scaler, modelo)))


# ---------------------------------------------------------------- barra lateral de ayuda
with st.sidebar:
    st.markdown(f'### {surtidor(22)} Guía rápida', unsafe_allow_html=True)
    st.markdown("""
1. **Elige un tipo de vehículo** de la galería para cargar valores de ejemplo, o escribe los tuyos.
2. **Ajusta** las características (pasa el cursor por el ícono ⓘ de cada campo para ver qué significa).
3. Pulsa **Calcular consumo**.
4. Revisa el **medidor**, la **comparación** con otros vehículos y la **calculadora de gasto**.

¿Muchos vehículos? Usa la pestaña **Predicción desde Excel**.
""")
    st.markdown('---')
    st.markdown('#### 📘 Glosario')
    st.markdown("""
- **mpg**: millas recorridas por galón. **Más alto = más eficiente.**
- **L/100 km**: litros que gasta cada 100 km. **Más bajo = más eficiente.** (L/100 km = 235,2 ÷ mpg)
- **1 galón** ≈ 3,785 litros · **1 milla** ≈ 1,609 km.
""")
    st.markdown('#### 🚦 Niveles de eficiencia')
    st.markdown('🔴 **Bajo**: menos de 25 mpg  \n🟡 **Medio**: 25 a 35 mpg  \n🟢 **Alto**: más de 35 mpg')
    st.markdown('---')
    st.caption('El modelo (Random Forest de bagging) se entrenó con 1.000 vehículos. Su error típico es de ±3 mpg; '
               'úsalo como estimación, no como medición oficial.')

tab_uno, tab_excel = st.tabs(['🚗  Predicción por vehículo', '📄  Predicción desde Excel'])

# =============================================================== pestaña 1: un vehículo
with tab_uno:
    st.markdown('<div class="paso"><span class="n">1</span> Elige un tipo de vehículo</div>', unsafe_allow_html=True)
    st.caption('Carga valores típicos de cada tipo; después puedes modificarlos en el paso 2.')
    preds_p = mpg_presets()
    cols = st.columns(len(PRESETS))
    for col, (nombre, p) in zip(cols, PRESETS.items()):
        activo = 'activo' if st.session_state['preset'] == nombre else ''
        nivel = nivel_eficiencia(preds_p[nombre])
        col.markdown(f"""
        <div class="modelo {activo}">{carro_svg(p['tipo'], p['color'], 118)}
          <div class="nom">{nombre}</div>
          <div class="det">{p['nota']}</div>
          <div class="det">{ICONO_NIVEL[nivel]} ≈ {preds_p[nombre]:.0f} mpg</div>
        </div>""", unsafe_allow_html=True)
        col.button('Usar', key=f'btn_{nombre}', on_click=aplicar_preset, args=(nombre,), use_container_width=True,
                   type='primary' if activo else 'secondary')

    izq, der = st.columns([1, 1.15], gap='large')

    with izq:
        st.markdown('<div class="paso"><span class="n">2</span> Características del vehículo</div>', unsafe_allow_html=True)
        with st.form('vehiculo'):
            c1, c2 = st.columns(2)
            for i, var in enumerate(nums):
                r = RANGOS.get(var, dict(min=None, max=None, default=0.0, step=None, label=var))
                entero = isinstance(r['default'], int)
                (c1 if i % 2 == 0 else c2).number_input(
                    r['label'], min_value=r['min'], max_value=r['max'], step=r['step'],
                    format='%d' if entero else '%.1f', key=f'in_{var}', help=AYUDA.get(var))
            for var in cats:
                st.segmented_control('Origen del fabricante' if var == 'origin' else var,
                                     opciones_categoricas(var, one_hot), key=f'in_{var}', help=AYUDA.get(var))
            enviar = st.form_submit_button('⛽  Calcular consumo', type='primary', use_container_width=True)

        if enviar:
            entrada = {v: st.session_state[f'in_{v}'] for v in nums + cats}
            if any(entrada[c] is None for c in cats):
                st.warning('Selecciona el origen del vehículo.')
            else:
                st.session_state['ultimo'] = entrada

        st.markdown('<div class="consejo">💡 <b>Consejo:</b> el número de cilindros y la antigüedad son las '
                    'variables que más influyen en la predicción. Prueba cambiarlas y observa cómo se mueve la aguja.</div>',
                    unsafe_allow_html=True)

    with der:
        st.markdown('<div class="paso"><span class="n">3</span> Resultado</div>', unsafe_allow_html=True)
        if 'ultimo' not in st.session_state:
            st.markdown(f"""
            <div class="panel" style="text-align:center; padding:34px 18px;">
              {carro_svg('sedan', '#CBD5E1', 170)}
              <p style="color:#64748B; margin:10px 0 0;">Elige un vehículo o completa el formulario y pulsa
              <b>Calcular consumo</b> para ver el resultado aquí.</p>
            </div>""", unsafe_allow_html=True)
        else:
            entrada = st.session_state['ultimo']
            X = preparar(entrada, one_hot, scaler, modelo)
            mpg = float(predecir(entrada, one_hot, scaler, modelo)[0])
            l100 = 235.215 / mpg
            km_l = mpg * 1.609 / 3.785
            nivel = nivel_eficiencia(mpg)
            tipo = PRESETS[st.session_state['preset']]['tipo'] if st.session_state['preset'] else 'sedan'

            # medidor tipo tablero de combustible (aguja animada)
            ang = -90 + min(max((mpg - 9) / (48 - 9), 0), 1) * 180
            st.markdown(f"""
            <div class="panel" style="text-align:center;">
              <style>@keyframes aguja{{from{{transform:rotate(-90deg)}} to{{transform:rotate({ang:.1f}deg)}}}}</style>
              <svg viewBox="0 0 240 140" width="100%" style="max-width:340px" xmlns="http://www.w3.org/2000/svg">
                <defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#E11D48"/>
                  <stop offset=".5" stop-color="#F59E0B"/><stop offset="1" stop-color="#10B981"/></linearGradient></defs>
                <path d="M20 120 A100 100 0 0 1 220 120" stroke="url(#g)" stroke-width="18" fill="none" stroke-linecap="round"/>
                <text x="16" y="138" font-size="11" fill="#64748B">9</text><text x="212" y="138" font-size="11" fill="#64748B">48</text>
                <g style="transform-origin:120px 120px; animation:aguja 1.4s cubic-bezier(.3,1.4,.5,1) both;">
                  <line x1="120" y1="120" x2="120" y2="36" stroke="#0F172A" stroke-width="4" stroke-linecap="round"/></g>
                <circle cx="120" cy="120" r="9" fill="#0F172A"/>
                <g transform="translate(108,62)">{surtidor(24, '#94A3B8')}</g>
              </svg>
              <div style="font-size:2.6rem; font-weight:900; color:#0F172A; line-height:1;">{mpg:.1f} <span style="font-size:1rem; color:#64748B;">mpg</span></div>
              <div style="margin-top:6px;">Eficiencia <span class="chip" style="background:{COLOR_NIVEL[nivel]}">{nivel.upper()}</span></div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="kpis">
              <div class="kpi"><div class="lbl">{gota(13)} Consumo</div><div class="val">{l100:.1f}</div><div class="sub">litros / 100 km</div></div>
              <div class="kpi"><div class="lbl">🛣️ Rendimiento</div><div class="val">{km_l:.1f}</div><div class="sub">km por litro</div></div>
              <div class="kpi"><div class="lbl">📊 vs. promedio</div><div class="val" style="color:{'#10B981' if mpg >= 29.9 else '#E11D48'}">{mpg - 29.9:+.1f}</div><div class="sub">mpg frente a 29,9</div></div>
            </div>""", unsafe_allow_html=True)

            # recorrido con 1 galón
            pct = min(mpg / 50, 1) * 100
            st.markdown(f"""
            <div class="panel"><h4>🛣️ ¿Hasta dónde llegas con 1 galón?</h4>
              <style>@keyframes recorre{{from{{left:-100px}} to{{left:calc({pct:.0f}% - 90px)}}}}</style>
              <div class="viaje"><div class="recorrido" style="width:{pct:.0f}%"></div><div class="via"></div>
                <div class="movil">{carro_svg(tipo, '#0F766E', 86)}</div>
                <div class="meta">{mpg:.0f} millas ≈ {mpg * 1.609:.0f} km</div></div>
            </div>""", unsafe_allow_html=True)

            ind = predicciones_individuales(modelo, X, scaler)
            if ind is not None and len(ind) > 1:
                p10, p90 = np.percentile(ind, [10, 90])
                st.info(f'🌲 El 80 % de los {len(ind)} árboles del modelo predicen entre **{p10:.1f} y {p90:.1f} mpg**: '
                        f'es el rango de confianza de esta estimación.')

    # ------------------------------------------------ comparación y calculadora (ancho completo)
    if 'ultimo' in st.session_state:
        st.markdown('<div class="paso"><span class="n">4</span> Compara y calcula tu gasto</div>', unsafe_allow_html=True)
        cmp_col, calc_col = st.columns([1.3, 1], gap='large')

        with cmp_col:
            filas = [('Tu vehículo', tipo, '#0F766E', mpg, True)] + \
                    [(n, p['tipo'], p['color'], preds_p[n], False) for n, p in PRESETS.items()]
            filas.sort(key=lambda f: -f[3])
            html = ''.join(f"""<div class="comp-fila {'tuyo' if t else ''}">{carro_svg(tp, col, 64)}
                       <div class="nm">{'⭐ ' if t else ''}{n}</div>
                       <div class="barra-c"><div style="width:{v / 50 * 100:.0f}%; background:{COLOR_NIVEL[nivel_eficiencia(v)]};"></div></div>
                       <div class="vl">{v:.1f} mpg</div></div>""" for n, tp, col, v, t in filas)
            st.markdown(f'<div class="panel"><h4>🏁 Comparación con otros tipos de vehículo</h4>{html}</div>',
                        unsafe_allow_html=True)

        with calc_col:
            with st.container(border=True):
                st.markdown(f'<div style="font-weight:700; font-size:1rem; margin-bottom:4px;">{surtidor(18)} Calculadora de gasto anual</div>', unsafe_allow_html=True)
                km_anio = st.slider('Kilómetros que recorres al año', 1000, 60000, 15000, step=500)
                precio = st.number_input('Precio por galón ($)', min_value=0.0, value=16000.0, step=100.0,
                                         help='Escribe el precio del galón de combustible en tu moneda.')
                galones = km_anio / (mpg * 1.609)
                costo = galones * precio
                st.markdown(f"""
                <div class="kpis" style="grid-template-columns:1fr 1fr;">
                  <div class="kpi"><div class="lbl">{gota(13)} Galones al año</div><div class="val">{galones:,.0f}</div><div class="sub">≈ {galones * 3.785:,.0f} litros</div></div>
                  <div class="kpi"><div class="lbl">💰 Gasto anual</div><div class="val">${costo:,.0f}</div><div class="sub">≈ ${costo / 12:,.0f} al mes</div></div>
                </div>""", unsafe_allow_html=True)
            mejor = max(preds_p, key=preds_p.get)
            ahorro = costo - km_anio / (preds_p[mejor] * 1.609) * precio
            if ahorro > 0:
                st.markdown(f'<div class="consejo">🌱 Con un <b>{mejor.lower()}</b> ({preds_p[mejor]:.0f} mpg) gastarías '
                            f'<b>${ahorro:,.0f} menos al año</b> en combustible.</div>', unsafe_allow_html=True)

        with st.expander('🔍 Ver datos transformados que recibe el modelo'):
            st.write('Entrada original:'); st.dataframe(pd.DataFrame([entrada]), hide_index=True)
            st.write('Después de One-Hot + MinMaxScaler:'); st.dataframe(X.round(4), hide_index=True)

# =============================================================== pestaña 2: Excel
with tab_excel:
    requeridas = nums + cats
    st.markdown('<div class="paso"><span class="n">1</span> Prepara tu archivo</div>', unsafe_allow_html=True)
    st.markdown('Sube un Excel (.xlsx) con **una fila por vehículo** y estas columnas: '
                + ', '.join(f'`{c}`' for c in requeridas) + '. Las columnas adicionales (por ejemplo, el nombre del vehículo) se conservan.')
    with st.expander('📘 ¿Qué va en cada columna?'):
        st.markdown('\n'.join(f'- **`{c}`**: {AYUDA.get(c, "")}' + (f" Valores: {', '.join(ORIGENES)}." if c == 'origin' else '')
                              for c in requeridas))
        st.caption('Si en lugar de `vehicle_age` tienes `model_year`, la app calcula la antigüedad automáticamente.')

    ejemplo = pd.DataFrame([{'vehiculo': n} | entrada_de_preset(n) for n in PRESETS])
    buf_pl = io.BytesIO(); ejemplo.to_excel(buf_pl, index=False)
    st.download_button('⬇ Descargar plantilla de Excel (con 6 vehículos de ejemplo)', buf_pl.getvalue(), 'plantilla_vehiculos.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    st.markdown('<div class="paso"><span class="n">2</span> Adjunta el Excel y predice</div>', unsafe_allow_html=True)
    archivo = st.file_uploader('Adjunta tu Excel', type=['xlsx', 'xls'])

    if st.button('⛽  Predecir desde Excel', type='primary', use_container_width=True, disabled=archivo is None):
        try:
            datos = pd.read_excel(archivo)
        except Exception as e:
            st.error(f'No se pudo leer el archivo: {e}')
            st.stop()

        claves = set(requeridas) | {'model_year'}
        datos = datos.rename(columns={c: str(c).strip().lower() for c in datos.columns
                                      if str(c).strip().lower() in claves})
        if 'vehicle_age' in requeridas and 'vehicle_age' not in datos and 'model_year' in datos:
            datos['vehicle_age'] = 2026 - datos['model_year']           # admite el año del modelo
        faltan = [c for c in requeridas if c not in datos.columns]
        if faltan:
            st.error('Faltan columnas en el Excel: ' + ', '.join(f'`{c}`' for c in faltan))
            st.stop()
        if 'origin' in datos:                                          # normaliza usa/Usa -> USA, etc.
            mapa = {o.lower(): o for o in opciones_categoricas('origin', one_hot)}
            datos['origin'] = datos['origin'].astype(str).str.strip().str.lower().map(mapa).fillna(datos['origin'])

        validas = datos[requeridas].notna().all(axis=1)
        resultado = datos.copy()
        resultado['mpg_predicho'] = np.nan
        if validas.any():
            resultado.loc[validas, 'mpg_predicho'] = predecir(datos.loc[validas, requeridas], one_hot, scaler, modelo).round(2)
        resultado['l_100km'] = (235.215 / resultado['mpg_predicho']).round(2)
        resultado['km_por_litro'] = (resultado['mpg_predicho'] * 1.609 / 3.785).round(2)
        resultado['eficiencia'] = resultado['mpg_predicho'].apply(lambda m: nivel_eficiencia(m) if pd.notna(m) else 'sin datos')
        st.session_state['resultado_excel'] = resultado
        if (~validas).any():
            st.warning(f'{int((~validas).sum())} fila(s) con datos vacíos no se pudieron predecir.')
        else:
            st.toast('¡Predicciones listas!', icon='⛽')

    if 'resultado_excel' in st.session_state:
        res = st.session_state['resultado_excel']
        ok = res['mpg_predicho'].dropna()
        if len(ok):
            st.markdown('<div class="paso"><span class="n">3</span> Resultados</div>', unsafe_allow_html=True)
            mejor_i, peor_i = ok.idxmax(), ok.idxmin()
            nombre_col = next((c for c in res.columns if c not in cats + ['eficiencia'] and
                               (pd.api.types.is_string_dtype(res[c]) or res[c].dtype == object)), None)
            etiqueta = lambda i: f'{res.loc[i, nombre_col]}' if nombre_col else f'Fila {i + 2} del Excel'
            st.markdown(f"""
            <div class="kpis" style="grid-template-columns:repeat(4,1fr);">
              <div class="kpi"><div class="lbl">🚗 Vehículos</div><div class="val">{len(ok)}</div><div class="sub">predichos</div></div>
              <div class="kpi"><div class="lbl">{gota(13)} Promedio</div><div class="val">{ok.mean():.1f}</div><div class="sub">mpg · {235.215 / ok.mean():.1f} L/100 km</div></div>
              <div class="kpi"><div class="lbl">🟢 Más eficiente</div><div class="val">{ok.max():.1f}</div><div class="sub">{etiqueta(mejor_i)}</div></div>
              <div class="kpi"><div class="lbl">🔴 Menos eficiente</div><div class="val">{ok.min():.1f}</div><div class="sub">{etiqueta(peor_i)}</div></div>
            </div>""", unsafe_allow_html=True)

            filtro = st.pills('Filtrar por eficiencia', ['alto', 'medio', 'bajo'], selection_mode='multi',
                              default=['alto', 'medio', 'bajo'],
                              format_func=lambda n: f'{ICONO_NIVEL[n]} {n.capitalize()}')
            vista = res[res['eficiencia'].isin(list(filtro or []) + ['sin datos'])]
            st.dataframe(vista, hide_index=True, use_container_width=True,
                         column_config={'mpg_predicho': st.column_config.ProgressColumn(
                             '⛽ mpg predicho', min_value=0, max_value=50, format='%.1f')})

            conteo = (res['eficiencia'].value_counts().reindex(['bajo', 'medio', 'alto']).fillna(0)
                      .rename_axis('eficiencia').reset_index(name='vehículos'))
            st.markdown('**Vehículos por nivel de eficiencia**')
            st.bar_chart(conteo, x='eficiencia', y='vehículos', color='#0F766E', horizontal=True,
                         height=170, sort=False)

            salida = io.BytesIO(); res.to_excel(salida, index=False)
            st.download_button('⬇ Descargar resultados en Excel', salida.getvalue(), 'predicciones_mpg.xlsx',
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                               use_container_width=True)

st.markdown(f'<div class="footer">{surtidor(16, "#94A3B8")} Proyecto de Machine Learning · Vehicle Fuel Consumption · '
            f'Andres Duque &amp; Paola Briñez</div>', unsafe_allow_html=True)
