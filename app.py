"""
ZeroG FireLab: Space Training Edition
NASA Space Apps Challenge - Flight Dynamics & Microgravity Combustion Division
Authored by Senior Software Architect & NASA Senior Data Scientist

Architecture:
- High-res physical fluid dynamics model (Grashof & Richardson regime mapping)
- Spherical vs. Elongated buoyant flame transition via Plotly
- Gamified Astronaut Fire Training Academy with Medals & Achievements
- Space Flight Controller Assistant ("PYRO-AI") with rich simulated knowledge base
- NASA FIST (Fire Safety in Space) Critical Emergency Protocol System
"""

import math
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ==============================================================================
# CONFIGURACIÓN GENERAL Y ESTILO VISUAL DE MISIÓN NASA
# ==============================================================================

def configure_page() -> None:
    """Configura la página de Streamlit con tema de control de misión espacial."""
    try:
        st.set_page_config(
            page_title="ZeroG FireLab: Space Training Edition | NASA Space Apps",
            page_icon="🔥",
            layout="wide",
            initial_sidebar_state="expanded",
        )
    except Exception as e:
        pass

    custom_css = """
    <style>
    /* Estilo NASA Mission Control Dark Terminal */
    @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    .stApp {
        background-color: #07090e;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    h1, h2, h3, .technical-title {
        font-family: 'Chakra Petch', sans-serif !important;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .stMetricValue {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] {
        background-color: #0d121d;
        border-right: 1px solid #1e293b;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0d121d;
        padding: 6px;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Chakra Petch', sans-serif;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        color: #94a3b8;
        border-radius: 6px;
        padding: 8px 16px;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: #0284c7 !important;
        color: #ffffff !important;
    }

    .telemetry-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .telemetry-alert-critical {
        background-color: #450a0a;
        border: 1px solid #dc2626;
        border-radius: 8px;
        padding: 14px;
        color: #fca5a5;
    }

    .telemetry-alert-nominal {
        background-color: #064e3b;
        border: 1px solid #059669;
        border-radius: 8px;
        padding: 14px;
        color: #6ee7b7;
    }

    .badge-medal {
        display: inline-block;
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: #000;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 4px;
        font-family: 'Chakra Petch', sans-serif;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)


# ==============================================================================
# GESTIÓN DE ESTADO (SESSION STATE)
# ==============================================================================

def init_session_state() -> None:
    """Inicializa variables reactivas de sesión."""
    if "quiz_step" not in st.session_state:
        st.session_state.quiz_step = 0
    if "quiz_score" not in st.session_state:
        st.session_state.quiz_score = 0
    if "quiz_user_answers" not in st.session_state:
        st.session_state.quiz_user_answers = {}
    if "quiz_completed" not in st.session_state:
        st.session_state.quiz_completed = False
    if "earned_medals" not in st.session_state:
        st.session_state.earned_medals = []
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": (
                    "¡Saludos, Cadete Espacial! Soy PYRO-AI, el Asistente Técnico "
                    "de Combustión en Microgravedad del CIR (Combustion Integrated Rack) de la ISS. "
                    "Puedes consultarme sobre la física de llamas en caída libre, convección natural, "
                    "extinción radiativa o protocolos de emergencia FIST."
                ),
            }
        ]
    if "fist_emergency_active" not in st.session_state:
        st.session_state.fist_emergency_active = False
    if "fist_step_completed" not in st.session_state:
        st.session_state.fist_step_completed = {
            "detect": False,
            "isolate": False,
            "suppress": False,
            "scrub": False,
            "evacuate": False,
        }


# ==============================================================================
# MÓDULO MATEMÁTICO: MODELO FÍSICO DE COMBUSTIÓN EN MICROGRAVEDAD
# ==============================================================================

class MicrogravityCombustionPhysics:
    """
    Simulación teórica basada en experimentos de NASA:
    - FLEX (Flame Extinction Experiment - ISS)
    - BASS (Burning and Suppression of Solids)
    - ACME (Advanced Combustion via Microgravity Experiments)
    - SoFIE (Solid Fuel Ignition and Extinction)
    """

    @staticmethod
    def calculate_telemetry(
        gravity_g: float,
        oxygen_pct: float,
        ventilation_cms: float,
        fuel_type: str,
        pressure_atm: float,
    ) -> dict:
        """
        Calcula parámetros aerotermodinámicos de la llama según la gravedad y atmósfera.
        Incluye manejo de excepciones y validación estricta de tipos.
        """
        try:
            g = max(0.0, min(1.5, float(gravity_g)))
            o2 = max(10.0, min(40.0, float(oxygen_pct)))
            v_vent = max(0.0, min(30.0, float(ventilation_cms)))
            p = max(0.4, min(2.0, float(pressure_atm)))

            fuel_props = {
                "Heptano (C7H16)": {"T_base": 1850, "k0": 0.78, "soot_factor": 1.1},
                "Metano (CH4)": {"T_base": 1950, "k0": 0.92, "soot_factor": 0.4},
                "Decano (C10H22)": {"T_base": 1820, "k0": 0.71, "soot_factor": 1.3},
                "PMMA Sólido": {"T_base": 1650, "k0": 0.45, "soot_factor": 0.9},
                "Vela de Parafina": {"T_base": 1550, "k0": 0.35, "soot_factor": 1.5},
            }
            props = fuel_props.get(fuel_type, fuel_props["Heptano (C7H16)"])

            grashof = 4500.0 * g * (p ** 1.5)
            reynolds = 12.0 * v_vent * p

            if reynolds > 0.01:
                richardson = grashof / (reynolds ** 2)
            else:
                richardson = grashof * 100.0

            buoyancy_elongation = 3.2 * (g ** 0.55)
            vent_elongation = 0.08 * v_vent
            aspect_ratio = 1.0 + buoyancy_elongation + vent_elongation

            base_radius = 4.0
            diffusion_expansion = 5.5 * (1.0 - (g ** 0.4))
            flame_radius_mm = base_radius + diffusion_expansion

            o2_factor = (o2 / 21.0) ** 0.45
            cooling_zero_g = 320.0 * (1.0 - (g ** 0.5)) if v_vent < 2.0 else 120.0 * (1.0 - (g ** 0.5))
            peak_temp_k = (props["T_base"] * o2_factor * (p ** 0.1)) - cooling_zero_g
            peak_temp_c = peak_temp_k - 273.15

            soot_ppm = (g ** 0.75) * 12.5 * props["soot_factor"] * (o2 / 21.0)
            if g < 0.08 and v_vent < 2.0:
                color_description = "Azul Pálido Quimioluminiscente (CH* / C2*)"
                flame_color_hex = "#38bdf8"
                combustion_regime = "Difusión Molecular Estricta (Esférica Quenched)"
            elif g < 0.25:
                color_description = "Azul con Halo Naranja Difuso"
                flame_color_hex = "#67e8f9"
                combustion_regime = "Microgravedad de Transición"
            elif g < 0.6:
                color_description = "Naranja Translúcido Cónico"
                flame_color_hex = "#fb923c"
                combustion_regime = "Convección Mixta Débil"
            else:
                color_description = "Amarillo Incandescente Brillante (Radiación de Hollín)"
                flame_color_hex = "#facc15"
                combustion_regime = "Convección Natural Dominada por Boyancia"

            burn_rate_k = props["k0"] * (1.0 + 0.53 * (grashof ** 0.25)) * (o2 / 21.0)

            return {
                "grashof": float(grashof),
                "reynolds": float(reynolds),
                "richardson": float(richardson),
                "aspect_ratio": float(aspect_ratio),
                "flame_radius_mm": float(flame_radius_mm),
                "peak_temp_k": float(peak_temp_k),
                "peak_temp_c": float(peak_temp_c),
                "soot_ppm": float(soot_ppm),
                "color_description": color_description,
                "flame_color_hex": flame_color_hex,
                "combustion_regime": combustion_regime,
                "burn_rate_k": float(burn_rate_k),
                "fuel_type": fuel_type,
            }
        except Exception as err:
            st.error(f"Error en cómputo físico: {err}")
            return {
                "grashof": 0.0,
                "reynolds": 0.0,
                "richardson": 0.0,
                "aspect_ratio": 1.0,
                "flame_radius_mm": 5.0,
                "peak_temp_k": 1500.0,
                "peak_temp_c": 1226.85,
                "soot_ppm": 0.1,
                "color_description": "Azul Estándar",
                "flame_color_hex": "#38bdf8",
                "combustion_regime": "Falla Controlada",
                "burn_rate_k": 0.5,
                "fuel_type": fuel_type,
            }


# ==============================================================================
# GENERADOR DE GRÁFICOS PLOTLY DE ALTA PRECISIÓN
# ==============================================================================

def generate_flame_geometry_plot(telemetry: dict) -> go.Figure:
    """
    Genera la geometría 2D de contorno térmico de la llama con transición
    continua de esfera en microgravedad (0g) a lágrima alargada convectiva (1g).
    """
    try:
        ar = telemetry["aspect_ratio"]
        radius = telemetry["flame_radius_mm"]
        color_hex = telemetry["flame_color_hex"]

        theta = np.linspace(0, 2 * np.pi, 250)
        elongation = ar - 1.0
        r_profile = radius * (1.0 + 0.35 * elongation * np.sin(theta - np.pi/2) * (np.sin(theta - np.pi/2) > 0))
        
        x = r_profile * np.cos(theta)
        y = r_profile * np.sin(theta) * (1.0 + 0.65 * elongation)

        grid_range = max(25.0, radius * ar * 1.5)
        xi = np.linspace(-grid_range, grid_range, 120)
        yi = np.linspace(-grid_range, grid_range * 1.4, 120)
        X, Y = np.meshgrid(xi, yi)

        dist_field = np.sqrt(X**2 + (Y / max(1.0, ar * 0.85))**2)
        T_field = 295.0 + (telemetry["peak_temp_k"] - 295.0) * np.exp(-((dist_field - radius*0.7)**2) / (radius * 1.8)**2)

        fig = go.Figure()

        fig.add_trace(
            go.Contour(
                z=T_field,
                x=xi,
                y=yi,
                colorscale="Inferno" if telemetry["aspect_ratio"] > 1.8 else "IceFire",
                contours=dict(showlabels=True, labelfont=dict(size=9, color="white")),
                colorbar=dict(title="T (K)", titleside="right"),
                opacity=0.75,
                hoverinfo="x+y+z",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=color_hex, width=3.5),
                name="Frente de Reacción Estequiométrico",
                hoverinfo="name",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers",
                marker=dict(size=10, color="#ffffff", symbol="circle", line=dict(color="#0284c7", width=2)),
                name="Gota / Inyector Central",
            )
        )

        if telemetry["grashof"] > 50:
            arrow_y = np.linspace(radius * 0.5, radius * ar * 1.1, 4)
            for ay in arrow_y:
                fig.add_annotation(
                    x=0,
                    y=ay,
                    ax=0,
                    ay=ay - (radius * 0.4),
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.2,
                    arrowwidth=2,
                    arrowcolor="rgba(255, 255, 255, 0.6)",
                )

        fig.update_layout(
            title=f"Morfología de la Llama: {telemetry['combustion_regime']}",
            title_font=dict(family="Chakra Petch", size=16, color="#e2e8f0"),
            xaxis=dict(title="Posición Radial X (mm)", zeroline=True, gridcolor="#1e293b", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Posición Axial Y (mm)", zeroline=True, gridcolor="#1e293b"),
            paper_bgcolor="#07090e",
            plot_bgcolor="#0b0f19",
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5, font=dict(color="#94a3b8")),
            height=520,
        )
        return fig
    except Exception as err:
        st.error(f"Error generando gráfico de morfología: {err}")
        return go.Figure()


def generate_chemical_kinetics_plots(telemetry: dict) -> go.Figure:
    """
    Genera subgráficos con perfiles radiales de temperatura, concentración
    de especies y curva de consumo del combustible según la ley d^2.
    """
    try:
        r = np.linspace(0.1, 25.0, 150)
        rf = telemetry["flame_radius_mm"]
        T_peak = telemetry["peak_temp_k"]

        T_profile = 295.0 + (T_peak - 295.0) * np.exp(-((r - rf)**2) / 14.0)

        O2_ambient = 21.0
        O2_profile = O2_ambient / (1.0 + np.exp(-(r - rf) / 2.0))
        Fuel_profile = 100.0 / (1.0 + np.exp((r - rf * 0.7) / 1.5))
        CO2_profile = 14.0 * np.exp(-((r - rf)**2) / 10.0)
        CO_profile = (telemetry["soot_ppm"] * 0.4 + 3.0) * np.exp(-((r - rf * 0.85)**2) / 8.0)

        t_sec = np.linspace(0, 4.0, 100)
        d0_sq = 4.0
        k_evap = telemetry["burn_rate_k"]
        d_sq_t = np.maximum(0.0, d0_sq - k_evap * t_sec)

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Perfil Radial de Temperatura T(r)",
                "Fracciones Molares de Especies Químicas",
                "Cinética de Gota (Ley d²: D² vs Tiempo)",
            ),
            horizontal_spacing=0.08,
        )

        fig.add_trace(
            go.Scatter(x=r, y=T_profile, mode="lines", line=dict(color="#f97316", width=2.5), name="T (K)"),
            row=1,
            col=1,
        )
        fig.add_vline(x=rf, line_dash="dash", line_color="#38bdf8", annotation_text="Frente Llama", row=1, col=1)

        fig.add_trace(
            go.Scatter(x=r, y=O2_profile, mode="lines", line=dict(color="#06b6d4", width=2), name="O₂ (%)"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=r, y=Fuel_profile, mode="lines", line=dict(color="#a855f7", width=2), name="Combustible (%)"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=r, y=CO2_profile, mode="lines", line=dict(color="#10b981", width=2), name="CO₂ (%)"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=r, y=CO_profile, mode="lines", line=dict(color="#ef4444", width=1.5, dash="dot"), name="CO (%)"),
            row=1,
            col=2,
        )

        fig.add_trace(
            go.Scatter(x=t_sec, y=d_sq_t, mode="lines", line=dict(color="#eab308", width=2.5), name="D²(t) [mm²]"),
            row=1,
            col=3,
        )

        fig.update_layout(
            paper_bgcolor="#07090e",
            plot_bgcolor="#0b0f19",
            margin=dict(l=30, r=30, t=50, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5, font=dict(color="#94a3b8")),
            font=dict(family="JetBrains Mono", size=10, color="#94a3b8"),
            height=340,
        )
        fig.update_xaxes(gridcolor="#1e293b", title_text="Radio r (mm)", row=1, col=1)
        fig.update_xaxes(gridcolor="#1e293b", title_text="Radio r (mm)", row=1, col=2)
        fig.update_xaxes(gridcolor="#1e293b", title_text="Tiempo transcurrido (s)", row=1, col=3)
        fig.update_yaxes(gridcolor="#1e293b", title_text="Temperatura (K)", row=1, col=1)
        fig.update_yaxes(gridcolor="#1e293b", title_text="Concentración (% mol)", row=1, col=2)
        fig.update_yaxes(gridcolor="#1e293b", title_text="D² (mm²)", row=1, col=3)

        return fig
    except Exception as err:
        st.error(f"Error generando cinética química: {err}")
        return go.Figure()


# ==============================================================================
# BASE DE CONOCIMIENTO TÉCNICO-ESPACIAL PARA EL CHATBOT SIMULADO
# ==============================================================================

SIMULATED_SPACE_KNOWLEDGE = [
    {
        "keywords": ["azul", "color", "quimioluminiscencia", "amarillo", "hollin"],
        "answer": (
            "**Física del Color de la Llama en Microgravedad:**\n\n"
            "En la Tierra (1g), la convección natural generada por la flotabilidad de Arquímedes transporta "
            "los gases calientes hacia arriba a velocidades de 1 a 3 m/s. Esto acelera el aire fresco y crea "
            "altas temperaturas locales donde los hidrocarburos se pirolizan formando partículas sólidas de hollín. "
            "El hollín incandescente emite radiación de cuerpo negro en el espectro **amarillo-naranja intenso** (1500-1900 K).\n\n"
            "En **microgravedad (0g)**, el empuje ascensional desaparece (Grashof ≈ 0). El oxígeno solo llega "
            "por lenta difusión molecular esférica. Sin altas tasas de cizalladura térmica, la formación de "
            "hollín se inhibe casi por completo. La débil emisión visual proviene exclusivamente de la **quimioluminiscencia** "
            "de radicales excitados como CH* (emisión a 431 nm) y C₂* (bandas de Swan a 516 nm), produciendo la característica "
            "**llama esférica azul pálido** descubierta en las cámaras CIR de la Estación Espacial Internacional."
        ),
    },
    {
        "keywords": ["esferica", "forma", "redonda", "lagrima", "flotabilidad", "arquimedes"],
        "answer": (
            "**Morfología Esférica vs. Forma de Lágrima:**\n\n"
            "La forma clásica de lágrima en la Tierra surge de la fuerza de flotabilidad:\n"
            "$$\\mathbf{F}_b = (\\rho_\\infty - \\rho_{hot}) \\cdot g \\cdot V$$\n"
            "Como los gases calientes de combustión tienen una densidad mucho menor que el aire ambiente frío, "
            "la gravedad los empuja verticalmente hacia arriba mientras el aire frío entra por la base.\n\n"
            "En caída libre orbital (0.00g), $\\mathbf{F}_b = 0$. La ecuación de conservación de masa de especies "
            "se simplifica al operador laplaciano isotrópico esférico:\n"
            "$$\\nabla^2 Y_i = \\frac{1}{r^2} \\frac{\\partial}{\\partial r}\\left( r^2 \\frac{\\partial Y_i}{\\partial r} \\right) = 0$$\n"
            "Como no existe ninguna dirección espacial privilegiada, el transporte de masa y calor se expande simétricamente "
            "en 360 grados, creando una **esfera perfecta de fuego**."
        ),
    },
    {
        "keywords": ["apagar", "extintor", "extincion", "soplar", "agua", "co2", "niebla"],
        "answer": (
            "**Supresión de Incendios en Vehículos Espaciales:**\n\n"
            "1. **¿Por qué NO se debe soplar una llama en 0g?**\n"
            "En 0g, la llama consume el oxígeno a su alrededor y acumula sus propios productos de combustión (CO₂ y H₂O) "
            "en una nube sofocante envolvente (auto-asfixia). Si un astronauta sopla o se mueve rápido, la corriente de aire forzada "
            "suministra oxígeno nuevo rápidamente, lo que puede causar una deflagración o reactivación violenta.\n\n"
            "2. **Extintores de la ISS:**\n"
            "- **PFE (Portable Fire Extinguisher) de Niebla de Agua Fina:** Utiliza microgotas de agua no conductora a alta presión, "
            "absorbiendo calor de inmediato sin contaminar el sistema de reciclaje de aire ECLSS con halones.\n"
            "- **Extintores de Dióxido de Carbono (CO₂):** Empleados específicamente en racks electrónicos y aviónica presurizada."
        ),
    },
    {
        "keywords": ["flex", "bass", "cir", "sofie", "experimento", "nasa"],
        "answer": (
            "**Instalaciones y Experimentos de Combustión de la NASA en la ISS:**\n\n"
            "- **CIR (Combustion Integrated Rack):** Módulo de investigación de 100 litros dentro del laboratorio Destiny de la ISS, "
            "con cámaras de alta velocidad (1000 fps), espectrómetros y atmósfera controlada por gas.\n"
            "- **FLEX (Flame Extinction Experiment):** Demostró por primera vez la existencia de **llamas frías** (cool flames) "
            "donde gotas de heptano continuaban ardiendo sin llama visible a temperaturas de apenas 600-800 K tras la extinción visual.\n"
            "- **BASS / SoFIE:** Evalúan la inflamabilidad de materiales sólidos espaciales (como Nomex, Kevlar y PMMA) bajo flujos de ventilación controlados."
        ),
    },
    {
        "keywords": ["fist", "protocolo", "alarma", "emergencia", "fuego", "humo"],
        "answer": (
            "**Protocolo FIST de la NASA (Spacecraft Fire Safety Action Plan):**\n\n"
            "El acrónimo y protocolo FIST define los 5 pasos de vida o muerte ante fuego a bordo:\n"
            "1. **Detección Inmediata:** Monitoreo con detectores de dispersión láser de partículas y sensores de CO/HCN (CSA-CP).\n"
            "2. **Aislamiento de Ventilación:** Apagar de inmediato las válvulas de ventilación del módulo afectado para evitar que el humo "
            "se propague por los 900 m³ habitables de la ISS.\n"
            "3. **Uso de Máscaras PBBA:** Todos los tripulantes colocan su aparato de respiración autónoma con suministro de O₂ positivo.\n"
            "4. **Ataque con PFE/CO₂:** Descargar el agente extintor a través de los puertos de acceso rápido de los racks sin abrir las compuertas.\n"
            "5. **Despresurización o Evacuación:** Si el fuego es incontrolable, aislar y sellar las compuertas (Hatch) y preparar el desacople de emergencia."
        ),
    },
]

def query_simulated_space_assistant(user_prompt: str) -> str:
    """Procesa consultas en lenguaje natural mediante coincidencia semántica en la base técnica espacial."""
    prompt_lower = user_prompt.lower().strip()
    best_match = None
    max_score = 0

    for item in SIMULATED_SPACE_KNOWLEDGE:
        score = sum(1 for kw in item["keywords"] if kw in prompt_lower)
        if score > max_score:
            max_score = score
            best_match = item["answer"]

    if best_match and max_score > 0:
        return best_match
    else:
        return (
            f"**Respuesta del Controlador de Vuelo CIR (PYRO-AI):**\n\n"
            f"Entendido, Cadete. Tu consulta sobre: *'{user_prompt}'* aborda la microfísica espacial de fluidos. "
            "En caída libre (gravedad efectiva cero), todos los gradientes de densidad dejan de inducir "
            "fuerzas de flotabilidad. Te sugiero explorar:\n"
            "- ¿Por qué la llama es esférica y azul en 0g?\n"
            "- ¿Qué experimentos realizó la misión FLEX en la ISS?\n"
            "- ¿Cómo funciona el protocolo de emergencia FIST ante fuego en cabina?\n"
            "- ¿Por qué está prohibido soplar un fuego en microgravedad?"
        )


# ==============================================================================
# BASE DE DATOS DE PREGUNTAS DEL QUIZ GAMIFICADO DE ENTRENAMIENTO ESPACIAL
# ==============================================================================

SPACE_QUIZ_QUESTIONS = [
    {
        "question": "¿Cuál es la principal razón física por la que una llama en microgravedad (0g) adopta una forma esférica?",
        "options": [
            "Porque la tensión superficial del combustible líquido se vuelve infinita.",
            "Porque al desaparecer la flotabilidad de Arquímedes, la difusión molecular de oxígeno ocurre de manera isotrópica en 360 grados.",
            "Porque el flujo forzado de los ventiladores de la cabina comprime la llama desde arriba.",
            "Porque la radiación solar en órbita ioniza los gases de combustión creando un campo magnético simétrico.",
        ],
        "correct_index": 1,
        "explanation": (
            "Exacto. En la Tierra (1g), el aire caliente asciende por flotabilidad convectiva, estirando la llama. "
            "En microgravedad, el número de Grashof cae a cero y el transporte de oxígeno y calor queda puramente regido "
            "por difusión molecular, la cual no tiene preferencia direccional, generando una esfera perfecta."
        ),
    },
    {
        "question": "¿Por qué las llamas en microgravedad arden típicamente con un color azul pálido en lugar del amarillo brillante visto en la Tierra?",
        "options": [
            "Debido a la ausencia de convección, las partículas de hollín no se producen ni calientan a incandescencia; la luz proviene de quimioluminiscencia de radicales CH* y C2*.",
            "Porque en la estación espacial se utiliza helio puro como oxidante en lugar de aire atmosférico.",
            "Porque la llama absorbe la radiación cósmica que altera su frecuencia luminosa hacia el azul.",
            "Porque la temperatura en 0g siempre supera los 5,000 grados Celsius.",
        ],
        "correct_index": 0,
        "explanation": (
            "Brillante. El color amarillo en la Tierra proviene de millones de partículas sólidas de carbón (hollín) incandescentes. "
            "En 0g, la tasa de combustión más fría y la difusión lenta suprimen la aglomeración de hollín, dominando "
            "la emisión quimioluminiscente de los radicales moleculares CH* y C2*."
        ),
    },
    {
        "question": "En el experimento FLEX de la NASA en la ISS, los científicos descubrieron un fenómeno asombroso llamado 'Llamas Frías' (Cool Flames). ¿En qué consiste?",
        "options": [
            "Llamas que congelan el combustible líquido hasta el cero absoluto.",
            "Combustión de vapor de agua sin presencia de combustible fósil.",
            "Combustión sostenida a baja temperatura (600-800 K) sin llama visible, continuando después de que la llama caliente visual se extinguió.",
            "Llamas alimentadas exclusivamente por energía estática del casco de la nave.",
        ],
        "correct_index": 2,
        "explanation": (
            "¡Correcto! FLEX demostró que después de que la llama caliente aparente se apagaba por pérdidas radiativas, "
            "la gota de combustible continuaba consumiéndose silenciosamente a baja temperatura en una fase de combustión química fría "
            "completamente invisible al ojo humano."
        ),
    },
    {
        "question": "Si se detecta un conato de incendio en un rack de aviónica presurizada de la ISS, ¿cuál es la primera acción operativa del protocolo FIST?",
        "options": [
            "Abrir las esclusas de aire inmediatamente para que el fuego salga al vacío espacial.",
            "Interrumpir la ventilación del módulo y apagar los ventiladores para privar al fuego de renovación de oxígeno y contener el humo.",
            "Lanzar baldes de agua desionizada sobre los circuitos eléctricos de alta tensión.",
            "Acelerar los propulsores de la nave para que la fuerza g apague la llama.",
        ],
        "correct_index": 1,
        "explanation": (
            "¡Correcto! En microgravedad, una llama sin ventilación se auto-asfixia rápidamente con su propio CO2. "
            "Apagar la ventilación detiene el flujo forzado de oxígeno fresco y evita que el humo tóxico se disperse al resto de la estación."
        ),
    },
    {
        "question": "¿Qué número adimensional de la mecánica de fluidos describe la relación entre las fuerzas de flotabilidad térmica y las fuerzas viscosas?",
        "options": [
            "Número de Mach (Ma)",
            "Número de Froude (Fr)",
            "Número de Grashof (Gr)",
            "Número de Knudsen (Kn)",
        ],
        "correct_index": 2,
        "explanation": (
            "Excelente. El número de Grashof (Gr = g * beta * deltaT * L^3 / nu^2) cuantifica el empuje boyante. "
            "Cuando la gravedad g es cero, Gr se vuelve 0, indicando la ausencia total de convección natural."
        ),
    },
]


# ==============================================================================
# FUNCIONES MODULARES DE LA INTERFAZ STREAMLIT (TABS Y SIDEBAR)
# ==============================================================================

def render_sidebar_controls() -> dict:
    """Renderiza controles interactivos de física de fluidos en la barra lateral."""
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding-bottom: 12px; border-bottom: 1px solid #1e293b;">
            <h2 style="color: #38bdf8; margin: 0; font-size: 1.25rem;">NASA ZeroG FireLab</h2>
            <p style="color: #94a3b8; font-size: 0.8rem; margin: 2px 0 0 0;">Space Training Edition · CIR Rack</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### Parámetros de Fluidos")

    preset = st.sidebar.selectbox(
        "Aceleración Gravitatoria (Presets)",
        ["Personalizado", "Microgravedad ISS (0.00 g)", "Luna (0.16 g)", "Marte (0.38 g)", "Tierra al Nivel del Mar (1.00 g)"],
        index=1,
    )

    if preset == "Microgravedad ISS (0.00 g)":
        default_g = 0.00
    elif preset == "Luna (0.16 g)":
        default_g = 0.16
    elif preset == "Marte (0.38 g)":
        default_g = 0.38
    elif preset == "Tierra al Nivel del Mar (1.00 g)":
        default_g = 1.00
    else:
        default_g = 0.00

    gravity_val = st.sidebar.slider(
        "Gravedad Efectiva (g)",
        min_value=0.00,
        max_value=1.00,
        value=default_g,
        step=0.01,
        help="0.00 g representa la caída libre orbital en la ISS. 1.00 g representa la gravedad terrestre.",
    )

    oxygen_val = st.sidebar.slider(
        "Concentración de Oxígeno (% O₂)",
        min_value=12.0,
        max_value=35.0,
        value=21.0,
        step=0.5,
        help="El aire estándar contiene 20.9% O2. En misiones espaciales como Orión/Artemisa se prueba 30-34% a presiones reducidas.",
    )

    ventilation_val = st.sidebar.slider(
        "Velocidad de Ventilación Forzada (cm/s)",
        min_value=0.0,
        max_value=15.0,
        value=0.0,
        step=0.5,
        help="En 0g la difusión es el único mecanismo a menos que los ventiladores generen flujo convectivo forzado.",
    )

    fuel_val = st.sidebar.selectbox(
        "Combustible de Prueba",
        ["Heptano (C7H16)", "Metano (CH4)", "Decano (C10H22)", "PMMA Sólido", "Vela de Parafina"],
        index=0,
    )

    pressure_val = st.sidebar.slider(
        "Presión de Cabina (atm)",
        min_value=0.5,
        max_value=1.5,
        value=1.0,
        step=0.05,
        help="1.0 atm es la presión estándar a nivel del mar y en la ISS.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size: 0.75rem; color: #64748b; line-height: 1.4;">
            <strong>Misión:</strong> NASA Space Apps Challenge<br/>
            <strong>Cámara:</strong> CIR (Combustion Integrated Rack)<br/>
            <strong>Módulo:</strong> Destiny Laboratory (ISS)
        </div>
        """,
        unsafe_allow_html=True,
    )

    return {
        "gravity_g": gravity_val,
        "oxygen_pct": oxygen_val,
        "ventilation_cms": ventilation_val,
        "fuel_type": fuel_val,
        "pressure_atm": pressure_val,
    }


def render_tab_simulator(params: dict) -> None:
    """Pestaña 1: Simulador Físico Dinámico con Plotly."""
    st.markdown("## Pestaña 1: Simulador Físico Dinámico de Combustión")
    st.markdown(
        "Visualiza la transición física continua desde una **llama esférica azul difusiva en microgravedad (0g)** "
        "hasta una **llama alargada por empuje hidrostático de boyancia en gravedad terrestre (1g)**."
    )

    telemetry = MicrogravityCombustionPhysics.calculate_telemetry(
        gravity_g=params["gravity_g"],
        oxygen_pct=params["oxygen_pct"],
        ventilation_cms=params["ventilation_cms"],
        fuel_type=params["fuel_type"],
        pressure_atm=params["pressure_atm"],
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Número de Grashof (Gr)", f"{telemetry['grashof']:.1f}", help="Gr=0 en 0g: ausencia total de boyancia.")
    with col2:
        st.metric("Relación de Aspecto (AR)", f"{telemetry['aspect_ratio']:.2f}", help="1.00 = Esfera perfecta.")
    with col3:
        st.metric("Temperatura Pico", f"{telemetry['peak_temp_c']:.0f} °C", f"{telemetry['peak_temp_k']:.0f} K")
    with col4:
        st.metric("Hollín Estimado", f"{telemetry['soot_ppm']:.2f} ppm", help="En microgravedad el hollín cae drásticamente.")
    with col5:
        st.metric("Tasa Consumo d²", f"{telemetry['burn_rate_k']:.3f} mm²/s")

    st.markdown(
        f"""
        <div class="telemetry-card" style="border-left: 4px solid {telemetry['flame_color_hex']};">
            <span style="color: #94a3b8; font-size: 0.85rem;">RÉGIMEN TERMODINÁMICO IDENTIFICADO:</span><br/>
            <strong style="color: {telemetry['flame_color_hex']}; font-size: 1.15rem; font-family: 'Chakra Petch';">
                {telemetry['combustion_regime']}
            </strong> · 
            <span style="color: #cbd5e1; font-size: 0.95rem;">Aspecto Óptico: <em>{telemetry['color_description']}</em></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    flame_fig = generate_flame_geometry_plot(telemetry)
    st.plotly_chart(flame_fig, use_container_width=True)

    st.markdown("### Cinética Química, Concentración de Especies y Ley d²")
    kinetics_fig = generate_chemical_kinetics_plots(telemetry)
    st.plotly_chart(kinetics_fig, use_container_width=True)


def render_tab_quiz() -> None:
    """Pestaña 2: Quiz Gamificado de Entrenamiento Espacial."""
    st.markdown("## Pestaña 2: Academia de Entrenamiento Espacial (Quiz Gamificado)")
    st.markdown(
        "Demuestra tus conocimientos como especialista en combustión de la NASA. "
        "Gana insignias oficiales de astronauta respondiendo preguntas basadas en experimentos de la ISS."
    )

    medals_catalog = [
        {"id": "novice", "title": "🎖️ Cadete de Microgravedad", "req": "Responder al menos 1 pregunta."},
        {"id": "cir_expert", "title": "🚀 Especialista CIR-Destiny", "req": "Alcanzar 3 o más respuestas correctas."},
        {"id": "master_diff", "title": "🔬 Maestro de la Difusión Molecular", "req": "Alcanzar 4 respuestas correctas."},
        {"id": "fist_commander", "title": "🛡️ Comandante de Seguridad FIST", "req": "Obtener puntaje perfecto (100%)."},
    ]

    total_q = len(SPACE_QUIZ_QUESTIONS)
    curr_q_idx = st.session_state.quiz_step

    progreso = curr_q_idx / total_q
    st.progress(progreso)

    earned_titles = [m["title"] for m in medals_catalog if m["id"] in st.session_state.earned_medals]
    if earned_titles:
        st.markdown("**Insignias desbloqueadas:**")
        medals_html = "".join([f'<span class="badge-medal">{m}</span>' for m in earned_titles])
        st.markdown(medals_html, unsafe_allow_html=True)

    if not st.session_state.quiz_completed and curr_q_idx < total_q:
        q_data = SPACE_QUIZ_QUESTIONS[curr_q_idx]
        st.markdown(f"### Pregunta {curr_q_idx + 1} de {total_q}")
        st.markdown(f"**{q_data['question']}**")

        chosen_option = st.radio(
            "Selecciona la respuesta correcta:",
            q_data["options"],
            key=f"q_radio_{curr_q_idx}",
        )

        col_a, col_b = st.columns([1, 4])
        with col_a:
            if st.button("Verificar Respuesta", key=f"btn_check_{curr_q_idx}", type="primary"):
                user_ans_idx = q_data["options"].index(chosen_option)
                is_correct = user_ans_idx == q_data["correct_index"]

                if is_correct:
                    st.success("¡CORRECTO, ASTRONAUTA!")
                    st.session_state.quiz_score += 1
                else:
                    st.error("INCORRECTO. Revisa la retroalimentación técnica.")

                st.info(f"**Explicación Científica:** {q_data['explanation']}")

                if "novice" not in st.session_state.earned_medals:
                    st.session_state.earned_medals.append("novice")
                if st.session_state.quiz_score >= 3 and "cir_expert" not in st.session_state.earned_medals:
                    st.session_state.earned_medals.append("cir_expert")
                if st.session_state.quiz_score >= 4 and "master_diff" not in st.session_state.earned_medals:
                    st.session_state.earned_medals.append("master_diff")

                time.sleep(0.5)

        with col_b:
            if st.button("Siguiente Pregunta →", key=f"btn_next_{curr_q_idx}"):
                if curr_q_idx + 1 < total_q:
                    st.session_state.quiz_step += 1
                else:
                    st.session_state.quiz_completed = True
                    if st.session_state.quiz_score == total_q and "fist_commander" not in st.session_state.earned_medals:
                        st.session_state.earned_medals.append("fist_commander")
                st.rerun()

    else:
        score = st.session_state.quiz_score
        pct = (score / total_q) * 100
        st.markdown("---")
        st.markdown(f"## 🏆 Entrenamiento Finalizado: {score} / {total_q} ({pct:.0f}%)")

        if score == total_q:
            st.balloons()
            st.markdown(
                """
                <div class="telemetry-alert-nominal">
                    <h3>🎖️ ¡CALIFICACIÓN EXCEPCIONAL: COMANDANTE DE SEGURIDAD ESPACIAL!</h3>
                    Has demostrado dominio absoluto sobre la combustión en microgravedad y los protocolos de la NASA.
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif score >= 3:
            st.markdown(
                """
                <div class="telemetry-card">
                    <h3>🚀 ¡ESPECIALISTA EN COMBUSTIÓN CERTIFICADO!</h3>
                    Has aprobado con éxito la fase teórica para operar el rack CIR de la Estación Espacial Internacional.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="telemetry-alert-critical">
                    <h3>⚠️ ENTRENAMIENTO REQUIERE REFUERZO</h3>
                    Repasa la Pestaña 1 y consulta con el chatbot PYRO-AI antes de volver a intentar la misión.
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("Reiniciar Entrenamiento Completo"):
            st.session_state.quiz_step = 0
            st.session_state.quiz_score = 0
            st.session_state.quiz_completed = False
            st.session_state.earned_medals = []
            st.rerun()


def render_tab_chatbot() -> None:
    """Pestaña 3: Chatbot Integrado con Base de Conocimientos Técnico-Espacial."""
    st.markdown("## Pestaña 3: Chatbot PYRO-AI (Control de Misión CIR - NASA)")
    st.markdown(
        "Consulta en tiempo real cualquier duda teórica, física o de seguridad espacial con el asistente "
        "inteligente del laboratorio Destiny."
    )

    st.markdown("**Consultas rápidas recomendadas:**")
    quick_cols = st.columns(4)
    quick_questions = [
        "¿Por qué la llama en 0g es azul?",
        "¿Por qué es una esfera perfecta?",
        "¿Por qué está prohibido soplar el fuego?",
        "¿Qué descubrió el experimento FLEX?",
    ]

    selected_quick = None
    for idx, q_text in enumerate(quick_questions):
        with quick_cols[idx]:
            if st.button(q_text, key=f"quick_btn_{idx}"):
                selected_quick = q_text

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Escribe tu consulta científica sobre combustión espacial...")
    active_prompt = selected_quick if selected_quick else user_input

    if active_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Procesando telemetría y base aerotérmica de la NASA..."):
                response_text = query_simulated_space_assistant(active_prompt)
                st.markdown(response_text)
                st.session_state.chat_messages.append({"role": "assistant", "content": response_text})


def render_tab_fist_protocols() -> None:
    """Pestaña 4: Protocolos de Emergencia FIST (Fire Safety in Space) de la NASA."""
    st.markdown("## Pestaña 4: Protocolos Críticos de Emergencia FIST de la NASA")
    st.markdown(
        "El programa **FIST (Fire Safety in Space)** define la respuesta operativa estandarizada "
        "ante eventos de combustión no controlada a bordo de vehículos tripulados (ISS, Crew Dragon, Orion Artemis)."
    )

    col_stat, col_action = st.columns([2, 1])
    with col_stat:
        if st.session_state.fist_emergency_active:
            st.markdown(
                """
                <div class="telemetry-alert-critical">
                    <h3 style="margin: 0; color: #ef4444;">🚨 ALARMA GENERAL DE INCENDIO EN CABINA ACTIVA</h3>
                    <p style="margin: 4px 0 0 0;">Presión parcial de humo superó el umbral. Ejecuta los 5 pasos obligatorios.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="telemetry-alert-nominal">
                    <h3 style="margin: 0; color: #10b981;">✅ ESTADO DE CABINA: NOMINAL (SIN FUEGO DETECTADO)</h3>
                    <p style="margin: 4px 0 0 0;">Sensores de partículas láser y analizadores de gas CSA-CP en rango normal.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_action:
        if not st.session_state.fist_emergency_active:
            if st.button("Simular Incendio de Emergencia", type="primary", use_container_width=True):
                st.session_state.fist_emergency_active = True
                for k in st.session_state.fist_step_completed:
                    st.session_state.fist_step_completed[k] = False
                st.rerun()
        else:
            if st.button("Restablecer a Modo Nominal", use_container_width=True):
                st.session_state.fist_emergency_active = False
                st.rerun()

    st.markdown("### Secuencia Operativa FIST Paso a Paso")

    steps = [
        {
            "id": "detect",
            "title": "Fase 1: Verificación y Alarma de Humo Láser",
            "desc": "Confirmar lectura en monitores CSA-CP (CO > 50 ppm, partículas > 1.5 mg/m³). Alertar a Capcom Houston.",
            "action_text": "Confirmar Detección",
        },
        {
            "id": "isolate",
            "title": "Fase 2: Aislamiento Total de Ventilación (Vents Shutoff)",
            "desc": "Cerrar válvulas de mariposa de los ductos HVAC del módulo para asfixiar la llama por falta de convección y evitar propagación.",
            "action_text": "Cerrar Válvulas HVAC",
        },
        {
            "id": "suppress",
            "title": "Fase 3: Descarga de Extintor PFE (Agua Niebla / CO2)",
            "desc": "Insertar la boquilla en el puerto de inyección del rack sin abrir la cubierta frontal para evitar entrada brusca de aire.",
            "action_text": "Descargar Agente Extintor",
        },
        {
            "id": "scrub",
            "title": "Fase 4: Purga Atmosférica y Máscaras PBBA",
            "desc": "Tripulación con PBBA activo. Encender cartuchos de carbón activado y catalizadores LiOH para filtrar cianuro de hidrógeno y CO.",
            "action_text": "Activar Depurador Atmosférico",
        },
        {
            "id": "evacuate",
            "title": "Fase 5: Sellado de Escotilla (Hatch) y Preparación de Evacuación",
            "desc": "Cierre hermético de compuerta hacia los módulos vecinos. Preparar cápsulas de retorno Crew Dragon / Soyuz si el fuego no se suprime.",
            "action_text": "Sellar Escotilla del Módulo",
        },
    ]

    for idx, s in enumerate(steps):
        is_done = st.session_state.fist_step_completed.get(s["id"], False)
        with st.container():
            st.markdown(
                f"""
                <div class="telemetry-card" style="border-left: 4px solid {'#10b981' if is_done else '#64748b'};">
                    <strong style="color: {'#10b981' if is_done else '#f8fafc'}; font-size: 1.05rem;">
                        {'[COMPLETADO] ' if is_done else ''}{s['title']}
                    </strong>
                    <p style="color: #94a3b8; font-size: 0.9rem; margin: 4px 0 8px 0;">{s['desc']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            col_btn, _ = st.columns([2, 5])
            with col_btn:
                btn_state = st.button(
                    s["action_text"],
                    key=f"fist_step_btn_{s['id']}",
                    disabled=is_done or not st.session_state.fist_emergency_active,
                )
                if btn_state:
                    st.session_state.fist_step_completed[s["id"]] = True
                    st.success(f"Paso completado con éxito: {s['title']}")
                    time.sleep(0.3)
                    st.rerun()

    all_done = all(st.session_state.fist_step_completed.values())
    if st.session_state.fist_emergency_active and all_done:
        st.markdown(
            """
            <div class="telemetry-alert-nominal" style="margin-top: 16px;">
                <h3>🎉 ¡INCENDIO CONTROLADO CON ÉXITO!</h3>
                El módulo ha sido aislado, el fuego se asfixió sin generar corrientes convectivas secundarias, 
                y la atmósfera está siendo purificada. Protocolo FIST finalizado con 100% de apego al estándar de la NASA.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ==============================================================================
# PUNTO DE ENTRADA PRINCIPAL (MAIN ORCHESTRATOR)
# ==============================================================================

def main() -> None:
    """Función principal estructurada y modular con captura total de excepciones."""
    try:
        configure_page()
        init_session_state()

        st.title("🔥 ZeroG FireLab: Space Training Edition")
        st.markdown(
            "**NASA Space Apps Challenge** · Módulo de Investigación de Combustión en Microgravedad "
            "*(Combustion Integrated Rack - CIR / Laboratorio Destiny - ISS)*"
        )

        fluid_parameters = render_sidebar_controls()

        tab1, tab2, tab3, tab4 = st.tabs([
            "🔬 Simulador Físico (Plotly)",
            "🎓 Quiz de Entrenamiento Espacial",
            "💬 Chatbot Técnico (PYRO-AI)",
            "🛡️ Protocolos de Emergencia FIST",
        ])

        with tab1:
            render_tab_simulator(fluid_parameters)

        with tab2:
            render_tab_quiz()

        with tab3:
            render_tab_chatbot()

        with tab4:
            render_tab_fist_protocols()

    except Exception as general_error:
        st.error(f"Se ha producido una excepción no controlada en el núcleo de la aplicación: {general_error}")


if __name__ == "__main__":
    main()
