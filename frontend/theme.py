"""
PaperCheckAI — shared design system
===================================
One place for the premium look used by every portal page.

* inject()          – global CSS (midnight navy / cyan / violet, glass panels, animations)
* landing_hero()    – animated opening panel (SVG neural network + logo reveal)
* page_header()     – consistent page title block
* kpi_card()        – stat card; the number is passed in by the caller from real API data
* empty_state()     – honest empty / unavailable state
* status_pill()     – coloured status badge

Animations use transform / opacity only (GPU friendly), are short, never block
login or data loading, and are fully disabled when the user prefers reduced
motion (OS setting) or turns on "Reduce motion" in the UI.
"""

from __future__ import annotations

import html
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
_CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --bg0:#050b1a; --bg1:#0a1630; --bg2:#0e1d3f;
  --glass:rgba(16,30,66,.55); --glass2:rgba(22,40,86,.65);
  --line:rgba(120,170,255,.16); --line2:rgba(120,170,255,.30);
  --blue:#3b82f6; --cyan:#22d3ee; --violet:#8b5cf6;
  --text:#e8efff; --muted:#93a4c8;
  --ok:#34d399; --warn:#fbbf24; --bad:#fb7185;
  --radius:16px;
  /* legacy aliases used by older pages */
  --bg:var(--bg0); --surface:var(--glass); --surface2:var(--glass2); --border:var(--line);
  --accent:var(--blue); --accent2:var(--cyan); --success:var(--ok); --danger:var(--bad); --purple:var(--violet);
}

html, body, [class*="css"]{ font-family:'Sora',system-ui,sans-serif !important; color:var(--text) !important; }
.stApp{
  background:
    radial-gradient(900px 500px at 8% -10%, rgba(59,130,246,.22), transparent 60%),
    radial-gradient(800px 500px at 100% 0%, rgba(139,92,246,.18), transparent 55%),
    radial-gradient(700px 500px at 50% 110%, rgba(34,211,238,.10), transparent 60%),
    linear-gradient(180deg,var(--bg0),var(--bg1)) !important;
}
header[data-testid="stHeader"]{ background:transparent !important; }
.main .block-container{ padding:1.6rem 2.2rem 4rem !important; max-width:1320px !important; animation:pc-rise .5s cubic-bezier(.2,.7,.2,1) both; }

/* ── Sidebar ─────────────────────────────────────────────── */
section[data-testid="stSidebar"]{
  background:linear-gradient(180deg,rgba(8,16,38,.96),rgba(10,22,52,.96)) !important;
  border-right:1px solid var(--line) !important; backdrop-filter:blur(14px);
}
section[data-testid="stSidebar"] > div{ padding:0 !important; }
section[data-testid="stSidebar"] *{ color:var(--text); }
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"]{ gap:.15rem !important; padding:0 .8rem; }
section[data-testid="stSidebar"] .stRadio label{
  position:relative; width:100%; padding:.58rem .8rem .58rem 1rem !important; border-radius:12px;
  font-size:.84rem !important; color:var(--muted) !important; cursor:pointer;
  transition:background .18s ease, color .18s ease, transform .18s ease;
}
section[data-testid="stSidebar"] .stRadio label > div:first-child{ display:none !important; }   /* hide radio dot */
section[data-testid="stSidebar"] .stRadio label:hover{ background:rgba(59,130,246,.10); color:var(--text) !important; transform:translateX(2px); }
section[data-testid="stSidebar"] .stRadio label:has(input:checked){
  background:linear-gradient(90deg,rgba(59,130,246,.22),rgba(139,92,246,.10));
  color:#fff !important; box-shadow:inset 0 0 0 1px var(--line2);
}
section[data-testid="stSidebar"] .stRadio label:has(input:checked)::before{
  content:""; position:absolute; left:0; top:22%; bottom:22%; width:3px; border-radius:3px;
  background:linear-gradient(var(--cyan),var(--violet)); animation:pc-bar .28s ease-out both;
}

.logo-block{ padding:1.4rem 1.25rem 1rem; border-bottom:1px solid var(--line); margin-bottom:.9rem; display:flex; gap:.7rem; align-items:center; }
.logo-mark{ width:38px; height:38px; flex:0 0 38px; }
.logo-wordmark{ font-size:1.12rem; font-weight:700; letter-spacing:-.3px; }
.logo-wordmark span{ background:linear-gradient(90deg,var(--cyan),var(--violet)); -webkit-background-clip:text; background-clip:text; color:transparent !important; }
.logo-inst{ font-size:.6rem; font-weight:500; color:var(--muted) !important; letter-spacing:1.4px; text-transform:uppercase; margin-top:2px; }

.role-badge{ display:inline-flex; align-items:center; padding:.22rem .6rem; border-radius:999px; font-size:.66rem; font-weight:600; text-transform:uppercase; letter-spacing:.9px; margin-top:.4rem; }
.role-admin  { background:rgba(251,113,133,.12); color:#fda4af !important; border:1px solid rgba(251,113,133,.35); }
.role-faculty{ background:rgba(59,130,246,.14);  color:#93c5fd !important; border:1px solid rgba(59,130,246,.40); }
.role-student{ background:rgba(52,211,153,.12);  color:#6ee7b7 !important; border:1px solid rgba(52,211,153,.35); }

.status-row{ padding:0 1.25rem; margin-bottom:.7rem; }
.status-dot{ display:inline-flex; align-items:center; gap:.45rem; font-size:.72rem; color:var(--muted); }
.dot-on,.dot-off{ width:8px; height:8px; border-radius:50%; }
.dot-on { background:var(--ok);  box-shadow:0 0 8px var(--ok);  animation:pc-pulse 2.4s ease-in-out infinite; }
.dot-off{ background:var(--bad); box-shadow:0 0 8px var(--bad); }

/* ── Page header / cards ─────────────────────────────────── */
.pg-title{ font-size:1.65rem; font-weight:700; letter-spacing:-.5px; margin-bottom:.15rem;
  background:linear-gradient(90deg,#fff,#b9d2ff 60%,#c4b5fd); -webkit-background-clip:text; background-clip:text; color:transparent; }
.pg-sub{ font-size:.83rem; color:var(--muted); margin-bottom:1.4rem; max-width:860px; }
.crumb{ font-size:.68rem; letter-spacing:1.2px; text-transform:uppercase; color:var(--cyan); margin-bottom:.35rem; font-weight:600; }

.card{
  background:var(--glass); border:1px solid var(--line); border-radius:var(--radius);
  padding:1.2rem 1.4rem 1.4rem; margin-bottom:1.1rem; backdrop-filter:blur(12px);
  box-shadow:0 10px 30px rgba(2,6,23,.35), inset 0 1px 0 rgba(255,255,255,.04);
  transition:transform .2s ease, border-color .2s ease, box-shadow .2s ease; animation:pc-rise .5s both;
}
.card:hover{ border-color:var(--line2); box-shadow:0 14px 38px rgba(2,6,23,.5), 0 0 0 1px rgba(34,211,238,.06); }
.card-label{ font-size:.66rem; font-weight:600; letter-spacing:1.3px; text-transform:uppercase; color:var(--muted); margin-bottom:.8rem; }

/* KPI cards */
.kpi{ position:relative; overflow:hidden; background:var(--glass); border:1px solid var(--line); border-radius:var(--radius);
  padding:1rem 1.15rem; backdrop-filter:blur(12px); transition:transform .2s ease, border-color .2s ease; animation:pc-rise .55s both; }
.kpi:hover{ transform:translateY(-3px); border-color:var(--line2); }
.kpi::after{ content:""; position:absolute; inset:auto -20% -60% auto; width:140px; height:140px; border-radius:50%;
  background:radial-gradient(circle,var(--kc,rgba(59,130,246,.35)),transparent 70%); opacity:.55; pointer-events:none; }
.kpi-label{ font-size:.64rem; letter-spacing:1.2px; text-transform:uppercase; color:var(--muted); font-weight:600; }
.kpi-num{ font-size:2rem; font-weight:700; letter-spacing:-1px; line-height:1.15; margin-top:.25rem; font-variant-numeric:tabular-nums; }
.kpi-hint{ font-size:.7rem; color:var(--muted); margin-top:.2rem; }
.kpi.cyan  {--kc:rgba(34,211,238,.35)} .kpi.violet{--kc:rgba(139,92,246,.38)}
.kpi.green {--kc:rgba(52,211,153,.30)} .kpi.amber {--kc:rgba(251,191,36,.28)} .kpi.rose{--kc:rgba(251,113,133,.30)}

/* Count-up for real values (Chromium/Edge); other browsers simply show the number. */
@property --n{ syntax:'<integer>'; initial-value:0; inherits:false; }
@supports (animation-timeline:auto){ .kpi-num.count{ counter-reset:cn var(--n); animation:pc-count 1.1s ease-out both; }
  .kpi-num.count > span{ display:none; } .kpi-num.count::before{ content:counter(cn); } }
@keyframes pc-count{ from{ --n:0; } to{ --n:var(--to); } }

.mtile{ background:var(--glass2); border:1px solid var(--line); border-radius:12px; padding:.85rem 1rem; text-align:center; }
.mtile-val{ font-size:1.6rem; font-weight:700; line-height:1.1; }
.mtile-label{ font-size:.64rem; color:var(--muted); margin-top:.3rem; font-weight:600; text-transform:uppercase; letter-spacing:.8px; }

.score-hero{ background:linear-gradient(135deg,rgba(30,58,138,.55),rgba(91,33,182,.35)); border:1px solid var(--line2); border-radius:16px; padding:1.75rem; text-align:center; margin:1rem 0; }
.score-num{ font-size:3.4rem; font-weight:700; color:#fff; line-height:1; letter-spacing:-2px; }
.score-denom{ font-size:1.4rem; opacity:.5; font-weight:300; }

.pill{ display:inline-flex; align-items:center; gap:.25rem; padding:.2rem .6rem; border-radius:999px; font-size:.72rem; font-weight:500; transition:transform .15s ease; }
.pill:hover{ transform:translateY(-1px); }
.pill-g{ background:rgba(52,211,153,.12); color:#6ee7b7; border:1px solid rgba(52,211,153,.4); }
.pill-r{ background:rgba(251,113,133,.12); color:#fda4af; border:1px solid rgba(251,113,133,.4); }
.pill-b{ background:rgba(59,130,246,.14); color:#93c5fd; border:1px solid rgba(59,130,246,.4); }
.pill-v{ background:rgba(139,92,246,.14); color:#c4b5fd; border:1px solid rgba(139,92,246,.4); }
.pill-a{ background:rgba(251,191,36,.12); color:#fde68a; border:1px solid rgba(251,191,36,.4); }

.chat-bubble{ padding:.85rem 1.1rem; border-radius:14px; margin-bottom:.75rem; font-size:.88rem; line-height:1.6; animation:pc-rise .3s both; }
.chat-user{ background:rgba(59,130,246,.16); border:1px solid rgba(59,130,246,.35); margin-left:2rem; }
.chat-asst{ background:var(--glass2); border:1px solid var(--line); margin-right:2rem; }
.voice-banner{ background:linear-gradient(135deg,rgba(59,130,246,.14),rgba(139,92,246,.14)); border:1px solid var(--line2); border-radius:14px; padding:.85rem 1.25rem; margin-bottom:1rem; display:flex; justify-content:space-between; align-items:center; }
.voice-controls-bar{ background:rgba(8,16,38,.7); border:1px solid var(--line); border-radius:12px; padding:.75rem 1rem; margin-bottom:1.2rem; }
.voice-status-pill{ display:inline-flex; align-items:center; gap:.4rem; padding:.25rem .65rem; border-radius:999px; font-size:.75rem; font-weight:600; }
.voice-active{ background:rgba(52,211,153,.14); color:#6ee7b7; border:1px solid rgba(52,211,153,.4); }
.voice-idle  { background:rgba(147,164,200,.12); color:var(--muted); border:1px solid var(--line); }

/* Empty state */
.empty{ text-align:center; padding:2rem 1rem; border:1px dashed var(--line2); border-radius:14px; background:rgba(14,29,63,.35); animation:pc-rise .4s both; }
.empty .e-ic{ font-size:1.8rem; margin-bottom:.3rem; } .empty .e-t{ font-weight:600; } .empty .e-s{ color:var(--muted); font-size:.8rem; margin-top:.2rem; }

/* ── Streamlit widgets ───────────────────────────────────── */
.stButton > button, .stDownloadButton > button, div[data-testid="stFormSubmitButton"] > button{
  border-radius:12px !important; border:1px solid var(--line2) !important; color:#fff !important; font-weight:600 !important;
  background:linear-gradient(135deg,rgba(59,130,246,.85),rgba(139,92,246,.80)) !important;
  transition:transform .15s ease, box-shadow .2s ease, filter .2s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover{
  transform:translateY(-1px); box-shadow:0 8px 24px rgba(59,130,246,.35); filter:brightness(1.08); }
.stButton > button:active{ transform:translateY(0) scale(.98); }
.stButton > button:focus-visible, input:focus-visible, textarea:focus-visible, [role="tab"]:focus-visible{ outline:2px solid var(--cyan) !important; outline-offset:2px; }
.stTextInput input, .stTextArea textarea, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div{
  background:rgba(8,16,38,.7) !important; border:1px solid var(--line) !important; border-radius:12px !important; color:var(--text) !important; }
.stTextInput input:focus, .stTextArea textarea:focus{ border-color:var(--cyan) !important; box-shadow:0 0 0 3px rgba(34,211,238,.15) !important; }
button[role="tab"]{ color:var(--muted) !important; transition:color .2s ease; } button[role="tab"][aria-selected="true"]{ color:#fff !important; }
div[data-baseweb="tab-highlight"]{ background:linear-gradient(90deg,var(--cyan),var(--violet)) !important; height:3px !important; border-radius:3px; }
div[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:14px; overflow:hidden; }
div[data-testid="stForm"]{ background:var(--glass); border:1px solid var(--line); border-radius:var(--radius); backdrop-filter:blur(12px); }
div[data-testid="stAlert"]{ border-radius:12px !important; animation:pc-rise .3s both; }
div[data-testid="stExpander"]{ border:1px solid var(--line) !important; border-radius:12px !important; background:rgba(14,29,63,.35); }

/* ── Landing / login ─────────────────────────────────────── */
.hero{ position:relative; overflow:hidden; border:1px solid var(--line); border-radius:24px; min-height:520px; padding:2.4rem 2.4rem 2rem;
  background:linear-gradient(145deg,rgba(14,29,63,.85),rgba(7,14,34,.9)); box-shadow:0 30px 80px rgba(2,6,23,.6), inset 0 1px 0 rgba(255,255,255,.05); }
.hero-net{ position:absolute; inset:0; width:100%; height:100%; opacity:.85; pointer-events:none; }
.hero-net .ln{ stroke:rgba(120,170,255,.22); stroke-width:1; stroke-dasharray:4 6; animation:pc-dash 14s linear infinite; }
.hero-net .nd{ fill:var(--cyan); transform-box:fill-box; transform-origin:center; animation:pc-float 5s ease-in-out infinite; }
.hero-net .nd.v{ fill:var(--violet); } .hero-net .nd.b{ fill:var(--blue); }
.hero-in{ position:relative; z-index:2; }
.brand{ display:flex; align-items:center; gap:.8rem; animation:pc-logo .9s cubic-bezier(.2,.8,.2,1) both; }
.brand svg{ width:54px; height:54px; filter:drop-shadow(0 0 14px rgba(34,211,238,.45)); }
.brand-t{ font-size:1.7rem; font-weight:700; letter-spacing:-.8px; }
.brand-t span{ background:linear-gradient(90deg,var(--cyan),var(--violet)); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero h1{ font-size:2.35rem; line-height:1.12; letter-spacing:-1.2px; margin:1.6rem 0 .8rem; font-weight:700; animation:pc-rise .7s .15s both; }
.hero h1 em{ font-style:normal; background:linear-gradient(90deg,var(--cyan),var(--blue),var(--violet)); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p.lead{ color:var(--muted); font-size:.95rem; max-width:520px; line-height:1.65; animation:pc-rise .7s .28s both; }
.feat{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.7rem; margin-top:1.6rem; max-width:560px; }
.feat div{ background:rgba(16,30,66,.55); border:1px solid var(--line); border-radius:14px; padding:.7rem .9rem; font-size:.78rem; backdrop-filter:blur(8px); animation:pc-rise .6s both; }
.feat div:nth-child(1){animation-delay:.40s} .feat div:nth-child(2){animation-delay:.48s} .feat div:nth-child(3){animation-delay:.56s} .feat div:nth-child(4){animation-delay:.64s}
.feat b{ display:block; font-size:.82rem; margin-bottom:.15rem; } .feat span{ color:var(--muted); }
.login-title{ font-size:1.25rem; font-weight:700; margin-bottom:.1rem; } .login-sub{ font-size:.8rem; color:var(--muted); margin-bottom:.9rem; }
.legal{ font-size:.68rem; color:var(--muted); margin-top:.8rem; line-height:1.5; }

/* Animation assistant orb (state driven by real events, see assistant_orb()) */
.orb{ width:54px; height:54px; border-radius:50%; position:relative; display:inline-block;
  background:radial-gradient(circle at 35% 30%,#fff,var(--cyan) 30%,var(--blue) 60%,var(--violet)); box-shadow:0 0 22px rgba(34,211,238,.5); }
.orb.idle{ animation:pc-breathe 4s ease-in-out infinite; }
.orb.processing{ animation:pc-breathe 1.1s ease-in-out infinite; box-shadow:0 0 30px rgba(139,92,246,.7); }
.orb.success{ box-shadow:0 0 26px rgba(52,211,153,.7); filter:hue-rotate(-60deg); }
.orb.error{ box-shadow:0 0 26px rgba(251,113,133,.7); filter:hue-rotate(140deg) saturate(1.3); }

/* ── Keyframes ───────────────────────────────────────────── */
@keyframes pc-rise  { from{opacity:0; transform:translateY(14px)} to{opacity:1; transform:none} }
@keyframes pc-logo  { from{opacity:0; transform:scale(.7) rotate(-8deg); filter:blur(6px)} to{opacity:1; transform:none; filter:none} }
@keyframes pc-bar   { from{transform:scaleY(0)} to{transform:scaleY(1)} }
@keyframes pc-pulse { 0%,100%{opacity:1} 50%{opacity:.45} }
@keyframes pc-float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }
@keyframes pc-dash  { to{stroke-dashoffset:-200} }
@keyframes pc-breathe{ 0%,100%{transform:scale(1)} 50%{transform:scale(1.08)} }

@media (max-width:900px){ .main .block-container{ padding:1rem !important; } .hero{ padding:1.4rem; min-height:auto; } .hero h1{ font-size:1.7rem; } .feat{ grid-template-columns:1fr; } }

/* Reduced motion: OS preference */
@media (prefers-reduced-motion:reduce){ *,*::before,*::after{ animation:none !important; transition:none !important; } }
"""

_REDUCED = "*,*::before,*::after{ animation:none !important; transition:none !important; }"

_LOGO_SVG = """
<svg class="%s" viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="PaperCheckAI logo">
  <defs><linearGradient id="pcg%s" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="#22d3ee"/><stop offset="1" stop-color="#8b5cf6"/></linearGradient></defs>
  <rect x="8" y="6" width="38" height="50" rx="8" fill="rgba(14,29,63,.9)" stroke="url(#pcg%s)" stroke-width="2.5"/>
  <path d="M16 20h22M16 28h22M16 36h13" stroke="url(#pcg%s)" stroke-width="2.5" stroke-linecap="round" opacity=".75"/>
  <circle cx="46" cy="44" r="13" fill="url(#pcg%s)"/>
  <path d="M40 44.5l4.2 4.2L52.5 40" stroke="#04122b" stroke-width="3.4" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


def logo_svg(cls: str = "logo-mark", uid: str = "a") -> str:
    return _LOGO_SVG % (cls, uid, uid, uid, uid)


# A fixed decorative node layout (purely visual — carries no data).
_NODES = [(70, 80, "nd"), (190, 150, "nd b"), (310, 70, "nd v"), (430, 170, "nd"), (560, 90, "nd b"),
          (640, 210, "nd v"), (120, 290, "nd v"), (260, 330, "nd"), (400, 300, "nd b"), (540, 360, "nd"),
          (650, 420, "nd v"), (90, 450, "nd b"), (220, 480, "nd"), (370, 450, "nd v")]
_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 6), (1, 7), (3, 8), (5, 9), (6, 7), (7, 8), (8, 9),
          (9, 10), (6, 11), (7, 12), (8, 13), (11, 12), (12, 13), (2, 4), (3, 5)]


def _network_svg() -> str:
    lines = "".join(
        f'<line class="ln" x1="{_NODES[a][0]}" y1="{_NODES[a][1]}" x2="{_NODES[b][0]}" y2="{_NODES[b][1]}"/>'
        for a, b in _EDGES)
    dots = "".join(
        f'<circle class="{c}" cx="{x}" cy="{y}" r="4.5" style="animation-delay:{(i % 5) * .6:.1f}s"/>'
        for i, (x, y, c) in enumerate(_NODES))
    return (f'<svg class="hero-net" viewBox="0 0 720 540" preserveAspectRatio="xMidYMid slice" '
            f'aria-hidden="true">{lines}{dots}</svg>')


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers
# ─────────────────────────────────────────────────────────────────────────────
_READABLE = """
textarea:disabled, input:disabled { color:#e5e9ff !important; -webkit-text-fill-color:#e5e9ff !important;
  opacity:1 !important; background:#0f1838 !important; }
"""


def inject(reduce_motion: bool = False) -> None:
    """Inject the global design system. Call once per script run, before any page content."""
    extra = (_REDUCED if reduce_motion else "") + _READABLE
    st.markdown(f"<style>{_CSS}\n{extra}</style>", unsafe_allow_html=True)


def landing_hero() -> None:
    """Animated opening panel shown beside the real login form."""
    st.markdown(
        f"""
<div class="hero">
  {_network_svg()}
  <div class="hero-in">
    <div class="brand">{logo_svg('', 'h')}<div class="brand-t">PaperCheck<span>AI</span></div></div>
    <h1>AI-assisted examination checking,<br/><em>explainable</em> and under examiner control.</h1>
    <p class="lead">Upload answer sheets, review question-wise AI evaluation with evidence for every
    deduction, correct marks as an examiner, and publish verified results — all in one secure portal.</p>
    <div class="feat">
      <div><b>Explainable marks</b><span>Every deduction shows the rubric reason.</span></div>
      <div><b>Examiner in control</b><span>AI scores stay provisional until approved.</span></div>
      <div><b>Role-based access</b><span>Admin, Faculty and Student permissions.</span></div>
      <div><b>Audit trail</b><span>Marks changes and publications are logged.</span></div>
    </div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def sidebar_brand() -> None:
    st.markdown(
        f'<div class="logo-block">{logo_svg("logo-mark", "s")}<div>'
        '<div class="logo-wordmark">PaperCheck<span>AI</span></div>'
        '<div class="logo-inst">University Examination System</div></div></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str = "", crumb: str = "") -> None:
    c = f'<div class="crumb">{html.escape(crumb)}</div>' if crumb else ""
    st.markdown(
        f'{c}<div class="pg-title">{html.escape(title)}</div><div class="pg-sub">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value, hint: str = "", tone: str = "", delay: float = 0.0) -> str:
    """Return KPI card HTML. `value` must come from real API data; None renders as an em dash.

    Integers get a CSS count-up (Chromium/Edge); the true value is always in the markup.
    """
    if value is None:
        num = '<div class="kpi-num">—</div>'
    elif isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        num = f'<div class="kpi-num count" style="--to:{value}"><span>{value}</span></div>'
    else:
        num = f'<div class="kpi-num">{html.escape(str(value))}</div>'
    return (f'<div class="kpi {tone}" style="animation-delay:{delay:.2f}s">'
            f'<div class="kpi-label">{html.escape(label)}</div>{num}'
            f'<div class="kpi-hint">{html.escape(hint)}</div></div>')


def kpi_row(items: list[dict]) -> None:
    """items: [{label, value, hint, tone}] — rendered as one responsive row."""
    cols = st.columns(len(items))
    for i, (col, it) in enumerate(zip(cols, items)):
        col.markdown(kpi_card(it["label"], it.get("value"), it.get("hint", ""), it.get("tone", ""), i * 0.07),
                     unsafe_allow_html=True)


def empty_state(title: str, subtitle: str = "", icon: str = "🗂️") -> None:
    st.markdown(
        f'<div class="empty"><div class="e-ic">{icon}</div><div class="e-t">{html.escape(title)}</div>'
        f'<div class="e-s">{html.escape(subtitle)}</div></div>',
        unsafe_allow_html=True,
    )


_PILL = {"ok": "pill-g", "bad": "pill-r", "info": "pill-b", "ai": "pill-v", "warn": "pill-a"}


def status_pill(text: str, kind: str = "info") -> str:
    return f'<span class="pill {_PILL.get(kind, "pill-b")}">{html.escape(text)}</span>'


def assistant_orb(state: str = "idle") -> str:
    """Return the AI assistant orb. `state` must reflect a real event: idle | processing | success | error."""
    state = state if state in ("idle", "processing", "success", "error") else "idle"
    return f'<span class="orb {state}" role="img" aria-label="AI assistant: {state}"></span>'
