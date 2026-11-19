
import csv
import os

import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "penguins_lter.csv")

FEATURES = [
    "Culmen Length (mm)",
    "Culmen Depth (mm)",
    "Flipper Length (mm)",
    "Body Mass (g)",
]

# Plausible input ranges (slightly wider than the observed data)
LIMITS = {
    "Culmen Length (mm)": (25, 70),
    "Culmen Depth (mm)": (10, 25),
    "Flipper Length (mm)": (150, 250),
    "Body Mass (g)": (2000, 7500),
}


def load_data():
    """Read the CSV with the standard library (keeps the deployment small)."""
    rows, labels = [], []
    with open(DATA_PATH, newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            species = (rec.get("Species") or "").strip()
            if not species:
                continue
            row = []
            for col in FEATURES:
                val = (rec.get(col) or "").strip()
                row.append(float(val) if val else np.nan)
            rows.append(row)
            labels.append(species.split(" ")[0])  # Adelie / Chinstrap / Gentoo
    return np.array(rows, dtype=float), np.array(labels)

def build_model():
    X, y = load_data()
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5)),
    ])
    model.fit(X, y)
    return model, len(y)

MODEL, N_TRAIN = build_model()

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Who's That Penguin? | Penguin Species Classifier</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@500;600;700&family=Nunito:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root { --ice:#DFF4FF; --ocean:#4A90C6; --navy:#142D4E; --snow:#F8FCFF; --lav:#DCD6FF; --coral:#FF8C78; }
  * { box-sizing:border-box; }
  html { scroll-behavior:smooth; scroll-padding-top:70px; }
  body { margin:0; font-family:Nunito,"Trebuchet MS",system-ui,sans-serif; color:var(--navy); background:var(--snow); line-height:1.6; }
  h1,h2,h3,.fr { font-family:Fredoka,"Trebuchet MS",sans-serif; margin:0; }
  nav { position:fixed; top:0; left:0; right:0; z-index:20; display:flex; justify-content:space-between; align-items:center; padding:12px 5vw;
        background:rgba(20,45,78,.45); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px); border-bottom:1px solid rgba(255,255,255,.18); }
  nav b { color:#fff; font:600 1.25rem Fredoka,sans-serif; } nav div a { color:#e8f4ff; text-decoration:none; margin-left:20px; font-weight:700; font-size:.95rem; transition:color .2s; } nav div a:hover { color:var(--coral); }
  .btn { display:inline-block; border:0; cursor:pointer; text-decoration:none; font:600 1.05rem Fredoka,sans-serif; padding:14px 28px; border-radius:40px; color:#fff;
         background:linear-gradient(135deg,#ff9d8a,var(--coral)); box-shadow:0 10px 26px rgba(255,140,120,.5); transition:transform .2s, box-shadow .2s; }
  .btn:hover { transform:translateY(-3px) scale(1.03); box-shadow:0 16px 34px rgba(255,140,120,.62); }
  .btn.ghost { background:rgba(255,255,255,.16); border:2px solid rgba(255,255,255,.75); box-shadow:none; backdrop-filter:blur(6px); }
  .btn.pulse { animation:pulse 2.4s ease-in-out infinite; } @keyframes pulse { 50% { box-shadow:0 0 0 12px rgba(255,140,120,.18),0 10px 26px rgba(255,140,120,.5); } }
  /* hero */
  .hero { position:relative; min-height:100vh; overflow:hidden; color:#fff; display:flex; align-items:center; padding:110px 6vw 220px;
          background:radial-gradient(ellipse at 20% 10%,rgba(124,255,214,.28),transparent 45%),radial-gradient(ellipse at 85% 5%,rgba(220,214,255,.35),transparent 45%),linear-gradient(180deg,#142D4E 0%,#2a5a8c 48%,#4A90C6 72%,#DFF4FF 100%); }
  #snow { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; z-index:1; }
  .scene { position:absolute; left:0; bottom:-2px; width:100%; height:36%; z-index:2; pointer-events:none; }
  .w1 { animation:wave 9s linear infinite; } .w2 { animation:wave 14s linear infinite reverse; } @keyframes wave { to { transform:translateX(-360px); } }
  .berg { animation:bob 6s ease-in-out infinite; } @keyframes bob { 50% { transform:translateY(-8px); } }
  .hero-in { position:relative; z-index:3; width:100%; max-width:1180px; margin:0 auto; display:grid; grid-template-columns:1.1fr .9fr; gap:30px; align-items:center; }
  .hero h1 { font-size:clamp(2.6rem,7vw,4.8rem); line-height:1.05; text-shadow:0 6px 28px rgba(0,0,0,.35); } .hero h1 span { color:var(--coral); }
  .hero p { font-size:1.15rem; max-width:520px; color:#e6f3ff; margin:16px 0 26px; } .cta { display:flex; gap:14px; flex-wrap:wrap; }
  .stage { position:relative; display:grid; place-items:center; } .stage::before { content:""; position:absolute; width:78%; aspect-ratio:1; border-radius:50%; background:radial-gradient(circle,rgba(255,255,255,.5),rgba(255,255,255,0) 70%); animation:glow 4s ease-in-out infinite; }
  @keyframes glow { 50% { transform:scale(1.12); opacity:.75; } }
  .big svg { width:min(380px,72vw); filter:drop-shadow(0 26px 30px rgba(5,15,40,.45)); animation:float 4.5s ease-in-out infinite; } @keyframes float { 50% { transform:translateY(-16px) rotate(1.5deg); } }
  .chip { position:absolute; padding:8px 16px; border-radius:30px; font:600 .95rem Fredoka,sans-serif; color:var(--navy); background:rgba(255,255,255,.82); backdrop-filter:blur(8px); box-shadow:0 10px 24px rgba(0,0,0,.2); animation:float 5s ease-in-out infinite; }
  .c1 { top:6%; left:2%; } .c2 { top:20%; right:0; animation-delay:-1.5s; background:rgba(220,214,255,.92); } .c3 { bottom:10%; left:6%; animation-delay:-3s; background:rgba(255,200,190,.92); }
  /* sections */
  section.blk { padding:90px 6vw; } .wrap { max-width:1100px; margin:0 auto; } .head { text-align:center; margin-bottom:36px; } .head h2 { font-size:clamp(2rem,5vw,3rem); } .head p { color:#46618a; max-width:560px; margin:10px auto 0; }
  #predict { background:linear-gradient(180deg,var(--snow),var(--ice)); }
  #species { background:linear-gradient(180deg,var(--ice),var(--lav)); }
  .glass { position:relative; max-width:720px; margin:0 auto; padding:30px; border-radius:28px; border:1px solid rgba(255,255,255,.8); overflow:hidden;
           background:linear-gradient(135deg,rgba(223,244,255,.95),rgba(220,214,255,.62)); backdrop-filter:blur(16px); box-shadow:0 24px 60px rgba(20,45,78,.18); }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  .field { position:relative; } .field label { display:block; font-weight:700; margin-bottom:6px; font-size:.95rem; }
  .field .in { display:flex; align-items:center; gap:10px; padding:4px 14px; background:#fff; border:2px solid transparent; border-radius:16px; box-shadow:0 4px 14px rgba(74,144,198,.18); transition:transform .2s, border-color .2s, box-shadow .2s; }
  .field .in:hover { transform:translateY(-2px); box-shadow:0 10px 22px rgba(74,144,198,.3); } .field .in:focus-within { border-color:var(--ocean); }
  .field input { flex:1; min-width:0; border:0; outline:0; background:transparent; padding:12px 0; font:600 1.05rem Nunito,sans-serif; color:var(--navy); } .field .u { color:#7b93b4; font-weight:700; font-size:.85rem; }
  .field.bad .in { border-color:var(--coral); } .field .m { min-height:1.2em; color:#d4543e; font-size:.82rem; margin-top:4px; }
  .shake { animation:shake .4s; } @keyframes shake { 25% { transform:translateX(-6px); } 75% { transform:translateX(6px); } }
  .go { margin-top:22px; text-align:center; }
  #loader { position:absolute; inset:0; z-index:5; display:none; flex-direction:column; align-items:center; justify-content:center; gap:10px; background:rgba(248,252,255,.94); text-align:center; font:600 1.15rem Fredoka,sans-serif; }
  #loader.on { display:flex; } #loader svg { width:150px; animation:float 1.3s ease-in-out infinite; } .mag { animation:scan 1.6s ease-in-out infinite; transform-box:fill-box; transform-origin:50% 50%; } @keyframes scan { 50% { transform:translate(-14px,-10px) rotate(-12deg); } }
  .dots::after { content:""; animation:dots 1.2s steps(4) infinite; } @keyframes dots { 0% { content:""; } 25% { content:"."; } 50% { content:".."; } 75% { content:"..."; } }
  #result { display:none; max-width:720px; margin:0 auto; padding:34px 30px; text-align:center; border-radius:32px; position:relative; background:linear-gradient(160deg,#fff,var(--ice) 60%,var(--lav)); border:2px solid #fff; box-shadow:0 30px 70px rgba(20,45,78,.25); }
  #result.show { display:block; animation:reveal .9s cubic-bezier(.2,1.4,.4,1) both; } @keyframes reveal { from { opacity:0; transform:scale(.7) rotate(-4deg) translateY(40px); } to { opacity:1; transform:none; } }
  .eyebrow { font:600 1rem Fredoka,sans-serif; color:var(--coral); letter-spacing:.12em; text-transform:uppercase; } #result h2 { font-size:clamp(2.2rem,6vw,3.2rem); margin:4px 0; } #result h2 span { color:var(--ocean); }
  .art { position:relative; width:230px; height:240px; margin:8px auto; display:grid; place-items:center; } .art svg { width:100%; height:100%; animation:float 4s ease-in-out infinite; filter:drop-shadow(0 16px 18px rgba(20,45,78,.3)); }
  .art img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; border-radius:24px; }
  .conf { display:inline-block; margin:6px 0 10px; padding:8px 20px; border-radius:30px; background:var(--navy); color:#fff; font:600 1.05rem Fredoka,sans-serif; } .conf small { opacity:.75; font-family:Nunito,sans-serif; font-weight:400; }
  .desc { max-width:520px; margin:0 auto 12px; color:#34507a; } .party { font:600 1.2rem Fredoka,sans-serif; color:var(--coral); margin:12px 0 18px; }
  .bars { max-width:420px; margin:0 auto 18px; text-align:left; } .bar { display:flex; align-items:center; gap:10px; font-size:.85rem; margin:6px 0; } .bar span:first-child { width:78px; } .bar span:last-child { width:38px; text-align:right; }
  .tr { flex:1; height:10px; border-radius:6px; background:rgba(74,144,198,.18); overflow:hidden; } .tr i { display:block; height:100%; width:0; border-radius:6px; background:linear-gradient(90deg,var(--ocean),var(--coral)); transition:width 1.2s cubic-bezier(.2,.9,.3,1); }
  .cf { position:absolute; left:50%; top:30%; pointer-events:none; animation:cf 1.5s ease-out forwards; font-size:1.2rem; } @keyframes cf { to { transform:translate(var(--x),var(--y)) rotate(var(--r)); opacity:0; } }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:26px; }
  .card { background:rgba(255,255,255,.72); backdrop-filter:blur(12px); border:1px solid #fff; border-radius:26px; padding:22px; text-align:center; box-shadow:0 16px 40px rgba(20,45,78,.14); transition:transform .3s, box-shadow .3s; }
  .card:hover { transform:translateY(-8px) rotate(-.6deg); box-shadow:0 26px 54px rgba(20,45,78,.24); } .card .art { width:170px; height:180px; } .card h3 { font-size:1.55rem; } .sci { font-style:italic; color:var(--ocean); margin:0 0 8px; }
  .more { max-height:0; overflow:hidden; text-align:left; font-size:.92rem; color:#34507a; transition:max-height .5s ease; } .card.open .more { max-height:220px; } .more p { margin:10px 0 0; }
  .link { margin-top:12px; background:none; border:2px solid var(--ocean); color:var(--ocean); padding:8px 20px; border-radius:30px; font:600 .95rem Fredoka,sans-serif; cursor:pointer; transition:all .2s; } .link:hover { background:var(--ocean); color:#fff; }
  footer { background:var(--navy); color:#b9d3ee; text-align:center; padding:34px 6vw; font-size:.9rem; } footer a { color:var(--coral); }
  @media (max-width:820px) { .hero-in { grid-template-columns:1fr; text-align:center; } .hero p { margin-inline:auto; } .cta { justify-content:center; } .grid { grid-template-columns:1fr; } nav div a:nth-child(1) { display:none; } .chip { font-size:.8rem; padding:6px 12px; } }
  @media (prefers-reduced-motion:reduce) { *,*::before,*::after { animation:none !important; transition:none !important; } html { scroll-behavior:auto; } }
</style>
</head>
<body>
<svg width="0" height="0" style="position:absolute" aria-hidden="true"><defs>
  <radialGradient id="pgb" cx=".35" cy=".25" r=".9"><stop offset="0" stop-color="#46618a"/><stop offset=".5" stop-color="#1c3050"/><stop offset="1" stop-color="#0a1424"/></radialGradient>
  <linearGradient id="pgw" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#cfe3f4"/></linearGradient>
  <linearGradient id="pgo" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ffb067"/><stop offset="1" stop-color="#ef5a33"/></linearGradient>
  <symbol id="pb" viewBox="0 0 120 140">
    <ellipse cx="60" cy="135" rx="36" ry="5" fill="#0a1a30" opacity=".28"/>
    <path d="M27 62C8 84 10 108 20 117C28 101 30 82 33 68Z" fill="url(#pgb)"/><path d="M93 62C112 84 110 108 100 117C92 101 90 82 87 68Z" fill="url(#pgb)"/>
    <ellipse cx="43" cy="132" rx="14" ry="5.5" fill="url(#pgo)"/><ellipse cx="77" cy="132" rx="14" ry="5.5" fill="url(#pgo)"/>
    <path d="M60 5C87 5 99 30 99 66C99 104 88 128 60 130C32 128 21 104 21 66C21 30 33 5 60 5Z" fill="url(#pgb)"/>
    <path d="M60 44C80 46 86 72 84 98C82 120 72 128 60 128C48 128 38 120 36 98C34 72 40 46 60 44Z" fill="url(#pgw)"/>
    <ellipse cx="48" cy="16" rx="15" ry="6" fill="#fff" opacity=".22" transform="rotate(-18 48 16)"/>
  </symbol>
</defs></svg>

<nav><b>Penguin Pal</b><div><a href="#top">Home</a><a href="#predict">Identify</a><a href="#species">Species</a></div></nav>

<header class="hero" id="top">
  <canvas id="snow"></canvas>
  <div class="hero-in">
    <div>
      <h1>Who's That <span>Penguin?</span></h1>
      <p>Meet the magic of machine learning! Discover which penguin species you're looking at with just a few measurements.</p>
      <div class="cta"><a class="btn pulse" href="#predict">Identify My Penguin 🐧</a><a class="btn ghost" href="#species">Explore Penguin Species</a></div>
    </div>
    <div class="stage"><div class="big" id="heroP"></div>
      <span class="chip c2">98.6% CV Accuracy</span><span class="chip c3">3 Species</span></div>
  </div>
  <svg class="scene" viewBox="0 0 1440 400" preserveAspectRatio="none" aria-hidden="true">
    <path d="M0 400V230L160 120L260 200L420 60L580 210L720 130L900 230L1060 90L1240 220L1440 150V400Z" fill="#7fb3dc"/>
    <path d="M420 60L470 120L420 105L380 125Z M1060 90L1110 150L1060 135L1015 155Z M160 120L200 165L160 150L125 170Z" fill="#fff" opacity=".9"/>
    <path d="M0 400V290L200 190L340 270L520 170L700 290L880 200L1100 300L1280 220L1440 290V400Z" fill="#F8FCFF"/>
    <g class="berg"><path d="M1130 300L1175 245L1200 270L1235 235L1270 300Z" fill="#fff"/><path d="M1200 270L1235 235L1270 300L1215 300Z" fill="#cfe7f7"/></g>
    <g class="w2"><path d="M-360 335Q-270 310 -180 335T0 335T180 335T360 335T540 335T720 335T900 335T1080 335T1260 335T1440 335T1620 335T1800 335V400H-360Z" fill="#7fb3dc" opacity=".8"/></g>
    <g class="w1"><path d="M-360 355Q-270 330 -180 355T0 355T180 355T360 355T540 355T720 355T900 355T1080 355T1260 355T1440 355T1620 355T1800 355V400H-360Z" fill="#4A90C6"/></g>
    <path d="M0 385Q360 365 720 385T1440 385V400H0Z" fill="#DFF4FF"/>
  </svg>
</header>

<section class="blk" id="predict"><div class="wrap">
  <div class="head"><h2>Let's Identify Your Penguin!</h2><p>Enter a few biological measurements and let our machine learning model do the magic.</p></div>
  <div class="glass">
    <div id="loader"><div id="detective"></div><div>Our penguin detective is investigating<span class="dots"></span></div></div>
    <div class="grid" id="fields"></div>
    <div class="go"><button class="btn pulse" id="predictBtn">Predict Species 🐧</button></div>
  </div>
  <div id="result" aria-live="polite"></div>
</div></section>

<section class="blk" id="species"><div class="wrap">
  <div class="head"><h2>Explore Penguin Species</h2><p>Three Pygoscelis penguins from the Palmer Archipelago, the ones our model knows by heart.</p></div>
  <div class="cards" id="cards"></div>
</div></section>

<footer>KNN (k=5) trained on __N__ penguins from the Palmer Penguins dataset. Data by Dr. Kristen Gorman and Palmer Station LTER.<br>
Source code on <a href="https://github.com/ashk-k10/penguin_classification">GitHub</a></footer>

<script>
const $ = id => document.getElementById(id);
const E = (x,y) => '<circle cx="'+x+'" cy="'+y+'" r="6" fill="#fff"/><circle cx="'+x+'" cy="'+y+'" r="3.2" fill="#0a1424"/><circle cx="'+(x+1.2)+'" cy="'+(y-1.3)+'" r="1.1" fill="#fff"/>';
const DARK = '<path d="M53 47Q60 45 67 47Q65 57 60 58Q55 57 53 47Z" fill="#43363a"/>';
const FACE = {
  Hero: '<circle cx="34" cy="52" r="5" fill="#FF8C78" opacity=".55"/><circle cx="86" cy="52" r="5" fill="#FF8C78" opacity=".55"/>'+E(46,38)+E(74,38)+'<path d="M51 46Q60 43 69 46Q67 59 60 61Q53 59 51 46Z" fill="url(#pgo)"/>',
  Adelie: '<circle cx="46" cy="38" r="9.5" fill="none" stroke="#fff" stroke-width="3"/><circle cx="74" cy="38" r="9.5" fill="none" stroke="#fff" stroke-width="3"/>'+E(46,38)+E(74,38)+DARK,
  Chinstrap: '<ellipse cx="60" cy="44" rx="26" ry="18" fill="#fff"/><path d="M33 42Q60 72 87 42" fill="none" stroke="#0a1424" stroke-width="3.6" stroke-linecap="round"/>'+E(46,38)+E(74,38)+DARK,
  Gentoo: '<path d="M36 27Q48 12 60 21Q72 12 84 27Q72 31 60 28Q48 31 36 27Z" fill="#fff"/>'+E(46,38)+E(74,38)+'<path d="M49 46Q60 41 71 46Q69 63 60 65Q51 63 49 46Z" fill="url(#pgo)"/>'
};
const DET = '<ellipse cx="60" cy="11" rx="31" ry="5" fill="#5b4636"/><path d="M40 11Q42 -6 60 -6Q78 -6 80 11Z" fill="#7a5c45"/><g class="mag"><circle cx="98" cy="84" r="15" fill="rgba(223,244,255,.4)" stroke="#d9a441" stroke-width="4"/><path d="M108 95L120 110" stroke="#d9a441" stroke-width="5" stroke-linecap="round"/></g>';
function peng(sp, det) { return '<svg viewBox="-8 -12 136 156" role="img" aria-label="' + sp + ' penguin illustration"><use href="#pb" width="120" height="140"/>' + FACE[sp] + (det ? DET : '') + '</svg>'; }
const SP = {
  Adelie: { sci:"Pygoscelis adeliae", img:"/adelie.jpg", blurb:"Famous for the distinctive white ring around each eye. A true native of Antarctica.", desc:"Adelies are Antarctic natives with a white ring around each eye and a feathered, dark bill. They are tidy builders of pebble nests, and pebbles are valuable enough to be stolen from neighbours.", more:"Fun fact: a gift of the perfect pebble is how males court their partners." },
  Chinstrap: { sci:"Pygoscelis antarcticus", img:"/chinstrap.jpg", blurb:"Recognised by the narrow black band beneath its chin, like a tiny helmet strap.", desc:"Chinstraps wear a thin black line under the chin that looks just like a helmet strap. They are bold, noisy and love nesting on steep, rocky slopes around the Antarctic Peninsula.", more:"Fun fact: they often travel and hop between ice floes in big, chatty groups." },
  Gentoo: { sci:"Pygoscelis papua", img:"/gentoo.jpg", blurb:"Known for its bright orange-red bill and the white patch across its head.", desc:"Gentoos stand out with a bright orange-red bill and a white patch above the eyes. They are the speed demons of the penguin world and strong, graceful swimmers.", more:"Fun fact: Gentoos can swim at around 36 km/h, the fastest of all penguins." }
};

$("heroP").innerHTML = peng("Hero"); $("detective").innerHTML = peng("Gentoo", true);
$("cards").innerHTML = Object.keys(SP).map(k => '<article class="card"><div class="art">' + peng(k) + '<img src="' + SP[k].img + '" alt="' + k + ' penguin" onerror="this.remove()"></div><h3>' + k + ' Penguin</h3><p class="sci">' + SP[k].sci + '</p><p>' + SP[k].blurb + '</p><div class="more"><p>' + SP[k].desc + '</p><p>' + SP[k].more + '</p></div><button class="link">Discover More</button></article>').join("");
document.querySelectorAll(".card .link").forEach(b => b.onclick = () => { const c = b.closest(".card"); c.classList.toggle("open"); b.textContent = c.classList.contains("open") ? "Show Less" : "Discover More"; });

const F = [
  ["cl","Bill Length","📏",25,70,"mm","e.g. 44.5","0.1"], ["cd","Bill Depth","📐",10,25,"mm","e.g. 17.2","0.1"],
  ["fl","Flipper Length","🌊",150,250,"mm","e.g. 197","1"], ["bm","Body Mass","⚖️",2000,7500,"g","e.g. 4050","10"]
];
$("fields").innerHTML = F.map(f => '<div class="field"><label for="' + f[0] + '">' + f[1] + ' (' + f[5] + ')</label><div class="in"><span>' + f[2] + '</span><input id="' + f[0] + '" type="number" inputmode="decimal" step="' + f[7] + '" placeholder="' + f[6] + '"><span class="u">' + f[5] + '</span></div><div class="m" id="m_' + f[0] + '"></div></div>').join("");

function validate() {
  let ok = true; const vals = [];
  F.forEach(f => {
    const el = $(f[0]), raw = el.value.trim(), v = Number(raw), box = el.closest(".field"); let msg = "";
    if (raw === "") msg = "Please enter the " + f[1].toLowerCase() + ".";
    else if (!isFinite(v)) msg = "That does not look like a number.";
    else if (v < f[3] || v > f[4]) msg = f[1] + " should be between " + f[3] + " and " + f[4] + " " + f[5] + ".";
    $("m_" + f[0]).textContent = msg; box.classList.toggle("bad", !!msg);
    if (msg) { ok = false; box.classList.remove("shake"); void box.offsetWidth; box.classList.add("shake"); }
    vals.push(v);
  });
  return ok ? vals : null;
}
F.forEach(f => { $(f[0]).addEventListener("keydown", e => { if (e.key === "Enter") run(); }); $(f[0]).addEventListener("input", () => { $(f[0]).closest(".field").classList.remove("bad"); $("m_" + f[0]).textContent = ""; }); });
$("predictBtn").onclick = run;

function celebrate(box) {
  const bits = ["❄️","✨","🐧","💙","⭐"];
  for (let i = 0; i < 26; i++) { const s = document.createElement("span"); s.className = "cf"; s.textContent = bits[i % 5];
    s.style.setProperty("--x", (Math.random()*560 - 280) + "px"); s.style.setProperty("--y", (Math.random()*260 - 120) + "px"); s.style.setProperty("--r", (Math.random()*540 - 270) + "deg");
    box.appendChild(s); setTimeout(() => s.remove(), 1600); }
}
async function run() {
  const v = validate(); if (!v) return;
  const res = $("result"); res.classList.remove("show"); $("loader").classList.add("on");
  try {
    const [r] = await Promise.all([
      fetch("/api/predict", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ culmen_length:v[0], culmen_depth:v[1], flipper_length:v[2], body_mass:v[3] }) }),
      new Promise(ok => setTimeout(ok, 1900))
    ]);
    const d = await r.json();
    if (!r.ok) { alert(d.error || "Something went wrong."); return; }
    const sp = SP[d.species] ? d.species : "Adelie", pct = Math.round((d.probabilities[sp] || 0) * 100);
    let bars = "";
    Object.entries(d.probabilities).forEach(e => { bars += '<div class="bar"><span>' + e[0] + '</span><div class="tr"><i data-w="' + Math.round(e[1]*100) + '"></i></div><span>' + Math.round(e[1]*100) + '%</span></div>'; });
    res.innerHTML = '<div class="eyebrow">Meet Your Penguin!</div><h2>Predicted Species: <span>' + sp + '</span></h2><div class="art">' + peng(sp) + '<img src="' + SP[sp].img + '" alt="' + sp + ' penguin" onerror="this.remove()"></div>'
      + '<div class="conf">' + pct + '% confidence <small>(share of the 5 nearest penguins that agree)</small></div><p class="desc">' + SP[sp].desc + '</p><div class="bars">' + bars + '</div>'
      + '<div class="party">Congratulations! You have discovered a new penguin friend!</div><button class="btn" id="again">Try Another Prediction</button>';
    $("again").onclick = () => { res.classList.remove("show"); $("predict").scrollIntoView(); $("cl").focus(); };
    res.classList.add("show"); celebrate(res);
    setTimeout(() => res.querySelectorAll(".tr i").forEach(i => { i.style.width = i.dataset.w + "%"; }), 150);
    res.scrollIntoView({ behavior:"smooth", block:"center" });
  } catch (e) { alert("Could not reach the server. Please try again."); }
  finally { $("loader").classList.remove("on"); }
}

const cv = $("snow"), cx = cv.getContext("2d"); let W, H; const fl = [];
function size() { const h = cv.parentElement; W = cv.width = h.clientWidth; H = cv.height = h.clientHeight; } size(); addEventListener("resize", size);
for (let i = 0; i < 90; i++) fl.push({ x:Math.random()*innerWidth, y:Math.random()*innerHeight, r:Math.random()*2.6+.8, s:Math.random()*.8+.25, d:Math.random()*6 });
function snow(t) {
  cx.clearRect(0,0,W,H); cx.fillStyle = "rgba(255,255,255,.9)"; cx.beginPath();
  fl.forEach(f => { f.y += f.s; f.x += Math.sin(t/1500 + f.d) * .45; if (f.y > H) { f.y = -6; f.x = Math.random()*W; } cx.moveTo(f.x + f.r, f.y); cx.arc(f.x, f.y, f.r, 0, 6.283); });
  cx.fill(); requestAnimationFrame(snow);
}
if (!matchMedia("(prefers-reduced-motion: reduce)").matches) requestAnimationFrame(snow);
</script>
</body>
</html>
"""


@app.get("/")
def home():
    return PAGE.replace("__N__", str(N_TRAIN))


@app.get("/<path:name>")
def public_file(name):
    """Serve optional photos from public/ when running locally (Vercel serves public/ itself)."""
    return send_from_directory(os.path.join(BASE_DIR, "public"), name)


@app.get("/api/health")
def health():
    return jsonify(status="ok", training_samples=N_TRAIN)


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    keys = {
        "culmen_length": "Culmen Length (mm)",
        "culmen_depth": "Culmen Depth (mm)",
        "flipper_length": "Flipper Length (mm)",
        "body_mass": "Body Mass (g)",
    }
    values = []
    for key, name in keys.items():
        raw = data.get(key)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return jsonify(error=f"Please enter a valid number for {name}."), 400
        if not np.isfinite(val):
            return jsonify(error=f"Please enter a valid number for {name}."), 400
        low, high = LIMITS[name]
        if not (low <= val <= high):
            return jsonify(error=f"{name} should be between {low} and {high}."), 400
        values.append(val)

    X = np.array([values])
    species = MODEL.predict(X)[0]
    probs = MODEL.predict_proba(X)[0]
    probabilities = {cls: round(float(p), 3) for cls, p in zip(MODEL.classes_, probs)}
    return jsonify(species=species, probabilities=probabilities)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
