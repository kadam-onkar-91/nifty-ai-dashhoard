"""Premium visual layer for the Nifty 50 AI dashboard.

Purely presentational: nothing here touches data, engines or trading logic.

    import ui_theme
    ui_theme.inject_theme()   # global CSS (dark glass look, animations)
    ui_theme.render_hero()    # animated particle-terrain hero banner

Both calls fail soft -- if anything goes wrong the dashboard simply renders
with the default Streamlit look.
"""
from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --bg:#05060d; --panel:rgba(255,255,255,.045); --panel-hi:rgba(255,255,255,.075);
  --line:rgba(255,255,255,.09); --txt:#e8ecf7; --mute:#8b93ad;
  --c1:#22d3ee; --c2:#8b5cf6; --c3:#f43f5e; --up:#22e6a8; --down:#ff5d7a;
}

html, body, .stApp, [class*="css"]{ font-family:'Space Grotesk',system-ui,sans-serif; }
code, pre, [data-testid="stMetricValue"]{ font-family:'JetBrains Mono','Space Grotesk',monospace; }

/* ---------- animated aurora background ---------- */
.stApp{
  background:
    radial-gradient(1100px 600px at 12% -10%, rgba(139,92,246,.20), transparent 60%),
    radial-gradient(900px 600px at 95% 5%,  rgba(34,211,238,.14), transparent 60%),
    radial-gradient(900px 700px at 50% 110%, rgba(244,63,94,.10), transparent 60%),
    var(--bg);
  color:var(--txt);
}
.stApp::before{
  content:""; position:fixed; inset:-20%; z-index:0; pointer-events:none;
  background:
    radial-gradient(40% 35% at 25% 30%, rgba(139,92,246,.16), transparent 70%),
    radial-gradient(35% 30% at 75% 65%, rgba(34,211,238,.12), transparent 70%);
  filter:blur(40px); animation:aurora 22s ease-in-out infinite alternate;
}
@keyframes aurora{
  0%{ transform:translate3d(-3%,-2%,0) scale(1); }
  50%{ transform:translate3d(4%,3%,0) scale(1.08); }
  100%{ transform:translate3d(-2%,5%,0) scale(1.02); }
}
header[data-testid="stHeader"]{ background:transparent; }
.main .block-container{ position:relative; z-index:1; padding-top:1.2rem; max-width:1500px; }

/* ---------- headings ---------- */
h1,h2,h3,h4{ letter-spacing:-.02em; }
h3,h4{ color:var(--txt); }
h4{ position:relative; padding-left:14px; }
h4::before{
  content:""; position:absolute; left:0; top:.28em; bottom:.28em; width:4px; border-radius:4px;
  background:linear-gradient(180deg,var(--c1),var(--c2));
  box-shadow:0 0 14px rgba(34,211,238,.6);
}
hr{ border:none !important; height:1px; background:linear-gradient(90deg,transparent,var(--line),transparent) !important; }

/* ---------- glass cards: metrics ---------- */
[data-testid="stMetric"]{
  background:linear-gradient(145deg,var(--panel-hi),var(--panel));
  border:1px solid var(--line); border-radius:16px; padding:16px 18px;
  backdrop-filter:blur(14px) saturate(140%); -webkit-backdrop-filter:blur(14px) saturate(140%);
  box-shadow:0 8px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.06);
  transition:transform .35s cubic-bezier(.2,.8,.2,1), border-color .35s, box-shadow .35s;
  animation:fadeUp .7s cubic-bezier(.2,.8,.2,1) both; position:relative; overflow:hidden;
}
[data-testid="stMetric"]::after{
  content:""; position:absolute; top:0; left:-60%; width:40%; height:100%;
  background:linear-gradient(100deg,transparent,rgba(255,255,255,.07),transparent);
  transform:skewX(-20deg); transition:left .8s ease;
}
[data-testid="stMetric"]:hover{
  transform:translateY(-4px); border-color:rgba(34,211,238,.45);
  box-shadow:0 14px 40px rgba(0,0,0,.5), 0 0 28px rgba(34,211,238,.15);
}
[data-testid="stMetric"]:hover::after{ left:130%; }
[data-testid="stMetricLabel"]{ color:var(--mute); text-transform:uppercase; font-size:.72rem; letter-spacing:.09em; }
[data-testid="stMetricValue"]{ font-weight:500; }

/* ---------- alerts ---------- */
[data-testid="stAlert"]{
  border-radius:14px; border:1px solid var(--line);
  background:var(--panel); backdrop-filter:blur(12px);
  animation:fadeUp .6s cubic-bezier(.2,.8,.2,1) both;
}

/* ---------- tabs as pills ---------- */
.stTabs [data-baseweb="tab-list"]{
  gap:6px; background:var(--panel); padding:6px; border-radius:999px;
  border:1px solid var(--line); backdrop-filter:blur(12px); overflow-x:auto;
}
.stTabs [data-baseweb="tab"]{
  height:40px; border-radius:999px; padding:0 18px; color:var(--mute);
  transition:color .25s, background .25s, transform .25s; white-space:nowrap;
}
.stTabs [data-baseweb="tab"]:hover{ color:var(--txt); background:rgba(255,255,255,.05); }
.stTabs [aria-selected="true"]{
  color:#fff !important;
  background:linear-gradient(135deg,rgba(139,92,246,.55),rgba(34,211,238,.35)) !important;
  box-shadow:0 0 22px rgba(139,92,246,.35);
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{ display:none; }
.stTabs [data-baseweb="tab-panel"]{ animation:fadeUp .55s cubic-bezier(.2,.8,.2,1) both; padding-top:1.1rem; }

/* ---------- buttons ---------- */
.stButton > button, .stDownloadButton > button{
  border-radius:12px; border:1px solid var(--line); color:var(--txt);
  background:linear-gradient(145deg,rgba(139,92,246,.35),rgba(34,211,238,.18));
  transition:transform .25s, box-shadow .25s, border-color .25s; font-weight:500;
}
.stButton > button:hover, .stDownloadButton > button:hover{
  transform:translateY(-2px); border-color:rgba(34,211,238,.6);
  box-shadow:0 8px 26px rgba(34,211,238,.22), 0 0 0 1px rgba(34,211,238,.25) inset; color:#fff;
}
.stButton > button:active{ transform:translateY(0) scale(.98); }

/* ---------- inputs ---------- */
.stTextInput input, .stNumberInput input, .stTextArea textarea, [data-baseweb="select"] > div{
  background:var(--panel) !important; border-radius:12px !important;
  border:1px solid var(--line) !important; color:var(--txt) !important;
  transition:border-color .25s, box-shadow .25s;
}
.stTextInput input:focus, .stTextArea textarea:focus{
  border-color:var(--c1) !important; box-shadow:0 0 0 3px rgba(34,211,238,.18) !important;
}

/* ---------- sidebar ---------- */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,rgba(14,16,30,.92),rgba(8,9,18,.92));
  border-right:1px solid var(--line); backdrop-filter:blur(18px);
}

/* ---------- tables, charts, expanders ---------- */
[data-testid="stDataFrame"], [data-testid="stTable"]{
  border:1px solid var(--line); border-radius:14px; overflow:hidden;
  box-shadow:0 8px 30px rgba(0,0,0,.3);
}
[data-testid="stPlotlyChart"]{
  border:1px solid var(--line); border-radius:16px; padding:6px;
  background:linear-gradient(145deg,rgba(255,255,255,.035),rgba(255,255,255,.01));
  animation:fadeUp .8s cubic-bezier(.2,.8,.2,1) both;
}
[data-testid="stExpander"]{
  border:1px solid var(--line) !important; border-radius:14px !important; background:var(--panel);
}
[data-testid="stExpander"] summary:hover{ color:var(--c1); }

/* ---------- misc ---------- */
@keyframes fadeUp{ from{ opacity:0; transform:translateY(18px); } to{ opacity:1; transform:none; } }
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-thumb{ background:linear-gradient(var(--c2),var(--c1)); border-radius:10px; }
::-webkit-scrollbar-track{ background:transparent; }
::selection{ background:rgba(139,92,246,.45); }
iframe[title="st.iframe"], iframe[title="streamlit_component"]{ border:0; }

@media (max-width:768px){
  .main .block-container{ padding-left:.6rem !important; padding-right:.6rem !important; padding-top:.8rem !important; }
  h1{ font-size:1.4rem !important; } h2{ font-size:1.1rem !important; }
  div[data-testid="stDataFrame"], div[data-testid="stTable"]{ width:100% !important; }
  [data-testid="stMetric"]{ padding:12px 14px; }
}
@media (prefers-reduced-motion:reduce){
  *, *::before, *::after{ animation:none !important; transition:none !important; }
}
</style>
"""

# ---------------------------------------------------------------------------
# Hero (self-contained: canvas + CSS + JS, no external libraries)
# ---------------------------------------------------------------------------
_HERO_HTML = r"""
<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:100%;height:100%;background:transparent;overflow:hidden;
  font-family:'Space Grotesk',system-ui,sans-serif;color:#eaf0ff}
#wrap{position:relative;width:100%;height:100%;border-radius:22px;overflow:hidden;
  background:radial-gradient(120% 120% at 70% 0%,#12122b 0%,#070812 55%,#04050b 100%);
  border:1px solid rgba(255,255,255,.09);box-shadow:0 20px 70px rgba(0,0,0,.55)}
canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
.vig{position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(90% 80% at 50% 45%,transparent 40%,rgba(3,4,10,.75) 100%),
             linear-gradient(90deg,rgba(4,5,12,.85) 0%,rgba(4,5,12,.35) 45%,transparent 70%)}
.content{position:absolute;left:clamp(18px,4vw,56px);top:50%;transform:translateY(-50%);
  max-width:min(620px,88%);z-index:3}
.eyebrow{display:inline-flex;align-items:center;gap:10px;font:500 11px/1 'JetBrains Mono',monospace;
  letter-spacing:.18em;text-transform:uppercase;color:#9fb0d8;padding:8px 14px;border-radius:999px;
  background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);backdrop-filter:blur(10px);
  opacity:0;animation:up .9s .1s cubic-bezier(.2,.8,.2,1) forwards}
.dot{width:7px;height:7px;border-radius:50%;background:#22e6a8;box-shadow:0 0 0 0 rgba(34,230,168,.7);
  animation:pulse 1.8s infinite}
@keyframes pulse{70%{box-shadow:0 0 0 10px rgba(34,230,168,0)}100%{box-shadow:0 0 0 0 rgba(34,230,168,0)}}
h1{margin:18px 0 14px;font-weight:700;line-height:.98;letter-spacing:-.035em;
  font-size:clamp(34px,6.2vw,76px);text-transform:uppercase}
h1 .l{display:block;white-space:nowrap}
.grad{background:linear-gradient(100deg,#22d3ee 0%,#8b5cf6 50%,#f43f5e 100%);
  -webkit-background-clip:text;background-clip:text;color:transparent;background-size:200% 100%;
  animation:shine 6s linear infinite}
@keyframes shine{to{background-position:-200% 0}}
.sub{font-size:clamp(12.5px,1.35vw,16px);line-height:1.55;color:#a7b2d1;max-width:480px;
  opacity:0;animation:up .9s .55s cubic-bezier(.2,.8,.2,1) forwards}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:20px;opacity:0;animation:up .9s .8s cubic-bezier(.2,.8,.2,1) forwards}
.chip{font:500 11px/1 'JetBrains Mono',monospace;letter-spacing:.06em;padding:9px 13px;border-radius:999px;
  color:#cfd8f5;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.12);
  backdrop-filter:blur(8px);transition:.3s}
.chip:hover{border-color:#22d3ee;color:#fff;box-shadow:0 0 18px rgba(34,211,238,.35);transform:translateY(-2px)}
@keyframes up{from{opacity:0;transform:translateY(22px)}to{opacity:1;transform:none}}

/* floating glass card */
.card{position:absolute;right:clamp(14px,4vw,56px);top:50%;width:min(300px,34%);z-index:3;
  transform:translateY(-50%) perspective(900px) rotateY(var(--ry,-10deg)) rotateX(var(--rx,4deg));
  transition:transform .15s ease-out;padding:16px 16px 12px;border-radius:20px;
  background:linear-gradient(145deg,rgba(255,255,255,.14),rgba(255,255,255,.04));
  border:1px solid rgba(255,255,255,.2);backdrop-filter:blur(18px) saturate(150%);
  box-shadow:0 30px 80px rgba(0,0,0,.55),inset 0 1px 0 rgba(255,255,255,.25);
  opacity:0;animation:cardIn 1.2s .6s cubic-bezier(.2,.8,.2,1) forwards}
@keyframes cardIn{from{opacity:0;margin-top:40px}to{opacity:1;margin-top:0}}
.card .row{display:flex;justify-content:space-between;align-items:center;
  font:500 10px/1 'JetBrains Mono',monospace;letter-spacing:.14em;color:#9fb0d8;text-transform:uppercase}
.card .px{font:500 clamp(20px,2.6vw,30px)/1.1 'JetBrains Mono',monospace;margin:10px 0 2px;letter-spacing:-.02em}
.card .chg{font:500 12px 'JetBrains Mono',monospace;color:#22e6a8}
.card canvas.spark{position:relative;inset:auto;width:100%;height:64px;margin-top:8px}
.card .bars{display:flex;gap:6px;margin-top:10px}
.bar{flex:1;font:500 9px/1 'JetBrains Mono',monospace;color:#9fb0d8;letter-spacing:.08em}
.bar i{display:block;height:4px;border-radius:4px;margin-top:6px;background:rgba(255,255,255,.1);position:relative;overflow:hidden}
.bar i::after{content:"";position:absolute;inset:0;width:var(--w);border-radius:4px;
  background:linear-gradient(90deg,#22d3ee,#8b5cf6);animation:grow 1.6s 1.2s cubic-bezier(.2,.8,.2,1) backwards}
@keyframes grow{from{width:0}}
.scroll{position:absolute;left:50%;bottom:14px;transform:translateX(-50%);z-index:3;
  font:500 10px 'JetBrains Mono',monospace;letter-spacing:.3em;color:#7f8bb0;text-transform:uppercase;
  display:flex;flex-direction:column;align-items:center;gap:8px;opacity:.9}
.scroll b{width:1px;height:26px;background:linear-gradient(#22d3ee,transparent);animation:drip 2s infinite}
@keyframes drip{0%{transform:scaleY(0);transform-origin:top}50%{transform:scaleY(1);transform-origin:top}
  51%{transform-origin:bottom}100%{transform:scaleY(0);transform-origin:bottom}}
@media (max-width:760px){
  .card{display:none}
  .vig{background:radial-gradient(90% 80% at 50% 45%,transparent 30%,rgba(3,4,10,.8) 100%),
        linear-gradient(90deg,rgba(4,5,12,.7),rgba(4,5,12,.3))}
  .sub{max-width:92%}
}
@media (prefers-reduced-motion:reduce){*{animation-duration:.01s!important}}
</style></head>
<body>
<div id="wrap">
  <canvas id="bg"></canvas>
  <div class="vig"></div>
  <div class="content">
    <div class="eyebrow"><span class="dot"></span>NSE &middot; Live AI Engine</div>
    <h1><span class="l" data-s="NIFTY 50">NIFTY 50</span><span class="l grad" data-s="AI TRADING">AI TRADING</span><span class="l" data-s="TERMINAL">TERMINAL</span></h1>
    <p class="sub">Smart-money flow, option-chain intelligence, regime detection and risk gating &mdash; fused into one real-time decision engine.</p>
    <div class="chips"><span class="chip">SMC / ICT</span><span class="chip">OPTIONS FLOW</span><span class="chip">REGIME AI</span><span class="chip">RISK ENGINE</span></div>
  </div>
  <div class="card" id="card">
    <div class="row"><span>NIFTY &middot; AI PULSE</span><span style="color:#22e6a8">&#9679; LIVE</span></div>
    <div class="px" id="px">24,812.40</div>
    <div class="chg" id="chg">&#9650; +0.62%</div>
    <canvas class="spark" id="spark"></canvas>
    <div class="bars">
      <div class="bar">TREND<i style="--w:78%"></i></div>
      <div class="bar">FLOW<i style="--w:64%"></i></div>
      <div class="bar">RISK<i style="--w:32%"></i></div>
    </div>
  </div>
  <div class="scroll"><span>Scroll</span><b></b></div>
</div>
<script>
(function(){
  const wrap=document.getElementById('wrap'), cv=document.getElementById('bg'), ctx=cv.getContext('2d');
  const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
  let W=0,H=0,DPR=Math.min(devicePixelRatio||1,2),mx=0,my=0,tx=0,ty=0,visible=true;

  function resize(){const r=wrap.getBoundingClientRect();W=r.width;H=r.height;
    cv.width=W*DPR;cv.height=H*DPR;ctx.setTransform(DPR,0,0,DPR,0,0);}
  addEventListener('resize',resize);resize();
  wrap.addEventListener('pointermove',e=>{const r=wrap.getBoundingClientRect();
    tx=(e.clientX-r.left)/r.width-.5;ty=(e.clientY-r.top)/r.height-.5;
    const c=document.getElementById('card');
    c.style.setProperty('--ry',(-10+tx*14)+'deg');c.style.setProperty('--rx',(4-ty*10)+'deg');});
  wrap.addEventListener('pointerleave',()=>{tx=0;ty=0;});
  new IntersectionObserver(es=>{visible=es[0].isIntersecting}).observe(wrap);

  /* ---- particle terrain ---- */
  const COLS=W<700?70:110, ROWS=W<700?34:52;
  function height(x,z,t,mxw){
    let y=Math.sin(x*1.3+t*.9)*.35+Math.sin(z*1.1-t*.7)*.3+Math.sin((x+z)*.6+t*.4)*.45;
    y+=Math.sin(x*3.1+z*2.3+t*1.6)*.08;
    const dx=x-mxw, dz=z-1.2; y+=Math.exp(-(dx*dx+dz*dz)*1.2)*.9*Math.sin(t*2.2); // pointer ripple
    return y;
  }
  /* drifting dust */
  const N=W<700?90:200, dust=[];
  for(let i=0;i<N;i++)dust.push({x:Math.random(),y:Math.random(),z:Math.random(),s:Math.random()*1.6+.3,v:Math.random()*.0004+.0001});
  const pal=[[34,211,238],[139,92,246],[244,63,94]];
  function col(k){k=Math.max(0,Math.min(1,k))*2;const i=Math.min(1,Math.floor(k)),f=k-i;
    const a=pal[i],b=pal[i+1];return [a[0]+(b[0]-a[0])*f,a[1]+(b[1]-a[1])*f,a[2]+(b[2]-a[2])*f];}

  let cx=0,cy=0,t0=performance.now();
  function frame(now){
    requestAnimationFrame(frame);
    if(!visible)return;
    const t=reduce?0:(now-t0)/1000;
    mx+=(tx-mx)*.05;my+=(ty-my)*.05;
    ctx.clearRect(0,0,W,H);
    ctx.globalCompositeOperation='lighter';
    const fov=Math.min(W*.42,520), camH=1.25+my*.5, hor=H*.34, rot=mx*.45;
    const cosr=Math.cos(rot), sinr=Math.sin(rot), midX=W*.55, mxw=mx*6;
    for(let r=0;r<ROWS;r++){
      const z=(r/ROWS)*9;                     /* 0 = near, 9 = far */
      for(let c=0;c<COLS;c++){
        const x=(c/COLS-.5)*16;
        const y=height(x,z,t,mxw);
        const rx=x*cosr-(z-4)*sinr, zz=2.0+x*sinr*.0+ (z-4)*cosr+4+x*sinr;
        if(zz<=.6)continue;
        const sc=fov/zz;
        const px=midX+rx*sc, py=hor+(camH-y*.55)*sc;
        if(px<-10||px>W+10||py<-10||py>H+10)continue;
        const k=(y+1.1)/2.2, depth=1-r/ROWS;
        const [R,G,B]=col(k*.85+(c/COLS)*.25);
        const a=(.2+depth*.8)*(.45+k*.75);
        const s=Math.max(.7,sc*.02);
        ctx.fillStyle=`rgba(${R|0},${G|0},${B|0},${Math.min(a,1)})`;
        ctx.fillRect(px,py,s,s);
      }
    }
    /* floating dust w/ glow */
    for(const d of dust){
      d.y-=d.v;if(d.y<0){d.y=1;d.x=Math.random();}
      const px=d.x*W+Math.sin(t*.5+d.z*9)*14+mx*40*d.z, py=d.y*H;
      const g=ctx.createRadialGradient(px,py,0,px,py,d.s*5);
      g.addColorStop(0,'rgba(180,220,255,.85)');g.addColorStop(1,'rgba(139,92,246,0)');
      ctx.fillStyle=g;ctx.beginPath();ctx.arc(px,py,d.s*5,0,6.283);ctx.fill();
    }
    ctx.globalCompositeOperation='source-over';
  }
  requestAnimationFrame(frame);

  /* ---- glitch/scramble text reveal (like the reference reel) ---- */
  const chars='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789#%&/<>';
  document.querySelectorAll('h1 .l').forEach((el,i)=>{
    const final=el.dataset.s;let f=0;const total=22+i*8,start=300+i*220;
    if(reduce){el.textContent=final;return;}
    el.textContent='';
    setTimeout(function tick(){
      f++;let out='';
      for(let k=0;k<final.length;k++){
        if(final[k]===' '){out+=' ';continue;}
        out+= (k<(f/total)*final.length*1.15)?final[k]:chars[(Math.random()*chars.length)|0];
      }
      el.textContent=out;
      if(f<total)setTimeout(tick,32);else el.textContent=final;
    },start);
  });

  /* ---- live-looking pulse card (decorative; real data is below) ---- */
  const sp=document.getElementById('spark'),sc2=sp.getContext('2d');
  let pts=[],val=24812.4;const base=val;
  for(let i=0;i<48;i++){val+=(Math.random()-.46)*9;pts.push(val);}
  function drawSpark(){
    const w=sp.clientWidth,h=sp.clientHeight;if(!w)return;
    sp.width=w*DPR;sp.height=h*DPR;sc2.setTransform(DPR,0,0,DPR,0,0);sc2.clearRect(0,0,w,h);
    const mn=Math.min(...pts),mx2=Math.max(...pts),rg=(mx2-mn)||1;
    const X=i=>i/(pts.length-1)*w, Y=v=>h-4-((v-mn)/rg)*(h-10);
    const g=sc2.createLinearGradient(0,0,0,h);g.addColorStop(0,'rgba(34,211,238,.35)');g.addColorStop(1,'rgba(34,211,238,0)');
    sc2.beginPath();pts.forEach((v,i)=>i?sc2.lineTo(X(i),Y(v)):sc2.moveTo(X(i),Y(v)));
    sc2.lineTo(w,h);sc2.lineTo(0,h);sc2.closePath();sc2.fillStyle=g;sc2.fill();
    sc2.beginPath();pts.forEach((v,i)=>i?sc2.lineTo(X(i),Y(v)):sc2.moveTo(X(i),Y(v)));
    sc2.strokeStyle='#22d3ee';sc2.lineWidth=1.6;sc2.shadowColor='#22d3ee';sc2.shadowBlur=8;sc2.stroke();sc2.shadowBlur=0;
    const lx=X(pts.length-1),ly=Y(pts[pts.length-1]);
    sc2.beginPath();sc2.arc(lx-1,ly,3,0,6.283);sc2.fillStyle='#fff';sc2.fill();
  }
  function tickPrice(){
    val+=(Math.random()-.48)*7;pts.push(val);pts.shift();
    const ch=(val-base)/base*100,up=ch>=0;
    document.getElementById('px').textContent=val.toLocaleString('en-IN',{minimumFractionDigits:2,maximumFractionDigits:2});
    const c=document.getElementById('chg');c.textContent=(up?'▲ +':'▼ ')+ch.toFixed(2)+'%';c.style.color=up?'#22e6a8':'#ff5d7a';
    drawSpark();
  }
  drawSpark();setInterval(tickPrice,900);
})();
</script></body></html>
"""


def inject_theme() -> None:
    """Inject the global dark-glass CSS. Safe to call once per script run."""
    try:
        st.markdown(_THEME_CSS, unsafe_allow_html=True)
    except Exception:  # never let styling break the dashboard
        pass


def render_hero(height: int = 460) -> None:
    """Render the animated hero banner (canvas particle terrain + glass card)."""
    try:
        import streamlit.components.v1 as components
        components.html(_HERO_HTML, height=height, scrolling=False)
    except Exception:
        st.title("⚡ Nifty 50 Institutional AI Trading Dashboard (Pro Edition)")
