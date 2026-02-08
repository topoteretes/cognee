import os
import json
import colorsys

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage

logger = get_logger()


def _generate_provenance_colors(values):
    """Generate a deterministic color map for a set of provenance values."""
    color_map = {}
    unique = sorted(set(v for v in values if v))
    for i, name in enumerate(unique):
        hue = (i * 137.5) % 360
        r, g, b = colorsys.hls_to_rgb(hue / 360, 0.6, 0.65)
        color_map[name] = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    return color_map


async def cognee_network_visualization(graph_data, destination_file_path: str = None):
    nodes_data, edges_data = graph_data

    color_map = {
        "Entity": "#6510F4",
        "EntityType": "#A550FF",
        "DocumentChunk": "#0DFF00",
        "TextSummary": "#6510F4",
        "TableRow": "#A550FF",
        "TableType": "#6510F4",
        "ColumnValue": "#747470",
        "SchemaTable": "#A550FF",
        "DatabaseSchema": "#6510F4",
        "SchemaRelationship": "#323332",
        "default": "#DBD8D8",
    }

    nodes_list = []
    for node_id, node_info in nodes_data:
        node_info = node_info.copy()
        node_info["id"] = str(node_id)
        node_info["color"] = color_map.get(node_info.get("type", "default"), "#DBD8D8")
        node_info["name"] = node_info.get("name", str(node_id))
        node_info.pop("updated_at", None)
        node_info.pop("created_at", None)
        nodes_list.append(node_info)

    links_list = []
    for source, target, relation, edge_info in edges_data:
        source = str(source)
        target = str(target)

        all_weights = {}
        primary_weight = None

        if edge_info:
            if "weight" in edge_info:
                all_weights["default"] = edge_info["weight"]
                primary_weight = edge_info["weight"]
            if "weights" in edge_info and isinstance(edge_info["weights"], dict):
                all_weights.update(edge_info["weights"])
                if primary_weight is None and edge_info["weights"]:
                    primary_weight = next(iter(edge_info["weights"].values()))
            for key, value in edge_info.items():
                if key.startswith("weight_") and isinstance(value, (int, float)):
                    all_weights[key[7:]] = value

        links_list.append(
            {
                "source": source,
                "target": target,
                "relation": relation,
                "weight": primary_weight,
                "all_weights": all_weights,
                "relationship_type": edge_info.get("relationship_type") if edge_info else None,
                "edge_info": edge_info if edge_info else {},
            }
        )

    task_color_map = _generate_provenance_colors([n.get("source_task") for n in nodes_list])
    pipeline_color_map = _generate_provenance_colors([n.get("source_pipeline") for n in nodes_list])
    note_set_color_map = _generate_provenance_colors([n.get("source_note_set") for n in nodes_list])

    html_content = _build_html(
        nodes_list, links_list, task_color_map, pipeline_color_map, note_set_color_map
    )

    if not destination_file_path:
        home_dir = os.path.expanduser("~")
        destination_file_path = os.path.join(home_dir, "graph_visualization.html")

    dir_path = os.path.dirname(destination_file_path)
    file_path = os.path.basename(destination_file_path)

    file_storage = LocalFileStorage(dir_path)
    file_storage.store(file_path, html_content, overwrite=True)

    logger.info(f"Graph visualization saved as {destination_file_path}")

    return html_content


def _build_html(
    nodes_list, links_list, task_color_map=None, pipeline_color_map=None, note_set_color_map=None
):
    def _safe_json_embed(obj):
        return json.dumps(obj).replace("</", "<\\/")

    html_template = _get_html_template()
    html_content = html_template.replace("__NODES_DATA__", _safe_json_embed(nodes_list))
    html_content = html_content.replace("__LINKS_DATA__", _safe_json_embed(links_list))
    html_content = html_content.replace("__TASK_COLORS__", _safe_json_embed(task_color_map or {}))
    html_content = html_content.replace(
        "__PIPELINE_COLORS__", _safe_json_embed(pipeline_color_map or {})
    )
    html_content = html_content.replace(
        "__NOTESET_COLORS__", _safe_json_embed(note_set_color_map or {})
    )
    return html_content


def _get_html_template():
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cognee Knowledge Graph</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#000000;--bg2:#323332;--surface:rgba(255,255,255,0.06);
  --border:rgba(219,216,216,0.12);--text:#F4F4F4;--text2:#DBD8D8;
  --accent:#6510F4;--accent2:#A550FF;--green:#0DFF00;
}
@font-face{font-family:'Inter';src:local('Inter'),local('Inter-Regular');font-display:swap}
body,html{width:100%;height:100%;overflow:hidden;background:#000000;color:#F4F4F4;font-family:'Inter',system-ui,-apple-system,sans-serif}
canvas{display:block;width:100vw;height:100vh;cursor:grab}
canvas:active{cursor:grabbing}

#header{
  position:fixed;top:0;left:0;right:0;height:48px;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 20px;z-index:100;
  background:linear-gradient(180deg,rgba(0,0,0,0.95) 0%,rgba(0,0,0,0) 100%);
  pointer-events:none;
}
#header>*{pointer-events:auto}
.logo{display:flex;align-items:center;gap:8px;font-size:13px;font-weight:600;letter-spacing:0.02em;color:var(--text)}
.logo img{height:28px;width:auto}
.stats{display:flex;gap:16px;font-size:11px;color:var(--text2);font-variant-numeric:tabular-nums}
.stats span{display:flex;align-items:center;gap:4px}
.stats .dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}

#controls{
  position:fixed;bottom:20px;left:50%;transform:translateX(-50%);z-index:100;
  display:flex;gap:2px;padding:4px;
  background:var(--bg2);border:1px solid var(--border);border-radius:10px;
}
.ctrl-btn{
  padding:6px 14px;font-size:11px;font-weight:500;font-family:inherit;
  color:var(--text2);background:transparent;border:none;border-radius:7px;
  cursor:pointer;transition:all 0.15s ease;white-space:nowrap;
}
.ctrl-btn:hover{color:var(--text);background:var(--surface)}
.ctrl-btn.active{color:#F4F4F4;background:rgba(101,16,244,0.2)}
.ctrl-sep{width:1px;background:var(--border);margin:2px 4px}

#search-box{
  position:fixed;top:56px;left:20px;z-index:100;width:280px;
}
#search-input{
  width:100%;padding:8px 12px 8px 32px;
  font-size:12px;font-family:inherit;color:var(--text);
  background:var(--bg2);border:1px solid var(--border);border-radius:8px;
  outline:none;transition:border-color 0.15s;
}
#search-input:focus{border-color:var(--accent)}
#search-input::placeholder{color:var(--text2)}
.search-icon{position:absolute;left:10px;top:50%;transform:translateY(-50%);color:var(--text2);pointer-events:none}

#info-panel{
  position:fixed;top:56px;right:20px;width:320px;max-height:calc(100vh - 80px);
  overflow-y:auto;z-index:100;
  background:var(--bg2);border:1px solid var(--border);border-radius:12px;
  padding:16px;opacity:0;transform:translateX(8px);
  transition:opacity 0.2s ease,transform 0.2s ease;
}
#info-panel.visible{opacity:1;transform:translateX(0)}
#info-panel::-webkit-scrollbar{width:4px}
#info-panel::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
.panel-header{display:flex;align-items:center;gap:8px;margin-bottom:12px}
.panel-type{
  font-size:9px;font-weight:600;text-transform:uppercase;letter-spacing:0.06em;
  padding:2px 6px;border-radius:4px;background:rgba(101,16,244,0.2);color:#A550FF;
}
.panel-title{font-size:14px;font-weight:600;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.panel-section{margin-top:12px;padding-top:12px;border-top:1px solid var(--border)}
.panel-section-title{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:var(--text2);margin-bottom:8px}
.panel-row{display:flex;justify-content:space-between;align-items:flex-start;padding:3px 0;font-size:12px;gap:12px}
.panel-row .k{color:var(--text2);flex-shrink:0}
.panel-row .v{color:var(--text);text-align:right;word-break:break-word;max-width:200px}

#legend{
  position:fixed;bottom:64px;left:20px;z-index:100;
  display:flex;flex-wrap:wrap;gap:6px 12px;
  padding:10px 14px;background:var(--bg2);border:1px solid var(--border);border-radius:10px;
  max-width:400px;
}
.legend-item{display:flex;align-items:center;gap:5px;font-size:10px;color:var(--text2);white-space:nowrap}
.legend-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}

.empty-state{
  position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
  text-align:center;color:var(--text2);font-size:14px;
}

#loading-overlay{
  position:fixed;top:0;left:0;right:0;bottom:0;z-index:200;
  background:rgba(0,0,0,0.92);
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  transition:opacity 0.4s ease;
}
#loading-overlay.hidden{opacity:0;pointer-events:none}
#loading-text{font-size:14px;color:var(--text2);margin-bottom:16px}
#loading-bar-bg{width:240px;height:4px;background:var(--bg2);border-radius:2px;overflow:hidden}
#loading-bar{width:0%;height:100%;background:var(--accent);border-radius:2px;transition:width 0.15s linear}
#loading-stats{font-size:11px;color:var(--text2);margin-top:10px;opacity:0.7}

#minimap-container{
  position:fixed;bottom:64px;right:20px;z-index:100;
  border:1px solid var(--border);border-radius:8px;overflow:hidden;
  background:rgba(0,0,0,0.7);cursor:pointer;display:none;
}
#minimap-canvas{display:block}

#fps-counter{
  position:fixed;top:12px;right:180px;z-index:100;
  font-size:10px;color:var(--text2);font-variant-numeric:tabular-nums;
  display:none;pointer-events:none;
}
</style>
</head>
<body>

<div id="header">
  <div class="logo">
    <img src="data:image/png;base64,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" alt="Cognee">
  </div>
  <div class="stats" id="stats"></div>
</div>

<div id="search-box">
  <svg class="search-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>
  <input type="text" id="search-input" placeholder="Search nodes..." autocomplete="off" spellcheck="false">
</div>

<div id="controls">
  <button class="ctrl-btn active" data-layer="nodes">Nodes</button>
  <button class="ctrl-btn active" data-layer="edges">Edges</button>
  <button class="ctrl-btn active" data-layer="labels">Labels</button>
  <button class="ctrl-btn" data-layer="heatmap">Heatmap</button>
  <button class="ctrl-btn" data-layer="typeclouds">Type Clouds</button>
  <div class="ctrl-sep"></div>
  <span style="font-size:10px;color:var(--text2);padding:6px 4px">Color:</span>
  <button class="ctrl-btn active" data-colorby="type">Type</button>
  <button class="ctrl-btn" data-colorby="task">Task</button>
  <button class="ctrl-btn" data-colorby="pipeline">Pipeline</button>
  <button class="ctrl-btn" data-colorby="noteset">Note Set</button>
  <div class="ctrl-sep"></div>
  <button class="ctrl-btn" id="btn-zoom-out" title="Zoom out (-)">&#x2212;</button>
  <button class="ctrl-btn" id="btn-zoom-in" title="Zoom in (+)">+</button>
  <button class="ctrl-btn" id="btn-fit" title="Fit to view (F)">Fit</button>
</div>

<div id="info-panel"></div>
<div id="legend"></div>

<div id="loading-overlay">
  <div id="loading-text">Laying out graph...</div>
  <div id="loading-bar-bg"><div id="loading-bar"></div></div>
  <div id="loading-stats"></div>
</div>

<div id="minimap-container"><canvas id="minimap-canvas"></canvas></div>
<div id="fps-counter"></div>

<canvas id="canvas"></canvas>

<script>
(function(){
"use strict";

var nodes = __NODES_DATA__;
var links = __LINKS_DATA__;
var taskColors = __TASK_COLORS__;
var pipelineColors = __PIPELINE_COLORS__;
var notesetColors = __NOTESET_COLORS__;

if (!nodes || nodes.length === 0) {
  document.body.innerHTML = '<div class="empty-state">No graph data available</div>';
  return;
}

// ── Helpers ──
function esc(s){return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;")}
function truncate(s,n){if(!s)return"";return s.length>n?s.slice(0,n)+"\u2026":s}
function hexToRgb(hex){
  if(!hex)return[219,216,216];
  var c=hex.replace("#","");
  if(c.length===3)c=c[0]+c[0]+c[1]+c[1]+c[2]+c[2];
  var n=parseInt(c,16);
  return[(n>>16)&255,(n>>8)&255,n&255];
}
function hslToRgb(h,s,l){
  s/=100;l/=100;
  var a=s*Math.min(l,1-l);
  function f(n){var k=(n+h/30)%12;return l-a*Math.max(-1,Math.min(k-3,9-k,1))}
  return[Math.round(f(0)*255),Math.round(f(8)*255),Math.round(f(4)*255)];
}
function clamp(v,lo,hi){return v<lo?lo:v>hi?hi:v}
function lerp(a,b,t){return a+(b-a)*t}

var N=nodes.length;
var M=links.length;

// ── Resolve IDs ──
nodes.forEach(function(n){
  if(!n.id)n.id=n.node_id||n.uuid||n.external_id||n.name||Math.random().toString(36);
});
var nodeMap={};
nodes.forEach(function(n){nodeMap[n.id]=n});
links=links.filter(function(l){
  var s=typeof l.source==="object"?(l.source.id||l.source):l.source;
  var t=typeof l.target==="object"?(l.target.id||l.target):l.target;
  return nodeMap[s]&&nodeMap[t];
});
M=links.length;

// ── Compute node degrees for sizing ──
var degreeMap={};
nodes.forEach(function(n){degreeMap[n.id]=0});
links.forEach(function(l){
  var s=typeof l.source==="object"?l.source.id:l.source;
  var t=typeof l.target==="object"?l.target.id:l.target;
  degreeMap[s]=(degreeMap[s]||0)+1;
  degreeMap[t]=(degreeMap[t]||0)+1;
});
var maxDeg=Math.max(1,Math.max.apply(null,Object.values(degreeMap)));
nodes.forEach(function(n){n._degree=degreeMap[n.id]||0;n._r=4+Math.sqrt(n._degree/maxDeg)*10});

// Pre-sort nodes by degree descending (for adaptive labels)
var nodesByDegree=nodes.slice().sort(function(a,b){return b._degree-a._degree});
var topDegreeSet=new Set();
nodesByDegree.slice(0,Math.min(10,N)).forEach(function(n){topDegreeSet.add(n.id)});

// ── Color palette ──
var colorByType={
  "Entity":"#6510F4","EntityType":"#A550FF","DocumentChunk":"#0DFF00",
  "TextSummary":"#6510F4","TableRow":"#A550FF","TableType":"#6510F4",
  "ColumnValue":"#747470","SchemaTable":"#A550FF","DatabaseSchema":"#6510F4",
  "SchemaRelationship":"#323332"
};
var typeColors={};
var typeRgbCache={};
var hueStep=0;
nodes.forEach(function(n){
  if(!n.color)n.color=colorByType[n.type]||null;
  if(!n.color){
    if(!typeColors[n.type]){
      var hue=(hueStep*137.5)%360;hueStep++;
      typeColors[n.type]="hsl("+hue+",55%,65%)";
    }
    n.color=typeColors[n.type];
  }
  // Cache RGB per type
  if(!typeRgbCache[n.type]){
    var c=n.color;
    if(c.indexOf("hsl")===0){
      var m=c.match(/[\d.]+/g);
      typeRgbCache[n.type]=hslToRgb(+m[0],+m[1],+m[2]);
    }else{
      typeRgbCache[n.type]=hexToRgb(c);
    }
  }
  n._rgb=typeRgbCache[n.type];
});

// ── Color-by mode ──
var colorByMode="type";

function recolorNodes(){
  nodes.forEach(function(n){
    if(colorByMode==="type"){
      n.color=colorByType[n.type]||typeColors[n.type]||"#DBD8D8";
    }else if(colorByMode==="task"){
      n.color=taskColors[n.source_task||"Unknown"]||"#DBD8D8";
    }else if(colorByMode==="pipeline"){
      n.color=pipelineColors[n.source_pipeline||"Unknown"]||"#DBD8D8";
    }else{
      n.color=notesetColors[n.source_note_set||"Unknown"]||"#DBD8D8";
    }
    // Recache RGB
    if(n.color.indexOf("hsl")===0){
      var m=n.color.match(/[\d.]+/g);
      n._rgb=hslToRgb(+m[0],+m[1],+m[2]);
    }else{
      n._rgb=hexToRgb(n.color);
    }
  });
}

document.querySelectorAll(".ctrl-btn[data-colorby]").forEach(function(btn){
  btn.addEventListener("click",function(){
    document.querySelectorAll(".ctrl-btn[data-colorby]").forEach(function(b){b.classList.remove("active")});
    btn.classList.add("active");
    colorByMode=btn.dataset.colorby;
    recolorNodes();
    updateLegend();
    draw();
  });
});

// ── Stats ──
var typeCounts={};
nodes.forEach(function(n){typeCounts[n.type]=(typeCounts[n.type]||0)+1});
var taskCounts={};
nodes.forEach(function(n){var t=n.source_task||"Unknown";taskCounts[t]=(taskCounts[t]||0)+1});
var pipeCounts={};
nodes.forEach(function(n){var p=n.source_pipeline||"Unknown";pipeCounts[p]=(pipeCounts[p]||0)+1});
var notesetCounts={};
nodes.forEach(function(n){var s=n.source_note_set||"Unknown";notesetCounts[s]=(notesetCounts[s]||0)+1});
var uniqueTasks=Object.keys(taskCounts).length;
var uniquePipelines=Object.keys(pipeCounts).length;
var uniqueNotesets=Object.keys(notesetCounts).length;
var statsEl=document.getElementById("stats");
var perfTier=N>10000?"large":N>2000?"medium":"small";
statsEl.innerHTML='<span><span class="dot" style="background:#6510F4"></span>'+N.toLocaleString()+" nodes</span>"+
  '<span><span class="dot" style="background:#747470"></span>'+M.toLocaleString()+" edges</span>"+
  '<span>'+Object.keys(typeCounts).length+" types</span>"+
  '<span>'+uniqueTasks+" tasks</span>"+
  '<span>'+uniquePipelines+" pipelines</span>"+
  '<span>'+uniqueNotesets+" note sets</span>"+
  '<span style="opacity:0.5">['+perfTier+']</span>';

// ── Legend ──
var legendEl=document.getElementById("legend");

function updateLegend(){
  var entries,counts,colorSource;
  if(colorByMode==="type"){
    counts=typeCounts;
    entries=Object.keys(counts).sort(function(a,b){return counts[b]-counts[a]});
    colorSource=function(t){return colorByType[t]||typeColors[t]||"#DBD8D8"};
  }else if(colorByMode==="task"){
    counts=taskCounts;
    entries=Object.keys(counts).sort(function(a,b){return counts[b]-counts[a]});
    colorSource=function(t){return taskColors[t]||"#DBD8D8"};
  }else if(colorByMode==="pipeline"){
    counts=pipeCounts;
    entries=Object.keys(counts).sort(function(a,b){return counts[b]-counts[a]});
    colorSource=function(t){return pipelineColors[t]||"#DBD8D8"};
  }else{
    counts=notesetCounts;
    entries=Object.keys(counts).sort(function(a,b){return counts[b]-counts[a]});
    colorSource=function(t){return notesetColors[t]||"#DBD8D8"};
  }
  if(entries.length>12)entries=entries.slice(0,12);
  legendEl.innerHTML=entries.map(function(t){
    return '<div class="legend-item"><div class="legend-dot" style="background:'+colorSource(t)+'"></div>'+esc(t)+" ("+counts[t]+")</div>";
  }).join("");
}
updateLegend();

// ── Canvas setup ──
var canvas=document.getElementById("canvas");
var ctx=canvas.getContext("2d");
var dpr=window.devicePixelRatio||1;
var W,H;
function resize(){
  W=window.innerWidth;H=window.innerHeight;
  canvas.width=W*dpr;canvas.height=H*dpr;
  canvas.style.width=W+"px";canvas.style.height=H+"px";
  ctx.setTransform(dpr,0,0,dpr,0,0);
}
resize();
window.addEventListener("resize",function(){resize();draw()});

// ── Layer toggles ──
var layers={nodes:true,edges:true,labels:true,heatmap:false,typeclouds:false};
document.querySelectorAll(".ctrl-btn[data-layer]").forEach(function(btn){
  btn.addEventListener("click",function(){
    var key=btn.dataset.layer;
    layers[key]=!layers[key];
    btn.classList.toggle("active",layers[key]);
    draw();
  });
});

// ── Transform state ──
var tx=W/2,ty=H/2,scale=1;

// ── Smooth zoom animation state ──
var zoomAnim=null; // {fromScale,toScale,fromTx,toTx,fromTy,toTy,startTime,duration}

function animateZoom(){
  if(!zoomAnim)return;
  var elapsed=performance.now()-zoomAnim.startTime;
  var t=Math.min(1,elapsed/zoomAnim.duration);
  // Ease out cubic
  var e=1-Math.pow(1-t,3);
  scale=lerp(zoomAnim.fromScale,zoomAnim.toScale,e);
  tx=lerp(zoomAnim.fromTx,zoomAnim.toTx,e);
  ty=lerp(zoomAnim.fromTy,zoomAnim.toTy,e);
  draw();
  if(t<1){requestAnimationFrame(animateZoom)}
  else{zoomAnim=null}
}

function smoothZoomTo(newScale,pivotX,pivotY,duration){
  newScale=clamp(newScale,0.02,30);
  var ratio=newScale/scale;
  var newTx=pivotX-(pivotX-tx)*ratio;
  var newTy=pivotY-(pivotY-ty)*ratio;
  zoomAnim={fromScale:scale,toScale:newScale,fromTx:tx,toTx:newTx,fromTy:ty,toTy:newTy,startTime:performance.now(),duration:duration||150};
  requestAnimationFrame(animateZoom);
}

function smoothPanZoomTo(newScale,newTx,newTy,duration){
  newScale=clamp(newScale,0.02,30);
  zoomAnim={fromScale:scale,toScale:newScale,fromTx:tx,toTx:newTx,fromTy:ty,toTy:newTy,startTime:performance.now(),duration:duration||300};
  requestAnimationFrame(animateZoom);
}

// ── Zoom-to-fit ──
function zoomToFit(animated){
  var minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  var valid=0;
  nodes.forEach(function(n){
    if(typeof n.x!=="number"||!isFinite(n.x))return;
    valid++;
    if(n.x<minX)minX=n.x;if(n.x>maxX)maxX=n.x;
    if(n.y<minY)minY=n.y;if(n.y>maxY)maxY=n.y;
  });
  if(valid===0){return}
  var pad=0.1;
  var gw=maxX-minX||100;
  var gh=maxY-minY||100;
  var cx=(minX+maxX)/2;
  var cy=(minY+maxY)/2;
  var sx=W*(1-2*pad)/gw;
  var sy=H*(1-2*pad)/gh;
  var ns=Math.min(sx,sy,10);
  var ntx=W/2-cx*ns;
  var nty=H/2-cy*ns;
  if(animated){smoothPanZoomTo(ns,ntx,nty,400)}
  else{scale=ns;tx=ntx;ty=nty;draw()}
}

// ── Force simulation (Phase 3f: adaptive tuning) ──
var repStr,linkDist,alphaDecayVal,velDecayVal,distMaxVal,collideIter;
if(N>10000){
  repStr=-20;linkDist=20;alphaDecayVal=0.05;velDecayVal=0.6;distMaxVal=200;collideIter=0;
}else if(N>5000){
  repStr=-40;linkDist=30;alphaDecayVal=0.04;velDecayVal=0.5;distMaxVal=300;collideIter=0;
}else if(N>2000){
  repStr=-60;linkDist=40;alphaDecayVal=0.03;velDecayVal=0.45;distMaxVal=400;collideIter=1;
}else if(N>500){
  repStr=-120;linkDist=70;alphaDecayVal=0.02;velDecayVal=0.4;distMaxVal=500;collideIter=1;
}else{
  repStr=-180;linkDist=100;alphaDecayVal=0.02;velDecayVal=0.4;distMaxVal=600;collideIter=1;
}

var MAX_TICKS=N>10000?300:N>5000?400:1000;
var simTickCount=0;
var simDone=false;

// ── Loading overlay (Phase 3g) ──
var loadingOverlay=document.getElementById("loading-overlay");
var loadingBar=document.getElementById("loading-bar");
var loadingText=document.getElementById("loading-text");
var loadingStatsEl=document.getElementById("loading-stats");
var showLoading=N>200;

if(!showLoading){
  loadingOverlay.classList.add("hidden");
}else{
  loadingStatsEl.textContent=N.toLocaleString()+" nodes, "+M.toLocaleString()+" edges";
}

var simulation=d3.forceSimulation(nodes)
  .force("link",d3.forceLink(links).id(function(d){return d.id}).distance(linkDist).strength(0.2))
  .force("charge",d3.forceManyBody().strength(repStr).distanceMax(distMaxVal))
  .force("center",d3.forceCenter(0,0))
  .force("x",d3.forceX(0).strength(0.04))
  .force("y",d3.forceY(0).strength(0.04))
  .alphaDecay(alphaDecayVal)
  .velocityDecay(velDecayVal);

if(collideIter>0){
  simulation.force("collide",d3.forceCollide().radius(function(d){return d._r+2}).iterations(collideIter));
}

// ── Quadtree (Phase 3a) ──
var quadtree=null;
var qtRebuildInterval=N>10000?10:N>2000?5:3;

function rebuildQuadtree(){
  quadtree=d3.quadtree()
    .x(function(d){return d.x})
    .y(function(d){return d.y})
    .addAll(nodes.filter(function(n){return typeof n.x==="number"&&isFinite(n.x)}));
}

// ── Adjacency index ──
var adjMap={};
nodes.forEach(function(n){adjMap[n.id]=[]});

// Simulation tick handler
simulation.on("tick",function(){
  simTickCount++;

  // Rebuild adjacency once after link objects are resolved
  if(simTickCount===1){
    adjMap={};
    nodes.forEach(function(n2){adjMap[n2.id]=[]});
    links.forEach(function(l){
      var sid=l.source.id||l.source;
      var tid=l.target.id||l.target;
      if(adjMap[sid])adjMap[sid].push(tid);
      if(adjMap[tid])adjMap[tid].push(sid);
    });
  }

  // Rebuild quadtree periodically
  if(simTickCount%qtRebuildInterval===0){rebuildQuadtree()}

  // Update loading progress
  if(showLoading&&!simDone){
    var pct=Math.min(99,Math.round(simTickCount/MAX_TICKS*100));
    loadingBar.style.width=pct+"%";
    loadingText.textContent="Laying out graph... "+pct+"%";
  }

  // Cap simulation ticks
  if(simTickCount>=MAX_TICKS&&!simDone){
    simulation.stop();
    simDone=true;
    rebuildQuadtree();
    computeDensity();
    if(showLoading){
      loadingBar.style.width="100%";
      loadingText.textContent="Done!";
      setTimeout(function(){loadingOverlay.classList.add("hidden")},300);
    }
    zoomToFit(false);
    setupMinimap();
    draw();
    return;
  }

  // Only draw during simulation if small graph, otherwise skip frames
  if(N<2000||simTickCount%3===0){draw()}
});

// When simulation ends naturally (alpha < alphaMin)
simulation.on("end",function(){
  if(!simDone){
    simDone=true;
    rebuildQuadtree();
    computeDensity();
    if(showLoading){
      loadingBar.style.width="100%";
      loadingText.textContent="Done!";
      setTimeout(function(){loadingOverlay.classList.add("hidden")},300);
    }
    zoomToFit(false);
    setupMinimap();
    draw();
  }
});

// ── Interaction state ──
var hoveredNode=null;
var searchQuery="";
var searchMatches=new Set();

// ── Search ──
var searchInput=document.getElementById("search-input");
searchInput.addEventListener("input",function(){
  searchQuery=searchInput.value.trim().toLowerCase();
  searchMatches.clear();
  if(searchQuery){
    nodes.forEach(function(n){
      var name=(n.name||"").toLowerCase();
      var type=(n.type||"").toLowerCase();
      if(name.indexOf(searchQuery)>=0||type.indexOf(searchQuery)>=0)searchMatches.add(n.id);
    });
  }
  draw();
});

// ── Info panel ──
var infoPanel=document.getElementById("info-panel");
function showNodeInfo(n){
  if(!n){infoPanel.classList.remove("visible");return}
  var html='<div class="panel-header">';
  html+='<span class="panel-type" style="background:'+n.color+'22;color:'+n.color+'">'+esc(n.type||"Node")+"</span>";
  html+='<span class="panel-title">'+esc(truncate(n.name||n.id,40))+"</span></div>";

  var desc=null;
  var descKeys=["description","summary","text","content"];
  for(var i=0;i<descKeys.length;i++){
    var v=n[descKeys[i]]||(n.properties&&n.properties[descKeys[i]]);
    if(typeof v==="string"&&v.trim()){desc=v.trim();break}
  }
  if(desc){
    html+='<div style="font-size:12px;color:var(--text2);line-height:1.5;margin-top:8px">'+esc(truncate(desc,300))+"</div>";
  }

  html+='<div class="panel-section"><div class="panel-section-title">Properties</div>';
  html+='<div class="panel-row"><span class="k">ID</span><span class="v" style="font-size:10px;opacity:0.7">'+esc(truncate(n.id,36))+"</span></div>";
  html+='<div class="panel-row"><span class="k">Connections</span><span class="v">'+n._degree+"</span></div>";
  if(n.source_task)html+='<div class="panel-row"><span class="k">Source Task</span><span class="v">'+esc(n.source_task)+"</span></div>";
  if(n.source_pipeline)html+='<div class="panel-row"><span class="k">Source Pipeline</span><span class="v">'+esc(n.source_pipeline)+"</span></div>";
  if(n.source_note_set)html+='<div class="panel-row"><span class="k">Source Note Set</span><span class="v">'+esc(n.source_note_set)+"</span></div>";

  if(n.properties){
    Object.keys(n.properties).slice(0,10).forEach(function(key){
      var v=n.properties[key];
      if(v!==undefined&&v!==null&&typeof v!=="object"){
        html+='<div class="panel-row"><span class="k">'+esc(key)+'</span><span class="v">'+esc(truncate(String(v),80))+"</span></div>";
      }
    });
  }
  html+="</div>";

  infoPanel.innerHTML=html;
  infoPanel.classList.add("visible");
}

function showEdgeInfo(e){
  var html='<div class="panel-header">';
  html+='<span class="panel-type">Edge</span>';
  html+='<span class="panel-title">'+esc(e.relation||"relationship")+"</span></div>";
  html+='<div class="panel-section"><div class="panel-section-title">Details</div>';
  var sName=e.source.name||e.source.id;
  var tName=e.target.name||e.target.id;
  html+='<div class="panel-row"><span class="k">From</span><span class="v">'+esc(truncate(sName,30))+"</span></div>";
  html+='<div class="panel-row"><span class="k">To</span><span class="v">'+esc(truncate(tName,30))+"</span></div>";
  if(e.weight!=null)html+='<div class="panel-row"><span class="k">Weight</span><span class="v">'+e.weight+"</span></div>";
  if(e.all_weights&&Object.keys(e.all_weights).length>0){
    Object.keys(e.all_weights).forEach(function(k){
      html+='<div class="panel-row"><span class="k">w:'+esc(k)+'</span><span class="v">'+e.all_weights[k]+"</span></div>";
    });
  }
  if(e.relationship_type)html+='<div class="panel-row"><span class="k">Type</span><span class="v">'+esc(e.relationship_type)+"</span></div>";
  html+="</div>";
  infoPanel.innerHTML=html;
  infoPanel.classList.add("visible");
}

// ── Hit testing (Phase 3a: quadtree-accelerated) ──
function screenToWorld(sx,sy){return{x:(sx-tx)/scale,y:(sy-ty)/scale}}
function worldToScreen(wx,wy){return{x:wx*scale+tx,y:wy*scale+ty}}

function findNodeAt(sx,sy){
  var w=screenToWorld(sx,sy);
  var searchR=(20)/scale; // max search radius in world coords
  if(quadtree){
    var found=quadtree.find(w.x,w.y,searchR);
    if(found){
      var dx=found.x-w.x,dy=found.y-w.y;
      var d=Math.sqrt(dx*dx+dy*dy);
      var hitR=(found._r+4)/scale;
      if(d<hitR)return found;
    }
    return null;
  }
  // Fallback for before quadtree is built
  var best=null,bestD=Infinity;
  for(var i=nodes.length-1;i>=0;i--){
    var n=nodes[i];
    if(typeof n.x!=="number")continue;
    var dx2=n.x-w.x,dy2=n.y-w.y;
    var d2=Math.sqrt(dx2*dx2+dy2*dy2);
    var hitR2=(n._r+4)/scale;
    if(d2<hitR2&&d2<bestD){best=n;bestD=d2}
  }
  return best;
}

function findEdgeAt(sx,sy){
  var w=screenToWorld(sx,sy);
  var threshold=6/scale;
  // Only check edges with at least one endpoint in viewport (Phase 3b)
  var vp=getViewportWorld();
  var margin=50/scale;
  for(var i=0;i<links.length;i++){
    var e=links[i];
    var ax=e.source.x,ay=e.source.y,bx=e.target.x,by=e.target.y;
    if(typeof ax!=="number")continue;
    // Skip if both endpoints off-screen
    var sIn=ax>=vp.x1-margin&&ax<=vp.x2+margin&&ay>=vp.y1-margin&&ay<=vp.y2+margin;
    var tIn=bx>=vp.x1-margin&&bx<=vp.x2+margin&&by>=vp.y1-margin&&by<=vp.y2+margin;
    if(!sIn&&!tIn)continue;
    var ddx=bx-ax,ddy=by-ay;
    var len2=ddx*ddx+ddy*ddy;
    if(len2===0)continue;
    var t=Math.max(0,Math.min(1,((w.x-ax)*ddx+(w.y-ay)*ddy)/len2));
    var px=ax+t*ddx,py=ay+t*ddy;
    var dist=Math.sqrt((w.x-px)*(w.x-px)+(w.y-py)*(w.y-py));
    if(dist<threshold)return e;
  }
  return null;
}

// ── Viewport helpers (Phase 3b) ──
function getViewportWorld(){
  return{
    x1:-tx/scale,
    y1:-ty/scale,
    x2:(W-tx)/scale,
    y2:(H-ty)/scale
  };
}

function isInViewport(wx,wy,margin,vp){
  return wx>=vp.x1-margin&&wx<=vp.x2+margin&&wy>=vp.y1-margin&&wy<=vp.y2+margin;
}

// ── Mouse ──
var isDragging=false,dragNode=null,isPanning=false;
var lastMx=0,lastMy=0;

canvas.addEventListener("mousedown",function(ev){
  var mx=ev.offsetX,my=ev.offsetY;
  lastMx=mx;lastMy=my;
  var hit=findNodeAt(mx,my);
  if(hit){
    dragNode=hit;isDragging=true;
    dragNode.fx=dragNode.x;dragNode.fy=dragNode.y;
    simulation.alphaTarget(0.3).restart();
    canvas.style.cursor="grabbing";
  }else{
    isPanning=true;
    canvas.style.cursor="grabbing";
  }
});

canvas.addEventListener("mousemove",function(ev){
  var mx=ev.offsetX,my=ev.offsetY;
  if(isDragging&&dragNode){
    var w=screenToWorld(mx,my);
    dragNode.fx=w.x;dragNode.fy=w.y;
  }else if(isPanning){
    tx+=mx-lastMx;ty+=my-lastMy;
    draw();
  }else{
    var hit=findNodeAt(mx,my);
    if(hit!==hoveredNode){
      hoveredNode=hit;
      if(hoveredNode){
        showNodeInfo(hoveredNode);
        canvas.style.cursor="pointer";
      }else{
        var edge=findEdgeAt(mx,my);
        if(edge){showEdgeInfo(edge);canvas.style.cursor="pointer"}
        else{infoPanel.classList.remove("visible");canvas.style.cursor="grab"}
      }
      draw();
    }
  }
  lastMx=mx;lastMy=my;
});

canvas.addEventListener("mouseup",function(){
  if(isDragging&&dragNode){
    dragNode.fx=null;dragNode.fy=null;
    simulation.alphaTarget(0);
  }
  isDragging=false;dragNode=null;isPanning=false;
  canvas.style.cursor="grab";
});

canvas.addEventListener("mouseleave",function(){
  if(isDragging&&dragNode){dragNode.fx=null;dragNode.fy=null;simulation.alphaTarget(0)}
  isDragging=false;dragNode=null;isPanning=false;
  hoveredNode=null;draw();
});

// ── Double-click to zoom to node (Phase 1) ──
canvas.addEventListener("dblclick",function(ev){
  ev.preventDefault();
  var mx=ev.offsetX,my=ev.offsetY;
  var hit=findNodeAt(mx,my);
  if(hit&&typeof hit.x==="number"){
    var ns=clamp(scale*2,0.02,30);
    var ntx=W/2-hit.x*ns;
    var nty=H/2-hit.y*ns;
    smoothPanZoomTo(ns,ntx,nty,300);
  }
});

// ── Wheel zoom (Phase 1: proportional to deltaY) ──
canvas.addEventListener("wheel",function(ev){
  ev.preventDefault();
  if(zoomAnim)zoomAnim=null; // cancel any running animation
  var mx=ev.offsetX,my=ev.offsetY;
  var factor=Math.exp(-ev.deltaY*0.002);
  var newScale=clamp(scale*factor,0.02,30);
  var ratio=newScale/scale;
  tx=mx-(mx-tx)*ratio;
  ty=my-(my-ty)*ratio;
  scale=newScale;
  draw();
},{passive:false});

// ── Zoom buttons (Phase 1) ──
document.getElementById("btn-zoom-in").addEventListener("click",function(){
  smoothZoomTo(scale*1.4,W/2,H/2,200);
});
document.getElementById("btn-zoom-out").addEventListener("click",function(){
  smoothZoomTo(scale/1.4,W/2,H/2,200);
});
document.getElementById("btn-fit").addEventListener("click",function(){
  zoomToFit(true);
});

// ── Density (Phase 2: Grid-based heatmap + type clouds) ──
var densityCellSize=30; // world-space cell size
var heatmapGrid=null; // {cols,rows,ox,oy,data:[]} global density
var typeGrids=null; // {type:{cols,rows,ox,oy,data:[],rgb:[]}} per-type density

function computeDensity(){
  if(!layers.heatmap&&!layers.typeclouds)return;
  // Compute world bounding box
  var minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  nodes.forEach(function(n){
    if(typeof n.x!=="number"||!isFinite(n.x))return;
    if(n.x<minX)minX=n.x;if(n.x>maxX)maxX=n.x;
    if(n.y<minY)minY=n.y;if(n.y>maxY)maxY=n.y;
  });
  if(minX===Infinity)return;

  var pad=densityCellSize*2;
  minX-=pad;minY-=pad;maxX+=pad;maxY+=pad;
  var cols=Math.ceil((maxX-minX)/densityCellSize);
  var rows=Math.ceil((maxY-minY)/densityCellSize);
  if(cols<1||rows<1||cols*rows>500000)return; // safety limit

  // Global heatmap
  if(layers.heatmap){
    var grid=new Float32Array(cols*rows);
    nodes.forEach(function(n){
      if(typeof n.x!=="number"||!isFinite(n.x))return;
      var ci=Math.floor((n.x-minX)/densityCellSize);
      var ri=Math.floor((n.y-minY)/densityCellSize);
      if(ci>=0&&ci<cols&&ri>=0&&ri<rows)grid[ri*cols+ci]++;
    });
    // Multi-pass 3x3 box blur (3 passes approximates Gaussian)
    var src=grid,dst;
    for(var pass=0;pass<3;pass++){
      dst=new Float32Array(cols*rows);
      for(var r=0;r<rows;r++){
        for(var c=0;c<cols;c++){
          var sum=0,cnt=0;
          for(var dr=-1;dr<=1;dr++){
            for(var dc=-1;dc<=1;dc++){
              var rr=r+dr,cc=c+dc;
              if(rr>=0&&rr<rows&&cc>=0&&cc<cols){sum+=src[rr*cols+cc];cnt++}
            }
          }
          dst[r*cols+c]=sum/cnt;
        }
      }
      src=dst;
    }
    var blurred=src;
    var maxVal=0;
    for(var i=0;i<blurred.length;i++){if(blurred[i]>maxVal)maxVal=blurred[i]}
    // Pre-render to offscreen canvas for smooth bilinear interpolation
    var hmCanvas=document.createElement("canvas");
    hmCanvas.width=cols;hmCanvas.height=rows;
    var hmCtx=hmCanvas.getContext("2d");
    var hmImg=hmCtx.createImageData(cols,rows);
    if(maxVal>0){
      for(var i=0;i<blurred.length;i++){
        var v=blurred[i];
        if(v>0){
          var a=Math.round((0.03+(v/maxVal)*0.17)*255);
          var idx=i*4;
          hmImg.data[idx]=255;hmImg.data[idx+1]=230;hmImg.data[idx+2]=200;hmImg.data[idx+3]=a;
        }
      }
    }
    hmCtx.putImageData(hmImg,0,0);
    heatmapGrid={cols:cols,rows:rows,ox:minX,oy:minY,data:blurred,maxVal:maxVal,canvas:hmCanvas};
  }

  // Per-type grids
  if(layers.typeclouds){
    var tg={};
    nodes.forEach(function(n){
      if(typeof n.x!=="number"||!isFinite(n.x))return;
      var t=n.type||"default";
      if(!tg[t]){tg[t]={grid:new Float32Array(cols*rows),rgb:n._rgb}}
      var ci=Math.floor((n.x-minX)/densityCellSize);
      var ri=Math.floor((n.y-minY)/densityCellSize);
      if(ci>=0&&ci<cols&&ri>=0&&ri<rows)tg[t].grid[ri*cols+ci]++;
    });
    // Multi-pass blur + offscreen canvas per type
    var result={};
    Object.keys(tg).forEach(function(t){
      var g=tg[t].grid;
      var rgb=tg[t].rgb;
      // 3-pass box blur
      var src2=g,dst2;
      for(var pass=0;pass<3;pass++){
        dst2=new Float32Array(cols*rows);
        for(var r=0;r<rows;r++){
          for(var c=0;c<cols;c++){
            var sum=0,cnt=0;
            for(var dr=-1;dr<=1;dr++){
              for(var dc=-1;dc<=1;dc++){
                var rr=r+dr,cc=c+dc;
                if(rr>=0&&rr<rows&&cc>=0&&cc<cols){sum+=src2[rr*cols+cc];cnt++}
              }
            }
            dst2[r*cols+c]=sum/cnt;
          }
        }
        src2=dst2;
      }
      var bl=src2;
      var mv=0;
      for(var i=0;i<bl.length;i++){if(bl[i]>mv)mv=bl[i]}
      // Pre-render to offscreen canvas
      var tcCanvas=document.createElement("canvas");
      tcCanvas.width=cols;tcCanvas.height=rows;
      var tcCtx=tcCanvas.getContext("2d");
      var tcImg=tcCtx.createImageData(cols,rows);
      if(mv>0){
        for(var i=0;i<bl.length;i++){
          var v=bl[i];
          if(v>0){
            var a=Math.round((0.05+(v/mv)*0.20)*255);
            var idx=i*4;
            tcImg.data[idx]=rgb[0];tcImg.data[idx+1]=rgb[1];tcImg.data[idx+2]=rgb[2];tcImg.data[idx+3]=a;
          }
        }
      }
      tcCtx.putImageData(tcImg,0,0);
      result[t]={cols:cols,rows:rows,ox:minX,oy:minY,data:bl,maxVal:mv,rgb:rgb,canvas:tcCanvas};
    });
    typeGrids=result;
  }
}

// ── FPS counter (Phase 4) ──
var fpsEl=document.getElementById("fps-counter");
var showFps=false;
var fpsFrames=0,fpsLast=performance.now(),fpsCurrent=0;

function updateFps(){
  fpsFrames++;
  var now=performance.now();
  if(now-fpsLast>=1000){
    fpsCurrent=Math.round(fpsFrames*1000/(now-fpsLast));
    fpsFrames=0;fpsLast=now;
    if(showFps)fpsEl.textContent=fpsCurrent+" fps | "+N+" nodes | "+M+" edges";
  }
}

// ── Drawing (Phase 3b-3e: viewport culling, batched edges, LOD, adaptive labels) ──
function draw(){
  updateFps();
  ctx.save();
  ctx.clearRect(0,0,W,H);

  // Background
  ctx.fillStyle="#000000";
  ctx.fillRect(0,0,W,H);

  // Subtle grid
  ctx.save();
  ctx.globalAlpha=0.03;
  ctx.strokeStyle="#fff";
  ctx.lineWidth=1;
  var gridSize=60*scale;
  if(gridSize>10){
    var ox=tx%gridSize,oy=ty%gridSize;
    ctx.beginPath();
    for(var gx=ox;gx<W;gx+=gridSize){ctx.moveTo(gx,0);ctx.lineTo(gx,H)}
    for(var gy=oy;gy<H;gy+=gridSize){ctx.moveTo(0,gy);ctx.lineTo(W,gy)}
    ctx.stroke();
  }
  ctx.restore();

  ctx.translate(tx,ty);
  ctx.scale(scale,scale);

  // Viewport in world coords
  var vp=getViewportWorld();
  var vpMargin=30/scale;

  // Build neighbor set for hover
  var neighborSet=null;
  if(hoveredNode){
    neighborSet=new Set();
    neighborSet.add(hoveredNode.id);
    if(adjMap[hoveredNode.id]){
      adjMap[hoveredNode.id].forEach(function(id){neighborSet.add(id)});
    }
  }

  // Search highlighting
  var hasSearch=searchQuery&&searchMatches.size>0;

  // ── Density heatmap layer (smooth offscreen canvas) ──
  if(layers.heatmap&&heatmapGrid&&heatmapGrid.maxVal>0&&heatmapGrid.canvas){
    ctx.save();
    ctx.imageSmoothingEnabled=true;
    ctx.imageSmoothingQuality="high";
    ctx.drawImage(heatmapGrid.canvas,heatmapGrid.ox,heatmapGrid.oy,
      heatmapGrid.cols*densityCellSize,heatmapGrid.rows*densityCellSize);
    ctx.restore();
  }

  // ── Type cloud layer (smooth offscreen canvas) ──
  if(layers.typeclouds&&typeGrids){
    ctx.save();
    ctx.imageSmoothingEnabled=true;
    ctx.imageSmoothingQuality="high";
    Object.keys(typeGrids).forEach(function(t){
      var tg=typeGrids[t];
      if(tg.maxVal<=0||!tg.canvas)return;
      ctx.drawImage(tg.canvas,tg.ox,tg.oy,
        tg.cols*densityCellSize,tg.rows*densityCellSize);
    });
    ctx.restore();
  }

  // ── Edges (Phase 3b-3c: viewport culling + batched rendering) ──
  if(layers.edges){
    // At extreme zoom-out with large graph, skip edges entirely
    var skipEdges=scale<0.1&&N>5000;
    if(!skipEdges){
      var skipArrows=scale<0.5||N>5000;

      // Classify edges into batches by visual state
      var normalEdges=[];
      var hoverEdges=[];
      var searchEdges=[];
      var arrowEdges=[];

      links.forEach(function(l){
        var sx2=l.source.x,sy2=l.source.y,tx2=l.target.x,ty2=l.target.y;
        if(typeof sx2!=="number")return;

        // Viewport culling: skip if both endpoints off-screen
        var sIn=sx2>=vp.x1-vpMargin&&sx2<=vp.x2+vpMargin&&sy2>=vp.y1-vpMargin&&sy2<=vp.y2+vpMargin;
        var tIn=tx2>=vp.x1-vpMargin&&tx2<=vp.x2+vpMargin&&ty2>=vp.y1-vpMargin&&ty2<=vp.y2+vpMargin;
        if(!sIn&&!tIn)return;

        var alpha=0.08;
        var lineW=0.5;
        var isHover=false,isSearch=false;

        if(hoveredNode){
          var sid=l.source.id||l.source;
          var tid=l.target.id||l.target;
          if(sid===hoveredNode.id||tid===hoveredNode.id){
            alpha=0.6;lineW=1.5;isHover=true;
          }else{alpha=0.02}
        }
        if(hasSearch){
          var sid3=l.source.id||l.source;
          var tid3=l.target.id||l.target;
          if(searchMatches.has(sid3)&&searchMatches.has(tid3)){
            alpha=Math.max(alpha,0.4);lineW=Math.max(lineW,1);isSearch=true;
          }else if(!hoveredNode){alpha=0.02}
        }
        if(l.weight){lineW=Math.max(lineW,Math.min(3,l.weight*2))}

        var entry={l:l,alpha:alpha,lineW:lineW};
        if(isHover){hoverEdges.push(entry);if(!skipArrows&&alpha>0.1)arrowEdges.push(entry)}
        else if(isSearch){searchEdges.push(entry);if(!skipArrows&&alpha>0.1)arrowEdges.push(entry)}
        else{normalEdges.push(entry)}
      });

      // Draw normal edges in a single batch
      if(normalEdges.length>0){
        var nAlpha=hoveredNode?0.02:hasSearch?0.02:0.08;
        ctx.beginPath();
        normalEdges.forEach(function(e){
          ctx.moveTo(e.l.source.x,e.l.source.y);
          ctx.lineTo(e.l.target.x,e.l.target.y);
        });
        ctx.strokeStyle="rgba(219,216,216,"+nAlpha+")";
        ctx.lineWidth=0.5/scale;
        ctx.stroke();
      }

      // Draw search-matched edges
      searchEdges.forEach(function(e){
        ctx.beginPath();
        ctx.moveTo(e.l.source.x,e.l.source.y);
        ctx.lineTo(e.l.target.x,e.l.target.y);
        ctx.strokeStyle="rgba(219,216,216,"+e.alpha+")";
        ctx.lineWidth=e.lineW/scale;
        ctx.stroke();
      });

      // Draw hover edges
      hoverEdges.forEach(function(e){
        ctx.beginPath();
        ctx.moveTo(e.l.source.x,e.l.source.y);
        ctx.lineTo(e.l.target.x,e.l.target.y);
        ctx.strokeStyle="rgba(219,216,216,"+e.alpha+")";
        ctx.lineWidth=e.lineW/scale;
        ctx.stroke();
      });

      // Draw arrows for highlighted edges
      if(!skipArrows){
        arrowEdges.forEach(function(e){
          var l=e.l;
          var sx3=l.source.x,sy3=l.source.y,tx3=l.target.x,ty3=l.target.y;
          var ddx=tx3-sx3,ddy=ty3-sy3;
          var len=Math.sqrt(ddx*ddx+ddy*ddy);
          if(len>0){
            var ux=ddx/len,uy=ddy/len;
            var tr=(l.target._r||6)+2;
            var ax=tx3-ux*tr,ay=ty3-uy*tr;
            var arrowSize=4/scale;
            ctx.beginPath();
            ctx.moveTo(ax,ay);
            ctx.lineTo(ax-ux*arrowSize+uy*arrowSize*0.5,ay-uy*arrowSize-ux*arrowSize*0.5);
            ctx.lineTo(ax-ux*arrowSize-uy*arrowSize*0.5,ay-uy*arrowSize+ux*arrowSize*0.5);
            ctx.closePath();
            ctx.fillStyle="rgba(219,216,216,"+e.alpha+")";
            ctx.fill();
          }
        });
      }
    }
  }

  // Edge labels on hover
  if(layers.labels&&hoveredNode&&scale>0.4){
    ctx.font=(10/scale)+"px Inter,system-ui,sans-serif";
    ctx.textAlign="center";
    ctx.textBaseline="middle";
    links.forEach(function(l){
      var sid=l.source.id||l.source;
      var tid=l.target.id||l.target;
      if(sid===hoveredNode.id||tid===hoveredNode.id){
        var mx2=(l.source.x+l.target.x)/2;
        var my2=(l.source.y+l.target.y)/2;
        ctx.fillStyle="rgba(219,216,216,0.7)";
        ctx.fillText(truncate(l.relation||"",30),mx2,my2-6/scale);
      }
    });
  }

  // ── Nodes (Phase 3d: LOD rendering) ──
  if(layers.nodes){
    var useDots=scale<0.2||N>10000;
    var skipGlow=N>5000;
    var skipBorder=N>2000;

    nodes.forEach(function(n){
      if(typeof n.x!=="number")return;
      // Viewport culling
      if(!isInViewport(n.x,n.y,n._r*3+vpMargin,vp))return;

      var r=n._r;
      var alpha=1;
      var glowAlpha=0;

      if(hoveredNode){
        if(neighborSet.has(n.id)){
          alpha=1;glowAlpha=n.id===hoveredNode.id?0.4:0.2;
        }else{
          alpha=0.15;
        }
      }
      if(hasSearch){
        if(searchMatches.has(n.id)){alpha=1;glowAlpha=0.3}
        else if(!hoveredNode){alpha=0.12}
      }

      if(useDots){
        // Simplified: small filled rectangle
        ctx.globalAlpha=alpha;
        ctx.fillStyle=n.color;
        var dotSize=Math.max(1,r*0.5);
        ctx.fillRect(n.x-dotSize,n.y-dotSize,dotSize*2,dotSize*2);
        ctx.globalAlpha=1;
        return;
      }

      // Glow (skip for large graphs)
      if(glowAlpha>0&&!skipGlow){
        var grd=ctx.createRadialGradient(n.x,n.y,r,n.x,n.y,r*3);
        var rgb=n._rgb;
        grd.addColorStop(0,"rgba("+rgb[0]+","+rgb[1]+","+rgb[2]+","+glowAlpha+")");
        grd.addColorStop(1,"rgba("+rgb[0]+","+rgb[1]+","+rgb[2]+",0)");
        ctx.fillStyle=grd;
        ctx.beginPath();
        ctx.arc(n.x,n.y,r*3,0,Math.PI*2);
        ctx.fill();
      }

      // Node circle
      ctx.globalAlpha=alpha;
      ctx.beginPath();
      ctx.arc(n.x,n.y,r,0,Math.PI*2);
      ctx.fillStyle=n.color;
      ctx.fill();

      // Subtle border (skip for medium+ graphs)
      if(!skipBorder){
        ctx.strokeStyle="rgba(244,244,244,0.2)";
        ctx.lineWidth=0.5;
        ctx.stroke();
      }
      ctx.globalAlpha=1;
    });
  }

  // ── Labels (Phase 3e: adaptive) ──
  if(layers.labels){
    var fontSize=Math.max(3,Math.min(12,10/scale));
    ctx.font="500 "+fontSize+"px Inter,system-ui,sans-serif";
    ctx.textAlign="center";
    ctx.textBaseline="middle";

    // Adaptive label strategy
    var disableLabels=N>10000&&scale<0.5;

    if(!disableLabels){
      var labelEvery;
      if(N>5000){labelEvery=Infinity} // only special nodes
      else if(N>2000){labelEvery=Math.ceil(N/20)}
      else if(N>500){labelEvery=scale<0.3?Math.ceil(N/20):scale<0.8?Math.ceil(N/80):1}
      else{labelEvery=1}

      nodes.forEach(function(n,i){
        if(typeof n.x!=="number")return;
        // Viewport culling for labels
        if(!isInViewport(n.x,n.y,n._r+fontSize*2+vpMargin,vp))return;

        var shouldLabel=false;

        // Always label hovered + neighbors
        if(hoveredNode&&neighborSet&&neighborSet.has(n.id))shouldLabel=true;
        // Always label search matches
        if(hasSearch&&searchMatches.has(n.id))shouldLabel=true;

        if(N>5000){
          // Only label top-degree or hovered/searched
          if(topDegreeSet.has(n.id))shouldLabel=true;
        }else{
          // High-degree nodes
          if(n._degree>=maxDeg*0.3)shouldLabel=true;
          // Sparse labeling
          if(labelEvery<Infinity&&i%labelEvery===0)shouldLabel=true;
        }

        if(!shouldLabel)return;

        var alpha2=0.85;
        if(hoveredNode){
          alpha2=neighborSet.has(n.id)?1:0.1;
        }
        if(hasSearch){
          alpha2=searchMatches.has(n.id)?1:0.08;
        }

        var label=truncate(n.name||"",30);
        var yOff=n._r+fontSize*0.8;

        // Text shadow
        ctx.fillStyle="rgba(0,0,0,"+alpha2*0.8+")";
        ctx.fillText(label,n.x+0.5,n.y+yOff+0.5);

        ctx.fillStyle="rgba(244,244,244,"+alpha2+")";
        ctx.fillText(label,n.x,n.y+yOff);
      });
    }else if(hoveredNode){
      // Even with labels disabled, label hovered node
      var hn=hoveredNode;
      var label2=truncate(hn.name||"",30);
      var yOff2=hn._r+fontSize*0.8;
      ctx.fillStyle="rgba(0,0,0,0.8)";
      ctx.fillText(label2,hn.x+0.5,hn.y+yOff2+0.5);
      ctx.fillStyle="rgba(244,244,244,1)";
      ctx.fillText(label2,hn.x,hn.y+yOff2);
    }
  }

  ctx.restore();

  // Update minimap
  if(minimapReady)drawMinimap();
}

// ── Density scheduling ──
var densityTimer=null;
function scheduleDensity(){
  if(densityTimer)clearTimeout(densityTimer);
  densityTimer=setTimeout(computeDensity,300);
}

// Recompute density when simulation settles
var densityTickCount=0;
simulation.on("tick.density",function(){
  densityTickCount++;
  if(densityTickCount%50===0)computeDensity();
});

// ── Minimap (Phase 4) ──
var minimapContainer=document.getElementById("minimap-container");
var minimapCanvas=document.getElementById("minimap-canvas");
var minimapCtx=minimapCanvas.getContext("2d");
var minimapW=150,minimapH=100;
var minimapReady=false;
var minimapBounds=null; // {minX,maxX,minY,maxY}
var minimapDragging=false;

function setupMinimap(){
  if(N<500){return} // hide minimap for small graphs
  minimapCanvas.width=minimapW*dpr;
  minimapCanvas.height=minimapH*dpr;
  minimapCanvas.style.width=minimapW+"px";
  minimapCanvas.style.height=minimapH+"px";
  minimapCtx.setTransform(dpr,0,0,dpr,0,0);
  minimapContainer.style.display="block";
  minimapReady=true;
  computeMinimapBounds();
  drawMinimap();
}

function computeMinimapBounds(){
  var minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
  nodes.forEach(function(n){
    if(typeof n.x!=="number"||!isFinite(n.x))return;
    if(n.x<minX)minX=n.x;if(n.x>maxX)maxX=n.x;
    if(n.y<minY)minY=n.y;if(n.y>maxY)maxY=n.y;
  });
  var pad=50;
  minimapBounds={minX:minX-pad,maxX:maxX+pad,minY:minY-pad,maxY:maxY+pad};
}

function drawMinimap(){
  if(!minimapReady||!minimapBounds)return;
  var mb=minimapBounds;
  var gw=mb.maxX-mb.minX||100;
  var gh=mb.maxY-mb.minY||100;
  var sx=minimapW/gw;
  var sy=minimapH/gh;
  var ms=Math.min(sx,sy);

  minimapCtx.clearRect(0,0,minimapW,minimapH);
  minimapCtx.fillStyle="rgba(0,0,0,0.6)";
  minimapCtx.fillRect(0,0,minimapW,minimapH);

  // Draw nodes as dots
  var dotR=Math.max(0.5,1);
  nodes.forEach(function(n){
    if(typeof n.x!=="number")return;
    var px=(n.x-mb.minX)*ms;
    var py=(n.y-mb.minY)*ms;
    minimapCtx.fillStyle=n.color;
    minimapCtx.fillRect(px-dotR,py-dotR,dotR*2,dotR*2);
  });

  // Draw viewport rectangle
  var vp=getViewportWorld();
  var vx1=(vp.x1-mb.minX)*ms;
  var vy1=(vp.y1-mb.minY)*ms;
  var vx2=(vp.x2-mb.minX)*ms;
  var vy2=(vp.y2-mb.minY)*ms;
  minimapCtx.strokeStyle="rgba(255,255,255,0.7)";
  minimapCtx.lineWidth=1;
  minimapCtx.strokeRect(vx1,vy1,vx2-vx1,vy2-vy1);
}

// Minimap click-to-navigate
minimapCanvas.addEventListener("mousedown",function(ev){
  ev.stopPropagation();
  minimapDragging=true;
  navigateMinimap(ev);
});
minimapCanvas.addEventListener("mousemove",function(ev){
  if(minimapDragging)navigateMinimap(ev);
});
minimapCanvas.addEventListener("mouseup",function(){minimapDragging=false});
minimapCanvas.addEventListener("mouseleave",function(){minimapDragging=false});

function navigateMinimap(ev){
  if(!minimapBounds)return;
  var rect=minimapCanvas.getBoundingClientRect();
  var mx=(ev.clientX-rect.left);
  var my=(ev.clientY-rect.top);
  var mb=minimapBounds;
  var gw=mb.maxX-mb.minX||100;
  var gh=mb.maxY-mb.minY||100;
  var ms=Math.min(minimapW/gw,minimapH/gh);
  var wx=mx/ms+mb.minX;
  var wy=my/ms+mb.minY;
  tx=W/2-wx*scale;
  ty=H/2-wy*scale;
  draw();
}

// Initial draw
draw();

// ── Keyboard shortcuts ──
document.addEventListener("keydown",function(ev){
  if(ev.target===searchInput)return;
  if(ev.key==="f"||ev.key==="/"){ev.preventDefault();searchInput.focus()}
  if(ev.key==="Escape"){searchInput.value="";searchQuery="";searchMatches.clear();searchInput.blur();draw()}
  if(ev.key==="0"){tx=W/2;ty=H/2;scale=1;draw()}
  // Phase 1: keyboard zoom
  if(ev.key==="="||ev.key==="+"){smoothZoomTo(scale*1.4,W/2,H/2,200)}
  if(ev.key==="-"||ev.key==="_"){smoothZoomTo(scale/1.4,W/2,H/2,200)}
  // Phase 4: FPS counter toggle
  if(ev.key==="d"||ev.key==="D"){
    showFps=!showFps;
    fpsEl.style.display=showFps?"block":"none";
    if(!showFps)fpsEl.textContent="";
  }
});

// Trigger density computation after initial layout
scheduleDensity();

})();
</script>
</body>
</html>"""
