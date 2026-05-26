(function(){
"use strict";

var nodes = __NODES_DATA__;
var links = __LINKS_DATA__;
var taskColors = __TASK_COLORS__;
var pipelineColors = __PIPELINE_COLORS__;
var nodesetColors = __NODESET_COLORS__;
var userColors = __USER_COLORS__;

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
// Node radius blends degree (base size from connections) with
// importance_weight (per-node hand-set or memify-set importance). Default
// importance is 0.5 → factor 1.0 (no change). High importance (1.0) →
// factor 1.5 (50% larger). Low importance (0.0) → factor 0.6 (smaller).
nodes.forEach(function(n){
  n._degree=degreeMap[n.id]||0;
  var imp=(typeof n.importance_weight==="number")?n.importance_weight:0.5;
  if(imp<0)imp=0;if(imp>1)imp=1;
  var impFactor=0.6+imp*0.9;
  n._r=(4+Math.sqrt(n._degree/maxDeg)*10)*impFactor;
});

// Pre-sort nodes by degree descending (for adaptive labels)
var nodesByDegree=nodes.slice().sort(function(a,b){return b._degree-a._degree});
var topDegreeSet=new Set();
nodesByDegree.slice(0,Math.min(10,N)).forEach(function(n){topDegreeSet.add(n.id)});

// ── Layout model ──
var layoutMode="story";
var rankColumns=[];
var rankColumnGap=260;
var organicClusters=[];

function boolValue(v){
  if(v===true)return true;
  if(v===false||v===undefined||v===null)return false;
  return String(v).toLowerCase()==="true";
}
function numberValue(v){
  var n=Number(v);
  return Number.isFinite(n)?n:null;
}
function stableHash(value){
  var text=String(value||"");
  var hash=0;
  for(var i=0;i<text.length;i++){hash=((hash<<5)-hash+text.charCodeAt(i))|0}
  return Math.abs(hash);
}

var hasMeaningfulTopologicalRank=nodes.some(function(n){
  var rank=numberValue(n.topological_rank);
  return rank!==null&&rank!==0;
});
var maxGlobalContextLevel=0;
nodes.forEach(function(n){
  if(n.type==="GlobalContextSummary"&&!boolValue(n.is_root)){
    var level=numberValue(n.level);
    if(level!==null)maxGlobalContextLevel=Math.max(maxGlobalContextLevel,level);
  }
});

function inferLayoutCategory(n){
  if(hasMeaningfulTopologicalRank){
    var rank=numberValue(n.topological_rank);
    return "Rank "+(rank===null?0:rank);
  }
  if(n.type==="TextDocument")return "Document";
  if(n.type==="DocumentChunk")return "Chunk";
  if(n.type==="TextSummary")return "Summary";
  if(n.type==="GlobalContextSummary"){
    if(boolValue(n.is_root))return "Context Root";
    var level=numberValue(n.level);
    return "Context L"+(level===null?0:level);
  }
  if(n.type==="Entity")return "Entity";
  if(n.type==="EntityType")return "Type";
  if(n.type==="DatabaseSchema")return "Database";
  if(n.type==="SchemaTable"||n.type==="TableType")return "Table";
  if(n.type==="SchemaRelationship")return "Schema Rel";
  if(n.type==="TableRow")return "Row";
  if(n.type==="ColumnValue")return "Column";
  return n.type||"Node";
}

function inferLayoutRank(n){
  if(hasMeaningfulTopologicalRank){
    var rank=numberValue(n.topological_rank);
    if(rank!==null)return rank;
  }
  if(n.type==="TextDocument")return 0;
  if(n.type==="DocumentChunk")return 1;
  if(n.type==="TextSummary")return 2;
  if(n.type==="GlobalContextSummary"){
    if(boolValue(n.is_root))return 4+maxGlobalContextLevel;
    var level=numberValue(n.level);
    return 3+(level===null?0:level);
  }
  if(n.type==="Entity")return 3;
  if(n.type==="EntityType")return 4;
  if(n.type==="DatabaseSchema")return 0;
  if(n.type==="SchemaTable"||n.type==="TableType")return 1;
  if(n.type==="SchemaRelationship")return 2;
  if(n.type==="TableRow")return 2;
  if(n.type==="ColumnValue")return 3;
  if(n.source_task&&String(n.source_task).indexOf("extract_chunks")>=0)return 1;
  if(n.source_task&&String(n.source_task).indexOf("summar")>=0)return 2;
  if(n.source_task&&String(n.source_task).indexOf("graph")>=0)return 3;
  return 3;
}

function layoutLaneKey(n){
  if(n.source_content_hash)return n.source_content_hash;
  if(n.source_node_set)return n.source_node_set;
  if(n.type==="GlobalContextSummary")return n.type+":"+(n.level||0)+":"+(n.id||n.name);
  if(n.type==="EntityType")return n.name||n.id;
  return (n.type||"Node")+":"+(n.name||n.id);
}

function labelForRank(rankStats){
  var entries=Object.keys(rankStats).sort(function(a,b){
    return rankStats[b]-rankStats[a]||a.localeCompare(b);
  });
  if(entries.length===0)return "Nodes";
  if(entries.length>1&&rankStats[entries[1]]/rankStats[entries[0]]>=0.25){
    return entries[0]+" / "+entries[1];
  }
  return entries[0];
}

// ── Pipeline (Story) layout — bins nodes by preprocessor.stage ──
// Each stage becomes one vertical column laid left-to-right in the
// canonical order set by preprocessor.STAGE_ORDER. Within each column
// nodes are sorted by (visual_rank, importance desc). Falls back to
// computeRankedLayout() for graphs without preprocessor stage info.
var STORY_STAGE_ORDER=["document","chunk","entity","type","summary","context","schema","other"];
var STORY_STAGE_LABEL={
  document:"Documents",
  chunk:"Chunks",
  entity:"Entities",
  type:"Types",
  summary:"Summaries",
  context:"Context",
  schema:"Schema",
  other:"Other",
};
function computePipelineLayout(){
  var stageCounts={};
  nodes.forEach(function(n){
    var s=n.stage||"other";
    stageCounts[s]=(stageCounts[s]||0)+1;
  });
  var presentStages=STORY_STAGE_ORDER.filter(function(s){return stageCounts[s]>0});
  if(presentStages.length<2){
    computeRankedLayout();
    return;
  }
  var gapBase=N>2000?190:N>500?220:270;
  rankColumnGap=gapBase;
  rankColumns=presentStages.map(function(stage,index){
    return {
      rank: index,
      index: index,
      x: (index-(presentStages.length-1)/2)*rankColumnGap,
      label: STORY_STAGE_LABEL[stage]+" ("+stageCounts[stage]+")",
      stage: stage,
    };
  });
  var stageIndexMap={};
  presentStages.forEach(function(stage,index){stageIndexMap[stage]=index});

  var byStage={};
  presentStages.forEach(function(s){byStage[s]=[]});
  nodes.forEach(function(n){
    var s=n.stage||"other";
    if(!byStage[s])byStage[s]=[];
    byStage[s].push(n);
  });
  presentStages.forEach(function(stage){
    byStage[stage].sort(function(a,b){
      var ra=numberValue(a.visual_rank)||0;
      var rb=numberValue(b.visual_rank)||0;
      if(ra!==rb)return ra-rb;
      return (b.importance||0)-(a.importance||0);
    });
  });

  // Dense stages (e.g. 200+ entities on the Alice book) get packed into
  // multiple sub-columns within their stage's allotted horizontal slot,
  // so the vertical span stays readable and auto-fit doesn't crush
  // everything to 0.05x zoom. Sub-column gap is kept tight (≤ 28px) so
  // each stage still reads as ONE cluster, not as multiple stages.
  //
  // Lane spacing has to clear the LARGEST possible importance-scaled
  // node diameter so high-importance nodes (impFactor up to 1.5 → radius
  // up to ~21px → diameter ~42px) don't overlap their neighbors. We add
  // a 12px breathing margin on top of that floor.
  var laneSpacing=N>1000?32:N>300?40:N>120?52:64;
  var MAX_ROWS_PER_SUBCOL=36;
  presentStages.forEach(function(stage){
    var colIdx=stageIndexMap[stage];
    var column=rankColumns[colIdx];
    var colNodes=byStage[stage];
    var n=colNodes.length;
    var subCols=Math.max(1,Math.ceil(n/MAX_ROWS_PER_SUBCOL));
    var rowsPerSub=Math.ceil(n/subCols);
    // Sub-column gap deliberately tight (max 28px) so the cluster reads
    // as a single stage, not as multiple stages. Constrain total cluster
    // width to ~55% of the stage slot — leaves clear gutters either side.
    var maxClusterW=rankColumnGap*0.55;
    var subGap=subCols>1?Math.min(28,maxClusterW/(subCols-1)):0;
    var totalHeight=(rowsPerSub-1)*laneSpacing;
    var startY=-totalHeight/2;
    var startSubX=column.x-(subCols-1)*subGap/2;
    colNodes.forEach(function(node,i){
      var sub=Math.floor(i/rowsPerSub);
      var row=i%rowsPerSub;
      node._layoutRank=colIdx;
      node._layoutCategory=STORY_STAGE_LABEL[stage];
      node._targetX=startSubX+sub*subGap;
      node._targetY=startY+row*laneSpacing;
      if(typeof node.x!=="number"||!isFinite(node.x)){
        var jitter=(stableHash(node.id)%100)/100-0.5;
        node.x=node._targetX+jitter*Math.max(subGap,12)*0.3;
        node.y=node._targetY+jitter*laneSpacing;
      }
      // Pin BOTH x and y in Story mode — the layout is meant to be a
      // clean deterministic grid (one row per node within each stage
      // column). Letting y float lets link forces pull connected nodes
      // toward each other, which collapsed nodes onto neighbors. Fully
      // pinned + sufficient laneSpacing = no overlap. Cleared by
      // applyLayoutMode on switch to Flow / Force / Organic layouts.
      node.fx=node._targetX;
      node.fy=node._targetY;
    });
  });
}

function computeRankedLayout(){
  var rankStats={};
  nodes.forEach(function(n){
    var rank=inferLayoutRank(n);
    var category=inferLayoutCategory(n);
    n._layoutRank=rank;
    n._layoutCategory=category;
    if(!rankStats[rank])rankStats[rank]={};
    rankStats[rank][category]=(rankStats[rank][category]||0)+1;
  });

  var ranks=Object.keys(rankStats).map(Number).sort(function(a,b){return a-b});
  var gapBase=N>2000?190:N>500?220:270;
  rankColumnGap=gapBase;
  rankColumns=ranks.map(function(rank,index){
    return {
      rank: rank,
      index: index,
      x: (index-(ranks.length-1)/2)*rankColumnGap,
      label: labelForRank(rankStats[rank]),
    };
  });

  var rankIndexByValue={};
  rankColumns.forEach(function(column,index){rankIndexByValue[column.rank]=index});
  var laneSpacing=N>1000?34:N>300?44:58;
  nodes.forEach(function(n){
    var column=rankColumns[rankIndexByValue[n._layoutRank]];
    var hash=stableHash(layoutLaneKey(n));
    var laneCount=N>1500?17:N>500?13:9;
    var lane=hash%laneCount-Math.floor(laneCount/2);
    var jitter=(stableHash(n.id)%100)/100-0.5;
    n._targetX=column?column.x:0;
    n._targetY=lane*laneSpacing+jitter*laneSpacing*0.35;
    if(typeof n.x!=="number"||!isFinite(n.x)){
      n.x=n._targetX+jitter*rankColumnGap*0.25;
      n.y=n._targetY+jitter*laneSpacing;
    }
  });
}

function computeOrganicLayout(){
  var counts={};
  nodes.forEach(function(n){
    var type=n.type||"Node";
    counts[type]=(counts[type]||0)+1;
  });

  var types=Object.keys(counts).sort(function(a,b){return counts[b]-counts[a]||a.localeCompare(b)});
  var radius=clamp(Math.sqrt(N)*20,180,520);
  organicClusters=types.map(function(type,index){
    var angle=types.length===1?0:-Math.PI/2+index*Math.PI*2/types.length;
    var clusterRadius=clamp(64+Math.sqrt(counts[type])*9,96,240);
    return {
      type:type,
      count:counts[type],
      x:types.length===1?0:Math.cos(angle)*radius,
      y:types.length===1?0:Math.sin(angle)*radius,
      radius:clusterRadius,
    };
  });

  var clusterByType={};
  organicClusters.forEach(function(cluster){clusterByType[cluster.type]=cluster});
  nodes.forEach(function(n){
    var cluster=clusterByType[n.type||"Node"]||{x:0,y:0,radius:120};
    var angleHash=stableHash((n.id||n.name||"")+":angle")%6283/1000;
    var distHash=(stableHash((n.id||n.name||"")+":distance")%1000)/1000;
    var localRadius=Math.sqrt(distHash)*cluster.radius*0.62;
    n._organicX=cluster.x+Math.cos(angleHash)*localRadius;
    n._organicY=cluster.y+Math.sin(angleHash)*localRadius;
  });
}
if(layoutMode==="story")computePipelineLayout();else computeRankedLayout();

// ── Color palette ──
var colorByType={
  "Entity":"#6510F4","EntityType":"#A550FF","DocumentChunk":"#0DFF00",
  "TextSummary":"#6510F4","GlobalContextSummary":"#00C2FF",
  "TableRow":"#A550FF","TableType":"#6510F4",
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
  if(n.ontology_valid===true){
    n.color="#D8D8D8";
    n._rgb=[216,216,216];
  }
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
    }else if(colorByMode==="nodeset"){
      n.color=nodesetColors[n.source_node_set||"Unknown"]||"#DBD8D8";
    }else if(colorByMode==="user"){
      n.color=userColors[n.source_user||"Unknown"]||"#DBD8D8";
    }
    if(colorByMode==="type"&&n.ontology_valid===true){
      n.color="#D8D8D8";
      n._rgb=[216,216,216];
    }else{
      // Recache RGB
      if(n.color.indexOf("hsl")===0){
        var m=n.color.match(/[\d.]+/g);
        n._rgb=hslToRgb(+m[0],+m[1],+m[2]);
      }else{
        n._rgb=hexToRgb(n.color);
      }
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
var nodesetCounts={};
nodes.forEach(function(n){var s=n.source_node_set||"Unknown";nodesetCounts[s]=(nodesetCounts[s]||0)+1});
var userCounts={};
nodes.forEach(function(n){var u=n.source_user||"Unknown";userCounts[u]=(userCounts[u]||0)+1});
var uniqueTasks=Object.keys(taskCounts).length;
var uniquePipelines=Object.keys(pipeCounts).length;
var uniqueNodesets=Object.keys(nodesetCounts).length;
var uniqueUsers=Object.keys(userCounts).length;
var statsEl=document.getElementById("stats");
var perfTier=N>10000?"large":N>2000?"medium":"small";
statsEl.innerHTML='<span><span class="dot" style="background:#6510F4"></span>'+N.toLocaleString()+" nodes</span>"+
  '<span><span class="dot" style="background:#747470"></span>'+M.toLocaleString()+" edges</span>"+
  '<span>'+Object.keys(typeCounts).length+" types</span>"+
  '<span>'+uniqueTasks+" tasks</span>"+
  '<span>'+uniquePipelines+" pipelines</span>"+
  '<span>'+uniqueNodesets+" node sets</span>"+
  '<span>'+uniqueUsers+" users</span>"+
  '<span style="opacity:0.5">['+perfTier+']</span>';

// ── Legend ──
var legendEl=document.getElementById("legend");

function updateLegend(){
  var entries,counts,colorSource,sectionLabel;
  if(colorByMode==="type"){
    counts=typeCounts;
    entries=Object.keys(counts).sort(function(a,b){return counts[b]-counts[a]});
    colorSource=function(t){return colorByType[t]||typeColors[t]||"#DBD8D8"};
    sectionLabel="Node type";
  }else if(colorByMode==="task"){
    counts=taskCounts;
    entries=Object.keys(counts).sort(function(a,b){return counts[b]-counts[a]});
    colorSource=function(t){return taskColors[t]||"#DBD8D8"};
    sectionLabel="Task";
  }else if(colorByMode==="pipeline"){
    counts=pipeCounts;
    entries=Object.keys(counts).sort(function(a,b){return counts[b]-counts[a]});
    colorSource=function(t){return pipelineColors[t]||"#DBD8D8"};
    sectionLabel="Pipeline";
  }else if(colorByMode==="nodeset"){
    counts=nodesetCounts;
    entries=Object.keys(counts).sort(function(a,b){return counts[b]-counts[a]});
    colorSource=function(t){return nodesetColors[t]||"#DBD8D8"};
    sectionLabel="Node set";
  }else{
    counts=userCounts;
    entries=Object.keys(counts).sort(function(a,b){return counts[b]-counts[a]});
    colorSource=function(t){return userColors[t]||"#DBD8D8"};
    sectionLabel="User";
  }
  if(entries.length>8)entries=entries.slice(0,8);
  var typeItems=entries.map(function(t){
    return '<span class="legend-item"><span class="legend-dot" style="background:'+
      colorSource(t)+'"></span>'+esc(t)+'</span>';
  }).join("");
  // Ontology and importance sections — explain the encodings we layer
  // on top of the color (green ring = ontology_valid, radius = importance).
  // Always shown so users learn what the cues mean even on graphs
  // that don't currently use them.
  var ontologySection=
    '<div class="legend-section">'+
      '<span class="legend-title">Ontology</span>'+
      '<span class="legend-item"><span class="legend-ring"></span>matched in OWL</span>'+
    '</div>';
  var importanceSection=
    '<div class="legend-section">'+
      '<span class="legend-title">Importance</span>'+
      '<span class="legend-sizes">'+
        '<span class="legend-dot" style="width:5px;height:5px"></span>'+
        '<span class="legend-dot" style="width:8px;height:8px"></span>'+
        '<span class="legend-dot" style="width:12px;height:12px"></span>'+
      '</span>'+
      '<span style="opacity:0.7">low → high</span>'+
    '</div>';
  legendEl.innerHTML=
    '<div class="legend-section">'+
      '<span class="legend-title">'+esc(sectionLabel)+'</span>'+
      '<span class="legend-items">'+typeItems+'</span>'+
    '</div>'+
    '<span class="legend-sep"></span>'+
    ontologySection+
    '<span class="legend-sep"></span>'+
    importanceSection;
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
    if((key==="heatmap"||key==="typeclouds")&&layers[key]){
      computeDensity();
    }
    draw();
  });
});

// ── Label budget (Phase 1b/visible): off | key (default) | all ──
// Key uses node.label_priority from the Python preprocessor — documents
// and entity-types are always-on landmarks; other nodes earn a label if
// their importance is in the top quartile.
var labelBudget="key";
document.querySelectorAll(".ctrl-btn[data-labelbudget]").forEach(function(btn){
  btn.addEventListener("click",function(){
    labelBudget=btn.dataset.labelbudget;
    document.querySelectorAll(".ctrl-btn[data-labelbudget]").forEach(function(b){
      b.classList.toggle("active",b===btn);
    });
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
  var pad=0.06;
  var gw=maxX-minX||100;
  var gh=maxY-minY||100;
  var cx=(minX+maxX)/2;
  var cy=(minY+maxY)/2;
  var sx=W*(1-2*pad)/gw;
  var sy=H*(1-2*pad)/gh;
  // Hard floor so a sprawling graph still renders nodes at a readable
  // size — user can pan to see anything outside the viewport.
  var MIN_FIT_SCALE=0.55;
  var ns=Math.max(MIN_FIT_SCALE,Math.min(sx,sy,10));
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
  .force("x",d3.forceX(function(d){return d._targetX||0}).strength(0.22))
  .force("y",d3.forceY(function(d){return d._targetY||0}).strength(0.08))
  .alphaDecay(alphaDecayVal)
  .velocityDecay(velDecayVal);

if(collideIter>0){
  simulation.force("collide",d3.forceCollide().radius(function(d){return d._r+2}).iterations(collideIter));
}

function applyLayoutMode(restart){
  if(layoutMode==="story")computePipelineLayout();
  else computeRankedLayout();
  // story mode reuses the column-layout positioning machinery; both
  // "ranked" (by topological_rank) and "story" (by preprocessor.stage)
  // produce rankColumns and per-node _targetX/_targetY.
  if((layoutMode==="ranked"||layoutMode==="story")&&rankColumns.length>1){
    // Pin both axes in Story mode (computePipelineLayout already did
    // this but applyLayoutMode may be invoked after a flow-mode unpin,
    // so re-pin here defensively). Other layouts must unpin.
    if(layoutMode==="story"){
      nodes.forEach(function(n){
        if(typeof n._targetX==="number")n.fx=n._targetX;
        if(typeof n._targetY==="number")n.fy=n._targetY;
      });
    }else{
      nodes.forEach(function(n){n.fx=null;n.fy=null});
    }
    simulation.force("link").distance(linkDist).strength(0.2);
    simulation.force("charge").strength(repStr);
    simulation.force("x",d3.forceX(function(d){return d._targetX||0}).strength(0.22));
    simulation.force("y",d3.forceY(function(d){return d._targetY||0}).strength(0.08));
  }else{
    // Organic / Force mode — clear any pinned coords from a previous
    // Story-mode run so the simulation can move nodes anywhere.
    nodes.forEach(function(n){n.fx=null;n.fy=null});
    // Force-mode semantic tuning (Phase 1 polish, Codex critique). Three
    // signals layered on top of the legacy organic layout:
    //  - Per-link strength: structural relations are LOOSER (entities can
    //    drift away from their chunk-containment skeleton) so semantic
    //    edges aren't drowned out. Semantic links pull TIGHTER, surfacing
    //    real connections.
    //  - Per-node charge: hubs (high-degree nodes) repel more strongly
    //    proportional to log(degree), spreading them out so they stop
    //    occluding their neighbors.
    //  - Per-link distance: structural links allowed to stretch further.
    computeOrganicLayout();
    simulation.force("link")
      .distance(function(l){return l.edge_class==="semantic"?linkDist*0.9:linkDist*1.5})
      .strength(function(l){return l.edge_class==="semantic"?0.32:0.06});
    simulation.force("charge").strength(function(d){
      var deg=d._degree||0;
      return repStr*(1+Math.log(1+deg)*0.45);
    });
    simulation.force("x",d3.forceX(function(d){return d._organicX||0}).strength(0.075));
    simulation.force("y",d3.forceY(function(d){return d._organicY||0}).strength(0.075));
  }
  if(restart){
    simDone=false;
    simTickCount=0;
    simulation.alpha(0.8).restart();
    draw();
  }
}

document.querySelectorAll(".ctrl-btn[data-layout]").forEach(function(btn){
  btn.addEventListener("click",function(){
    document.querySelectorAll(".ctrl-btn[data-layout]").forEach(function(b){b.classList.remove("active")});
    btn.classList.add("active");
    layoutMode=btn.dataset.layout;
    applyLayoutMode(true);
  });
});

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

// ── Info panel (Phase 1e: sectioned inspector reading preprocessor enrichment) ──
var infoPanel=document.getElementById("info-panel");
var nodesById={};
nodes.forEach(function(n2){nodesById[n2.id]=n2});

function inspectorKv(key,value){
  if(value===undefined||value===null||value==="")return "";
  return '<div class="panel-row"><span class="k">'+esc(key)+'</span><span class="v">'+esc(truncate(String(value),80))+"</span></div>";
}

function inspectorSection(id,title,openByDefault,body){
  if(!body)return "";
  var openClass=openByDefault?" open":"";
  var caret=openByDefault?"&#x25BE;":"&#x25B8;"; // ▾ / ▸
  return '<div class="panel-section inspector-section'+openClass+'" data-section="'+id+'">'+
    '<div class="panel-section-title" style="cursor:pointer;user-select:none" data-toggle="'+id+'">'+
    '<span class="inspector-caret" style="display:inline-block;width:12px">'+caret+'</span>'+esc(title)+
    '</div>'+
    '<div class="inspector-section-body"'+(openByDefault?"":' style="display:none"')+'>'+body+'</div>'+
  '</div>';
}

// Plain-English one-liner — explains WHAT this node is in pipeline terms,
// not just what type it is. Reads preprocessor.stage and walks links for context.
function inspectorOverviewLine(n){
  var stage=n.stage||"other";
  var label=truncate(n.name||n.id,30);
  // Find structural parents (edges INTO this node with structural relation)
  var parents=[];
  var children=0;
  for(var li=0;li<links.length;li++){
    var l=links[li];
    var sid=l.source.id||l.source;
    var tid=l.target.id||l.target;
    if(tid===n.id&&l.edge_class==="structural")parents.push({id:sid,relation:l.relation});
    if(sid===n.id&&l.edge_class==="structural")children++;
  }
  if(stage==="document"){
    return "Top-level document. Cognee extracted "+children+" child node(s) from it.";
  }
  if(stage==="chunk"){
    var doc=null;
    for(var pi=0;pi<parents.length;pi++){
      var p=nodesById[parents[pi].id];
      if(p&&p.stage==="document"){doc=p;break}
    }
    return doc
      ? "Chunk extracted from document “"+truncate(doc.name||doc.id,40)+"”."
      : "Chunk of text extracted by the chunking task.";
  }
  if(stage==="entity"){
    var fromChunk=null;
    var entityType=null;
    for(var pi2=0;pi2<parents.length;pi2++){
      var p2=nodesById[parents[pi2].id];
      if(p2&&p2.stage==="chunk")fromChunk=p2;
    }
    // Find type via outgoing is_a edge
    for(var li2=0;li2<links.length;li2++){
      var l2=links[li2];
      var sid2=l2.source.id||l2.source;
      if(sid2===n.id&&(l2.relation==="is_a"||(l2.edge_info&&l2.edge_info.relationship_name==="is_a"))){
        var t2=nodesById[l2.target.id||l2.target];
        if(t2){entityType=t2.name||t2.id;break}
      }
    }
    var parts=["Entity “"+label+"”"];
    if(entityType)parts[0]+=" of type "+entityType;
    if(fromChunk)parts.push("extracted from a chunk");
    return parts.join(", ")+".";
  }
  if(stage==="type"){
    return "Entity type. Groups "+children+" entit"+(children===1?"y":"ies")+".";
  }
  if(stage==="summary"){
    return "Summary built by the summarization task.";
  }
  if(stage==="context"){
    var lvl=n.level;
    var isRoot=n.is_root;
    if(isRoot)return "Root of the global context hierarchy.";
    if(lvl!==undefined&&lvl!==null)return "Global context bucket at level "+lvl+".";
    return "Global context bucket.";
  }
  if(stage==="schema"){
    return "Schema element ("+esc(n.type||"")+").";
  }
  return null;
}

function showNodeInfo(n){
  if(!n){infoPanel.classList.remove("visible");return}

  // Header
  var html='<div class="panel-header">';
  html+='<span class="panel-type" style="background:'+n.color+'22;color:'+n.color+'">'+esc(n.type||"Node")+"</span>";
  html+='<span class="panel-title">'+esc(truncate(n.name||n.id,40))+"</span></div>";

  // Plain-English overview
  var overview=inspectorOverviewLine(n);
  if(overview){
    html+='<div class="inspector-overview" style="font-size:12px;color:var(--text);line-height:1.5;margin-top:6px;padding-bottom:8px">'+overview+"</div>";
  }

  // Description text (if the node carries one)
  var desc=null;
  var descKeys=["description","summary","text","content"];
  for(var di=0;di<descKeys.length;di++){
    var v=n[descKeys[di]]||(n.properties&&n.properties[descKeys[di]]);
    if(typeof v==="string"&&v.trim()){desc=v.trim();break}
  }
  if(desc){
    html+='<div style="font-size:12px;color:var(--text2);line-height:1.5;margin-top:4px;font-style:italic">"'+esc(truncate(desc,300))+'"</div>';
  }

  // Source section: chain to originating document/chunk
  var sourceBody="";
  var seenSource=false;
  var pid=n.id;
  for(var depth=0;depth<3;depth++){
    var parent=null;
    for(var pli=0;pli<links.length;pli++){
      var pl=links[pli];
      var psid=pl.source.id||pl.source;
      var ptid=pl.target.id||pl.target;
      if(ptid===pid&&pl.edge_class==="structural"){
        parent=nodesById[psid];
        if(parent){break}
      }
    }
    if(!parent)break;
    sourceBody+=inspectorKv(parent.stage||parent.type||"parent",parent.name||parent.id);
    seenSource=true;
    pid=parent.id;
  }
  if(seenSource){
    html+=inspectorSection("source","Source",true,sourceBody);
  }

  // Provenance section (only when at least one field is set)
  var provBody=
    inspectorKv("Task",n.source_task)+
    inspectorKv("Pipeline",n.source_pipeline)+
    inspectorKv("Node Set",n.source_node_set)+
    inspectorKv("User",n.source_user)+
    (n.topological_rank&&n.topological_rank>0
      ? inspectorKv("Pipeline rank",n.topological_rank)
      : "");
  if(provBody){
    html+=inspectorSection("provenance","Provenance",true,provBody);
  }

  // Weights section: importance + feedback + ontology validity, plus a
  // tiny inline bar so the values are scannable. Only shown when any
  // weight or the ontology flag is set to something interesting
  // (non-default-0.5 weight, or ontology_valid set).
  function weightBar(value){
    var pct=Math.max(0,Math.min(1,(typeof value==="number")?value:0.5));
    var w=Math.round(pct*60);
    return '<div style="display:inline-block;width:60px;height:5px;background:var(--border);'+
      'border-radius:3px;vertical-align:middle;margin-right:6px;position:relative;">'+
      '<div style="width:'+w+'px;height:5px;background:var(--accent);border-radius:3px;"></div></div>';
  }
  var hasImp=(typeof n.importance_weight==="number")&&Math.abs(n.importance_weight-0.5)>0.001;
  var hasFb=(typeof n.feedback_weight==="number")&&Math.abs(n.feedback_weight-0.5)>0.001;
  var hasOnto=n.ontology_valid===true||n.ontology_valid===false;
  if(hasImp||hasFb||hasOnto){
    var wBody="";
    if(hasImp){
      wBody+='<div class="panel-row"><span class="k">Importance</span><span class="v">'+
        weightBar(n.importance_weight)+n.importance_weight.toFixed(2)+'</span></div>';
    }
    if(hasFb){
      wBody+='<div class="panel-row"><span class="k">Feedback</span><span class="v">'+
        weightBar(n.feedback_weight)+n.feedback_weight.toFixed(2)+'</span></div>';
    }
    if(hasOnto){
      var ovColor=n.ontology_valid?'#0DFF00':'var(--text2)';
      var ovLabel=n.ontology_valid?'ontology valid':'not in ontology';
      wBody+='<div class="panel-row"><span class="k">Ontology</span>'+
        '<span class="v" style="color:'+ovColor+';font-weight:500;">●  '+ovLabel+'</span></div>';
    }
    html+=inspectorSection("weights","Weights",true,wBody);
  }

  // Relations section: counts grouped by relation
  var relCounts={};
  for(var rli=0;rli<links.length;rli++){
    var rl=links[rli];
    var rsid=rl.source.id||rl.source;
    var rtid=rl.target.id||rl.target;
    if(rsid===n.id||rtid===n.id){
      var rel=rl.relation||"related";
      relCounts[rel]=(relCounts[rel]||0)+1;
    }
  }
  var relBody=inspectorKv("Connections",n._degree);
  Object.keys(relCounts).sort().forEach(function(r){
    relBody+=inspectorKv(r,relCounts[r]);
  });
  html+=inspectorSection("relations","Relations",true,relBody);

  // Raw section: collapsed by default
  var rawBody=
    inspectorKv("ID",truncate(n.id,36))+
    inspectorKv("Stage",n.stage)+
    inspectorKv("Visual rank",n.visual_rank);
  if(n.ontology_valid!==undefined&&n.ontology_valid!==null){
    rawBody+=inspectorKv("Ontology valid",String(n.ontology_valid));
  }
  if(n.properties){
    Object.keys(n.properties).slice(0,10).forEach(function(key){
      var pv=n.properties[key];
      if(pv!==undefined&&pv!==null&&typeof pv!=="object"){
        rawBody+=inspectorKv(key,pv);
      }
    });
  }
  html+=inspectorSection("raw","Raw",false,rawBody);

  infoPanel.innerHTML=html;
  infoPanel.classList.add("visible");

  // Wire section toggles
  infoPanel.querySelectorAll("[data-toggle]").forEach(function(tHeader){
    tHeader.addEventListener("click",function(){
      var sec=tHeader.parentElement;
      var body=sec.querySelector(".inspector-section-body");
      var caret=tHeader.querySelector(".inspector-caret");
      var isOpen=sec.classList.toggle("open");
      if(body)body.style.display=isOpen?"":"none";
      if(caret)caret.innerHTML=isOpen?"&#x25BE;":"&#x25B8;";
    });
  });
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

// Top-chrome reserved zone in screen-pixels. Tabs (32px), header (~40px),
// theme toggle, search box all live in the first ~64px of the viewport.
// Column gridlines and column labels must start below this so they don't
// bleed into the chrome.
var TOP_CHROME_PAD=72;
// Bottom-chrome zone for the floating controls bar.
var BOT_CHROME_PAD=64;
function drawRankColumns(vp,_light){
  if((layoutMode!=="ranked"&&layoutMode!=="story")||rankColumns.length<2)return;
  ctx.save();
  var topY=vp.y1+TOP_CHROME_PAD/scale;
  var botY=vp.y2-BOT_CHROME_PAD/scale;
  if(botY<=topY)botY=vp.y2; // nothing to clip if zoom is extreme
  rankColumns.forEach(function(column,index){
    var half=rankColumnGap/2;
    if(column.x+half<vp.x1||column.x-half>vp.x2)return;

    // Stage separator: drawn on the RIGHT edge of each column except the
    // last, so it acts as a gutter between stages rather than cutting
    // through the middle of the cluster (which it would do at column.x
    // when a stage packs nodes into multiple sub-columns).
    if(index<rankColumns.length-1){
      ctx.beginPath();
      ctx.moveTo(column.x+half,topY);
      ctx.lineTo(column.x+half,botY);
      ctx.strokeStyle=_light ? "rgba(80,80,80,0.18)" : "rgba(219,216,216,0.15)";
      ctx.lineWidth=1/scale;
      ctx.stroke();
    }

    if(scale>0.12){
      // Pill-style stage header drawn below the tab bar
      var label=truncate(column.label,22);
      ctx.font="600 "+(11/scale)+"px Inter,system-ui,sans-serif";
      ctx.textAlign="center";
      ctx.textBaseline="top";
      // Small backdrop chip so the label reads cleanly even when nodes
      // approach the top of the column.
      var labelY=topY+8/scale;
      var pillH=20/scale;
      var pillW=Math.max(60/scale,ctx.measureText(label).width+18/scale);
      ctx.fillStyle=_light?"rgba(255,255,255,0.85)":"rgba(20,20,20,0.85)";
      ctx.beginPath();
      ctx.roundRect
        ? ctx.roundRect(column.x-pillW/2,labelY-2/scale,pillW,pillH,6/scale)
        : ctx.rect(column.x-pillW/2,labelY-2/scale,pillW,pillH);
      ctx.fill();
      ctx.fillStyle=_light ? "rgba(30,30,30,0.85)" : "rgba(244,244,244,0.85)";
      ctx.fillText(label,column.x,labelY+2/scale);
    }
  });
  ctx.restore();
}

function drawOrganicClusters(vp,_light){
  if(layoutMode!=="organic"||organicClusters.length<2)return;
  ctx.save();
  organicClusters.forEach(function(cluster){
    var rx=cluster.radius*1.3;
    var ry=cluster.radius*0.86;
    if(cluster.x+rx<vp.x1||cluster.x-rx>vp.x2||cluster.y+ry<vp.y1||cluster.y-ry>vp.y2)return;
    var rgb=typeRgbCache[cluster.type]||[165,80,255];
    ctx.beginPath();
    ctx.ellipse(cluster.x,cluster.y,rx,ry,0,0,Math.PI*2);
    ctx.fillStyle="rgba("+rgb[0]+","+rgb[1]+","+rgb[2]+","+(_light?0.055:0.075)+")";
    ctx.fill();
    ctx.strokeStyle="rgba("+rgb[0]+","+rgb[1]+","+rgb[2]+","+(_light?0.18:0.24)+")";
    ctx.lineWidth=1/scale;
    ctx.stroke();

    if(scale>0.12){
      ctx.font="600 "+(11/scale)+"px Inter,system-ui,sans-serif";
      ctx.textAlign="center";
      ctx.textBaseline="middle";
      ctx.fillStyle=_light ? "rgba(30,30,30,0.5)" : "rgba(244,244,244,0.5)";
      ctx.fillText(truncate(cluster.type+" ("+cluster.count+")",26),cluster.x,cluster.y-ry-12/scale);
    }
  });
  ctx.restore();
}

// ── Drawing (Phase 3b-3e: viewport culling, batched edges, LOD, adaptive labels) ──
function draw(){
  updateFps();
  ctx.save();
  ctx.clearRect(0,0,W,H);

  // Background
  var _light = window._isLightMode;
  ctx.fillStyle=_light ? "#f5f5f5" : "#000000";
  ctx.fillRect(0,0,W,H);

  // Subtle grid
  ctx.save();
  ctx.globalAlpha=0.03;
  ctx.strokeStyle=_light ? "#000" : "#fff";
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

  drawRankColumns(vp,_light);
  drawOrganicClusters(vp,_light);

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

      // Classify edges into batches by visual state.
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
        ctx.strokeStyle=_light ? "rgba(80,80,80,"+Math.min(nAlpha*3,0.3)+")" : "rgba(219,216,216,"+nAlpha+")";
        ctx.lineWidth=(_light ? 0.8 : 0.5)/scale;
        ctx.stroke();
      }


      // Draw search-matched edges
      searchEdges.forEach(function(e){
        ctx.beginPath();
        ctx.moveTo(e.l.source.x,e.l.source.y);
        ctx.lineTo(e.l.target.x,e.l.target.y);
        ctx.strokeStyle=_light ? "rgba(60,60,60,"+e.alpha+")" : "rgba(219,216,216,"+e.alpha+")";
        ctx.lineWidth=e.lineW/scale;
        ctx.stroke();
      });

      // Draw hover edges
      hoverEdges.forEach(function(e){
        ctx.beginPath();
        ctx.moveTo(e.l.source.x,e.l.source.y);
        ctx.lineTo(e.l.target.x,e.l.target.y);
        ctx.strokeStyle=_light ? "rgba(60,60,60,"+e.alpha+")" : "rgba(219,216,216,"+e.alpha+")";
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
        ctx.fillStyle=_light ? "rgba(30,30,30,0.9)" : "rgba(219,216,216,0.7)";
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
        ctx.strokeStyle=_light ? "rgba(0,0,0,0.15)" : "rgba(244,244,244,0.2)";
        ctx.lineWidth=0.5;
        ctx.stroke();
      }

      // Ontology-valid marker: bright outer ring so these nodes are
      // identifiable at a glance — the previous gray-fill cue was too
      // subtle to spot.
      if(n.ontology_valid===true){
        ctx.beginPath();
        ctx.arc(n.x,n.y,r+2.2/scale,0,Math.PI*2);
        ctx.strokeStyle=_light ? "rgba(13,180,40,0.85)" : "rgba(13,255,0,0.85)";
        ctx.lineWidth=1.6/scale;
        ctx.stroke();
      }
      ctx.globalAlpha=1;
    });
  }

  // ── Labels (Phase 3e: adaptive; Phase 1b: budget-aware) ──
  // Effective labels are gated by both the legacy Labels layer toggle
  // (data-layer="labels") and the new label budget (off | key | all).
  // The labels layer drives disable-all; the budget shapes the picker.
  if(layers.labels&&labelBudget!=="off"){
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

        // Always label hovered + neighbors (budget-independent)
        if(hoveredNode&&neighborSet&&neighborSet.has(n.id))shouldLabel=true;
        // Always label search matches
        if(hasSearch&&searchMatches.has(n.id))shouldLabel=true;

        if(labelBudget==="key"){
          // Default: label only nodes the preprocessor marked as
          // label_priority (documents, entity-types, top-quartile
          // importance). Hovered/searched already opted-in above.
          if(n.label_priority===true)shouldLabel=true;
        }else if(N>5000){
          // "all" mode on a huge graph still has to cap somewhere
          if(topDegreeSet.has(n.id))shouldLabel=true;
        }else{
          // "all" mode — original legacy adaptive behavior
          if(n._degree>=maxDeg*0.3)shouldLabel=true;
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
        ctx.fillStyle=_light ? "rgba(255,255,255,"+alpha2*0.8+")" : "rgba(0,0,0,"+alpha2*0.8+")";
        ctx.fillText(label,n.x+0.5,n.y+yOff+0.5);

        ctx.fillStyle=_light ? "rgba(30,30,30,"+alpha2+")" : "rgba(244,244,244,"+alpha2+")";
        ctx.fillText(label,n.x,n.y+yOff);
      });
    }else if(hoveredNode){
      // Even with labels disabled, label hovered node
      var hn=hoveredNode;
      var label2=truncate(hn.name||"",30);
      var yOff2=hn._r+fontSize*0.8;
      ctx.fillStyle=_light ? "rgba(255,255,255,0.8)" : "rgba(0,0,0,0.8)";
      ctx.fillText(label2,hn.x+0.5,hn.y+yOff2+0.5);
      ctx.fillStyle=_light ? "rgba(30,30,30,1)" : "rgba(244,244,244,1)";
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
// ── Keyboard navigation (Phase 1 polish) ──
// Arrow keys walk the selection through the Story columns.
// Right/Left: move to the next/previous stage column, snapping to the
// closest-y node in that column. Up/Down: move within the current
// column. Enter opens the inspector for the selected node. Escape
// clears selection.
function selectByKey(dir){
  if(layoutMode!=="story"&&layoutMode!=="ranked")return;
  if(rankColumns.length<2)return;
  var current=hoveredNode;
  if(!current){
    // Pick the first node in the first column as a starting point
    for(var ii=0;ii<nodes.length;ii++){
      if(nodes[ii]._layoutRank===0){current=nodes[ii];break}
    }
    if(!current)return;
    hoveredNode=current;
    showNodeInfo(current);
    draw();
    return;
  }
  var curRank=current._layoutRank||0;
  var byRank={};
  nodes.forEach(function(n){
    var r=n._layoutRank||0;
    if(!byRank[r])byRank[r]=[];
    byRank[r].push(n);
  });
  Object.keys(byRank).forEach(function(r){
    byRank[r].sort(function(a,b){return (a._targetY||0)-(b._targetY||0)});
  });
  var target=null;
  if(dir==="left"||dir==="right"){
    var nextRank=curRank+(dir==="right"?1:-1);
    if(!byRank[nextRank])return;
    // Snap to closest-y in next column
    var refY=current._targetY||current.y||0;
    var best=null,bestDy=Infinity;
    byRank[nextRank].forEach(function(n){
      var dy=Math.abs((n._targetY||n.y||0)-refY);
      if(dy<bestDy){bestDy=dy;best=n}
    });
    target=best;
  }else if(dir==="up"||dir==="down"){
    var col=byRank[curRank]||[];
    var idx=col.indexOf(current);
    if(idx===-1)return;
    var nextIdx=idx+(dir==="down"?1:-1);
    if(nextIdx<0||nextIdx>=col.length)return;
    target=col[nextIdx];
  }
  if(!target)return;
  hoveredNode=target;
  showNodeInfo(target);
  // Pan to keep the selection in view
  var sx=target.x*scale+tx,sy=target.y*scale+ty;
  var margin=80;
  if(sx<margin||sx>W-margin||sy<margin||sy>H-margin){
    tx=W/2-target.x*scale;
    ty=H/2-target.y*scale;
  }
  draw();
}

document.addEventListener("keydown",function(ev){
  if(ev.target===searchInput)return;
  if(ev.key==="f"||ev.key==="/"){ev.preventDefault();searchInput.focus()}
  if(ev.key==="Escape"){
    searchInput.value="";searchQuery="";searchMatches.clear();
    hoveredNode=null;
    infoPanel.classList.remove("visible");
    searchInput.blur();
    draw();
  }
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
  // Phase 1 polish: pipeline keyboard nav
  if(ev.key==="ArrowRight"){ev.preventDefault();selectByKey("right")}
  if(ev.key==="ArrowLeft"){ev.preventDefault();selectByKey("left")}
  if(ev.key==="ArrowDown"){ev.preventDefault();selectByKey("down")}
  if(ev.key==="ArrowUp"){ev.preventDefault();selectByKey("up")}
  if(ev.key==="Enter"&&hoveredNode){showNodeInfo(hoveredNode)}
});

// Trigger density computation after initial layout
scheduleDensity();

// ── URL hash sync (Phase 1 polish) ──
// Encodes view state in #view=graph|schema&layout=story|ranked|organic
// &label=key|all|off&color=type|task|pipeline|nodeset|user so a screenshot
// URL reproduces the same view. Reads on load (clicking the appropriate
// control buttons so all existing wiring fires), writes on every state
// change. Debounced 200ms to avoid hammering history on slider-style use.
function _viewTabFromHash(){
  var tab=document.querySelector('.tab-btn[data-view="schema"]')&&document.querySelector('.tab-btn[data-view="graph"]');
  return tab;
}
function readHashState(){
  if(!location.hash||location.hash.length<2)return;
  var params={};
  location.hash.slice(1).split("&").forEach(function(p){
    var kv=p.split("=");
    if(kv[0])params[decodeURIComponent(kv[0])]=decodeURIComponent(kv[1]||"");
  });
  function clickIf(selector){
    var btn=document.querySelector(selector);
    if(btn&&!btn.classList.contains("active"))btn.click();
  }
  if(params.layout)clickIf('.ctrl-btn[data-layout="'+params.layout+'"]');
  if(params.label)clickIf('.ctrl-btn[data-labelbudget="'+params.label+'"]');
  if(params.color)clickIf('.ctrl-btn[data-colorby="'+params.color+'"]');
  if(params.view==="schema"){
    var t=document.querySelector('.tab-btn[data-view="schema"]');
    if(t)t.click();
  }
}
var _hashWriteTimer=null;
function writeHash(){
  if(_hashWriteTimer)clearTimeout(_hashWriteTimer);
  _hashWriteTimer=setTimeout(function(){
    var activeTab=document.querySelector('.tab-btn.active');
    var view=activeTab?activeTab.dataset.view:"graph";
    var parts=[];
    if(view!=="graph")parts.push("view="+view);
    if(layoutMode!=="story")parts.push("layout="+layoutMode);
    if(labelBudget!=="key")parts.push("label="+labelBudget);
    if(colorByMode!=="type")parts.push("color="+colorByMode);
    var hash=parts.length?"#"+parts.join("&"):"";
    if(hash!==location.hash){
      history.replaceState(null,"",location.pathname+location.search+hash);
    }
  },200);
}
// Augment existing control handlers to write the hash on click.
document.querySelectorAll(".ctrl-btn[data-layout], .ctrl-btn[data-labelbudget], .ctrl-btn[data-colorby], .tab-btn[data-view]").forEach(function(btn){
  btn.addEventListener("click",writeHash);
});
// Apply hash on initial load (deferred so all listeners are wired).
setTimeout(readHashState,0);
window.addEventListener("hashchange",readHashState);

})();
