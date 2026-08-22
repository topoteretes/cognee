// Schema type inspector side panel (PR3 + instance drill-down).
//
// Opened by schema_view.js's node-box click handler via
// window._showSchemaInspector(typeName). Reads:
//   * window._schemaTypeIndex        — per-type aggregate (count, samples,
//                                       relationships) from the PR2 payload.
//   * window._schemaInstancesByType  — {type: [{id, name}, …]} every instance.
//   * window._schemaInstanceIndex    — {id: {id, name, type, out[], in[]}}
//                                       compact per-instance adjacency.
//
// Two navigable views share one panel:
//   • Type view     — count + clickable instance chips + type-level flows.
//   • Instance view — a breadcrumb back to the type, then the instance's own
//                     outgoing/incoming connections; each neighbour is itself
//                     clickable, so you can walk the ownership/data hierarchy
//                     (Org → User → Brain → File → Memory) in place.
(function () {
  var panel = document.getElementById("schema-side-panel");
  if (!panel) return;

  function esc(s) {
    return String(s == null ? "" : s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function typeIndex() { return window._schemaTypeIndex || {}; }
  function instancesByType() { return window._schemaInstancesByType || {}; }
  function instanceIndex() { return window._schemaInstanceIndex || {}; }
  function nameOf(id) { var n = instanceIndex()[id]; return n ? n.name : id; }

  window._hideSchemaInspector = function () {
    panel.style.display = "none";
    panel.innerHTML = "";
  };

  function chip(label, iid) {
    if (iid) {
      return '<span class="si-chip si-chip-link" data-iid="' + esc(iid) + '">' + esc(label) + "</span>";
    }
    return '<span class="si-chip">' + esc(label) + "</span>";
  }

  // ── Type view ──────────────────────────────────────────────────────
  function renderTypeRelationships(t) {
    var rels = t.relationships || [];
    if (!rels.length) return '<div class="si-empty">No outgoing relationships.</div>';
    return rels
      .map(function (r) {
        return (
          '<div class="si-rel">' +
          '<span class="si-rel-name">' + esc(r.relation) + "</span>" +
          '<span class="si-rel-to">&rarr; ' + esc(r.to_type) + "</span>" +
          '<span class="si-rel-count">' + Number(r.count ?? 0) + "</span>" +
          "</div>"
        );
      })
      .join("");
  }

  function renderType(name, showAll) {
    var t = typeIndex()[name];
    if (!t) { window._hideSchemaInspector(); return; }
    var insts = instancesByType()[name] || [];
    var count = Number(t.instance_count ?? insts.length);

    var chipsHtml;
    if (insts.length) {
      var visible = showAll ? insts : insts.slice(0, 5);
      chipsHtml =
        '<div class="si-chips">' +
        visible.map(function (r) { return chip(r.name, r.id); }).join("") +
        "</div>";
      if (!showAll && insts.length > 5) {
        chipsHtml += '<button class="si-showall" data-action="showall-type">Show all ' + insts.length + "</button>";
      }
    } else {
      // Fallback to non-clickable sample names if no instance index is present.
      var samples = t.samples || [];
      chipsHtml =
        '<div class="si-chips">' +
        samples.map(function (n) { return chip(n, null); }).join("") +
        "</div>";
      var beyond = count - samples.length;
      if (beyond > 0) chipsHtml += '<span class="si-more">… +' + beyond + " more</span>";
    }

    panel.innerHTML =
      '<div class="si-close" data-action="close" title="Close">&times;</div>' +
      '<div class="si-title">' + esc(t.name) + "</div>" +
      '<div class="si-count">' + esc(t.name) + " — " + count + (count === 1 ? " instance" : " instances") + "</div>" +
      '<div class="si-heading">Instances</div>' + chipsHtml +
      '<div class="si-heading">Relationships</div>' + renderTypeRelationships(t) +
      '<button class="si-highlight" data-action="highlight">Highlight in graph</button>';
    panel.style.display = "block";
    panel.dataset.name = name;
    wire();
  }

  // ── Instance view ──────────────────────────────────────────────────
  function groupEdges(list) {
    var grouped = {};
    (list || []).forEach(function (e) {
      (grouped[e.relation] = grouped[e.relation] || []).push(e.id);
    });
    return grouped;
  }

  function relBlock(relation, ids, dir) {
    var arrow = dir === "out" ? "&rarr;" : "&larr;";
    var chips = ids.map(function (id) { return chip(nameOf(id), id); }).join("");
    return (
      '<div class="si-irel">' +
      '<div class="si-irel-head"><span class="si-rel-name">' + esc(relation) + "</span> " +
      '<span class="si-rel-to">' + arrow + "</span></div>" +
      '<div class="si-chips">' + chips + "</div>" +
      "</div>"
    );
  }

  function renderInstance(id) {
    var node = instanceIndex()[id];
    if (!node) { window._hideSchemaInspector(); return; }
    var out = groupEdges(node.out);
    var inc = groupEdges(node["in"]);
    var blocks = "";
    Object.keys(out).forEach(function (rel) { blocks += relBlock(rel, out[rel], "out"); });
    Object.keys(inc).forEach(function (rel) { blocks += relBlock(rel, inc[rel], "in"); });
    if (!blocks) blocks = '<div class="si-empty">No connections.</div>';

    panel.innerHTML =
      '<div class="si-close" data-action="close" title="Close">&times;</div>' +
      '<div class="si-crumb" data-action="back-type" data-type="' + esc(node.type) + '">&lsaquo; ' + esc(node.type) + "</div>" +
      '<div class="si-title">' + esc(node.name) + "</div>" +
      '<div class="si-count">' + esc(node.type) + "</div>" +
      '<div class="si-heading">Connections</div>' + blocks;
    panel.style.display = "block";
    wire();
  }

  // ── Shared event wiring ────────────────────────────────────────────
  function wire() {
    var close = panel.querySelector('[data-action="close"]');
    if (close) {
      close.addEventListener("click", function () {
        if (window._clearSchemaLens) window._clearSchemaLens();
        window._hideSchemaInspector();
      });
    }
    panel.querySelectorAll("[data-iid]").forEach(function (el) {
      el.addEventListener("click", function (e) {
        e.stopPropagation();
        var iid = el.getAttribute("data-iid");
        renderInstance(iid);
        // Lens the diagram to this instance's subtree as you drill in.
        if (window._focusSchemaInstance) window._focusSchemaInstance(iid);
      });
    });
    var showall = panel.querySelector('[data-action="showall-type"]');
    if (showall) showall.addEventListener("click", function () { renderType(panel.dataset.name, true); });
    var back = panel.querySelector('[data-action="back-type"]');
    if (back) {
      back.addEventListener("click", function () {
        // Returning to the type view also clears the diagram lens.
        if (window._clearSchemaLens) window._clearSchemaLens();
        renderType(back.getAttribute("data-type"), false);
      });
    }
    var highlight = panel.querySelector('[data-action="highlight"]');
    if (highlight) {
      highlight.addEventListener("click", function () {
        var graphTab = document.querySelector('.tab-btn[data-view="graph"]');
        if (graphTab) graphTab.click();
        if (window._highlightSchemaType) window._highlightSchemaType(panel.dataset.name);
      });
    }
  }

  window._showSchemaInspector = function (name) { renderType(name, false); };
  // Entry point used by the lensed diagram when an instance box is clicked.
  window._showSchemaInstanceInspector = function (id) { renderInstance(id); };
})();
