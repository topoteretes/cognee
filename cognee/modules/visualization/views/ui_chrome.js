// Theme toggle
(function(){
  const btn = document.getElementById('theme-toggle');

  // Apply a theme everywhere it matters: CSS class, JS flag (read by the
  // canvas draw loop), button label, tab bar, the canvas itself (which only
  // repaints on interaction — without an explicit frame the 95% of the
  // screen that is canvas stayed in the old theme until the next pan), and
  // the schema SVG when visible.
  function applyTheme(isLight) {
    window._isLightMode = isLight;
    document.documentElement.classList.toggle('light', isLight);
    btn.textContent = isLight ? 'Dark mode' : 'Light mode';
    const tabBar = document.getElementById('view-tabs');
    if (tabBar) tabBar.style.background = isLight ? '#f5f5f5' : '#000000';
    if (window._requestGraphRedraw) window._requestGraphRedraw();
    if (document.getElementById('schema-view').style.display !== 'none' && window._renderSchemaGraph) {
      window._renderSchemaGraph(true);  // preserve pan/zoom across the re-render
    }
    const memoryViewEl = document.getElementById('memory-view');
    if (memoryViewEl && memoryViewEl.style.display !== 'none' && window._renderMemoryView) {
      window._renderMemoryView(true);  // preserve pan/zoom across the re-render
    }
  }

  // Restore the user's last choice (default light). Syncing the CSS class on
  // load also fixes the original bug: the page shipped dark :root variables
  // while the JS assumed light, so the first toggle was a visual no-op.
  let stored = null;
  try { stored = localStorage.getItem('cognee-viz-theme'); } catch (e) { /* file:// may block */ }
  applyTheme(stored ? stored === 'light' : true);

  btn.addEventListener('click', () => {
    const isLight = !window._isLightMode;
    try { localStorage.setItem('cognee-viz-theme', isLight ? 'light' : 'dark'); } catch (e) { /* best effort */ }
    applyTheme(isLight);
  });
})();

// Tab switching logic
(function(){
  const tabs = document.querySelectorAll('.tab-btn');
  const graphView = document.getElementById('graph-view');
  const schemaView = document.getElementById('schema-view');
  const memoryView = document.getElementById('memory-view');
  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      tabs.forEach(t => { t.style.background='transparent'; t.style.color='var(--text2)'; t.classList.remove('active'); });
      btn.style.background='#1F9E6E'; btn.style.color='#fff'; btn.classList.add('active');
      const view = btn.dataset.view;
      graphView.style.display = view === 'graph' ? '' : 'none';
      schemaView.style.display = view === 'schema' ? '' : 'none';
      if (memoryView) memoryView.style.display = view === 'memory' ? '' : 'none';
      if (view === 'schema' && window._renderSchemaGraph) window._renderSchemaGraph();
      if (view === 'memory' && window._renderMemoryView) window._renderMemoryView();
    });
  });
})();
