// Theme toggle
(function(){
  const btn = document.getElementById('theme-toggle');
  // Expose for canvas draw() to read. Default light so the Graph view matches
  // the Schema view instead of opening dark.
  window._isLightMode = true;
  // Sync the CSS theme class with the JS default. Without this the page
  // loads with dark :root variables while the JS believes it's light, and
  // the first "Dark mode" click ADDS the light class — a visual no-op — so
  // dark mode was unreachable.
  document.documentElement.classList.add('light');
  btn.addEventListener('click', () => {
    document.documentElement.classList.toggle('light');
    const isLight = document.documentElement.classList.contains('light');
    window._isLightMode = isLight;
    btn.textContent = isLight ? 'Dark mode' : 'Light mode';
    // Update tab bar background (fully opaque so canvas content can't bleed through)
    const tabBar = document.getElementById('view-tabs');
    tabBar.style.background = isLight ? '#f5f5f5' : '#000000';
    // Re-render schema if visible
    if (document.getElementById('schema-view').style.display !== 'none' && window._renderSchemaGraph) {
      window._renderSchemaGraph();
    }
  });
})();

// Tab switching logic
(function(){
  const tabs = document.querySelectorAll('.tab-btn');
  const graphView = document.getElementById('graph-view');
  const schemaView = document.getElementById('schema-view');
  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      tabs.forEach(t => { t.style.background='transparent'; t.style.color='var(--text2)'; t.classList.remove('active'); });
      btn.style.background='#1F9E6E'; btn.style.color='#fff'; btn.classList.add('active');
      const view = btn.dataset.view;
      graphView.style.display = view === 'graph' ? '' : 'none';
      schemaView.style.display = view === 'schema' ? '' : 'none';
      if (view === 'schema' && window._renderSchemaGraph) window._renderSchemaGraph();
    });
  });
})();
