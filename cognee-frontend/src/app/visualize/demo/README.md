# Graph Visualization Demo

An isolated, interactive demo of Cognee's Three.js-based graph visualization with a rich AI/ML knowledge graph dataset.

## Features

### 🎨 Visual Design
- **Vibrant Color Palette**: 10 distinct colors for different node types
- **Dark Theme**: Optimized background (#0a0a0f) for maximum contrast
- **Metaball Rendering**: Smooth, organic blob visualization of node clusters
- **Responsive Labels**: Context-aware labels that appear on hover and zoom

### 🎯 Interactive Controls
- **Pan**: Click and drag to move around the graph
- **Zoom**: Scroll to zoom in (6x max) or out (0.5x min)
- **Hover**: Mouse over nodes to see labels and connections
- **Click**: Select nodes to highlight their relationships
- **Smooth Animation**: Fluid camera motion with optimized damping

### 📊 UI Components
- **Legend Panel**: Categorizes nodes by type with color coding
- **Statistics**: Real-time graph metrics (nodes, edges, connections)
- **Instructions Overlay**: Quick reference for interaction methods
- **Toggle Controls**: Show/hide legend and stats as needed

## Dataset

The demo includes a comprehensive **AI/ML Knowledge Graph** with:

### Node Types (52 total)
- **Concepts** (6): AI, Machine Learning, Deep Learning, NLP, CV, RL
- **Algorithms** (10): SVM, Decision Trees, K-Means, Q-Learning, etc.
- **Architectures** (12): CNN, RNN, Transformer, GAN, VAE, etc.
- **Technologies** (9): BERT, GPT, ResNet, YOLO, Word2Vec, etc.
- **Applications** (5): Chatbots, Autonomous Vehicles, Medical Imaging, etc.
- **Data** (4): Datasets, Feature Engineering, Augmentation, Normalization
- **Optimization** (5): Gradient Descent, Adam, Backprop, Regularization, Dropout

### Relationships (56 edges)
- Hierarchical: "is subfield of", "type of", "variant of"
- Functional: "implements", "uses", "powered by", "trains"
- Application: "application of", "task in", "used in"

## Technical Implementation

### Architecture
```
GraphVisualization Component
    ↓
animate.ts (Main Render Loop)
    ↓
┌─────────────────┬─────────────────┬─────────────────┐
│  Graph Layout   │   Rendering     │    Interaction  │
│  (ngraph)       │   (Three.js)    │    (Picking)    │
├─────────────────┼─────────────────┼─────────────────┤
│ • Force layout  │ • Node swarm    │ • Mouse hover   │
│ • 800 iterations│ • Edge mesh     │ • Click select  │
│ • Spring physics│ • Metaballs     │ • Label display │
│                 │ • Density cloud │ • Pan/Zoom      │
└─────────────────┴─────────────────┴─────────────────┘
```

### Performance Optimizations
- **GPU-Accelerated**: All rendering uses WebGL shaders
- **Instanced Rendering**: Nodes rendered in a single draw call
- **Texture-Based Positions**: Node positions stored in GPU texture
- **Culling**: Labels only shown for visible/hovered nodes
- **Adaptive Layout**: Physics stabilizes after initial iterations

### Scalability
The implementation is designed to handle:
- ✅ **100+ nodes**: Excellent performance
- ✅ **500+ nodes**: Good performance with metaballs
- ✅ **1000+ nodes**: Recommended to reduce metaball density
- ⚠️ **5000+ nodes**: Consider simplified rendering mode

## Configuration Options

The visualization accepts a `config` prop:

```typescript
config={{
  fontSize: 12,  // Label font size (default: 10)
}}
```

### Force Layout Parameters (in animate.ts)
```typescript
{
  dragCoefficient: 0.8,      // Node movement resistance
  springLength: 180,          // Ideal distance between connected nodes
  springCoefficient: 0.25,    // Connection strength
  gravity: -1200,             // Repulsion force
}
```

### Camera Controls
```typescript
{
  minZoom: 0.5,              // Maximum zoom out
  maxZoom: 6,                // Maximum zoom in
  dampingFactor: 0.08,       // Camera smoothness
}
```

## How to Use in Development

1. **Start the frontend**:
   ```bash
   cd cognee-frontend
   npm run dev
   ```

2. **Navigate to the demo**:
   ```
   http://localhost:3000/visualize/demo
   ```

3. **Interact with the graph**:
   - Hover over nodes to see labels
   - Zoom in to see more connections
   - Click to select nodes
   - Toggle legend/stats panels

## Extending the Demo

### Add More Nodes
```typescript
mockNodes.push({
  id: "new-concept",
  label: "New Concept",
  type: "Concept"
});
```

### Add Connections
```typescript
mockEdges.push({
  id: "e-new",
  source: "new-concept",
  target: "ai",
  label: "related to"
});
```

### Customize Colors
Update the `typeColors` mapping in the demo page:
```typescript
const typeColors: Record<string, string> = {
  "YourType": "#YOUR_COLOR",
  // ...
};
```

## Future Enhancements

Potential improvements:
- [ ] Search functionality to find and highlight nodes
- [ ] Filter nodes by type
- [ ] Export graph as image/SVG
- [ ] Node clustering by community detection
- [ ] Time-based animation of graph evolution
- [ ] 3D visualization mode
- [ ] Multi-graph comparison view

## Related Files

- `src/ui/elements/GraphVisualization.tsx` - Main component wrapper
- `src/ui/rendering/animate.ts` - Render loop and Three.js setup
- `src/ui/rendering/graph/createGraph.ts` - Graph creation from data
- `src/ui/rendering/materials/` - Shader materials for visual effects
- `src/ui/rendering/meshes/` - Mesh generation for nodes/edges/labels

## Performance Tips

For large graphs (1000+ nodes):
1. Reduce `densityCloudTarget` resolution (line 135 in animate.ts)
2. Decrease label display limit (line 372)
3. Consider disabling metaball rendering for very large graphs
4. Use node clustering/aggregation for massive datasets

## License

Part of the Cognee project - Apache 2.0 License
