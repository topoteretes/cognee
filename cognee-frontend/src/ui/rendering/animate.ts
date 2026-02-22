import { Graph, Node as GraphNode, Link as GraphLink } from "ngraph.graph";
import * as three from "three";
import {
  Color,
  DataTexture,
  OrthographicCamera,
  RGBAFormat,
  Scene,
  UnsignedByteType,
  Vector2,
  WebGLRenderer,
  WebGLRenderTarget,
} from "three";
import { OrbitControls } from "three/examples/jsm/Addons.js";
import createForceLayout, { Layout } from "ngraph.forcelayout";

import { Edge, Node } from "./graph/types";
import createGraph from "./graph/createGraph";

import createLabel from "./meshes/createLabel";
import pickNodeIndex from "./picking/pickNodeIndex";
import createEdgeMesh from "./meshes/createEdgeMesh";
import createPickingMesh from "./meshes/createPickingMesh";
import createNodeSwarmMesh from "./meshes/createNodeSwarmMesh";
import createNodePositionsTexture from "./textures/createNodePositionsTexture";
import createDensityRenderTarget from "./render-targets/createDensityRenderTarget";
import createDensityAccumulatorMesh from "./meshes/createDensityAccumulatorMesh";
import createMetaballMesh from "./meshes/createMetaballMesh";
import createClusterBoundaryMesh, { ClusterInfo } from "./meshes/createClusterBoundaryMesh";

const INITIAL_CAMERA_DISTANCE = 2000;

// Extended config for layered view controls + zoom semantics
interface Config {
  fontSize?: number;
  showNodes?: boolean;
  showEdges?: boolean;
  showMetaballs?: boolean;
  pathFilterMode?: "all" | "hoverOnly" | "strongOnly"; // Path filtering
  zoomLevel?: "far" | "mid" | "near"; // Zoom-based semantics
  highlightedNodeIds?: Set<string>; // Nodes to highlight (neutral-by-default)
}

export default function animate(
  nodes: Node[],
  edges: Edge[],
  parentElement: HTMLElement,
  config?: Config
): () => void {
  const nodeLabelMap = new Map();
  const edgeLabelMap = new Map();

  // Semantic color encoding: hierarchy drives saturation + brightness
  const typeColorMap: Record<string, string> = {
    "Domain": "#C4B5FD",       // Bright Purple - Highest importance
    "Field": "#67E8F9",         // Bright Cyan - High importance
    "Subfield": "#A78BFA",      // Medium Purple - Medium-high importance
    "Concept": "#5EEAD4",       // Teal - Medium importance
    "Method": "#6EE7B7",        // Green - Medium importance
    "Theory": "#F9A8D4",        // Pink - Medium importance
    "Technology": "#FCA5A5",    // Soft Red - Lower importance
    "Application": "#71717A",   // Desaturated Gray - Background/lowest importance
  };

  // Size hierarchy: more important = larger
  const typeSizeMap: Record<string, number> = {
    "Domain": 2.5,       // Largest
    "Field": 2.0,
    "Subfield": 1.6,
    "Concept": 1.2,
    "Method": 1.1,
    "Theory": 1.0,
    "Technology": 0.9,
    "Application": 0.6,  // Smallest
  };

  function getColorForType(nodeType: string): Color {
    const colorHex = typeColorMap[nodeType];
    if (colorHex) {
      return new Color(colorHex);
    }
    // Fallback for unknown types
    return new Color("#9CA3AF"); // Gray for unknown types
  }

  const mousePosition = new Vector2();

  // Node related data
  const nodeColors = new Float32Array(nodes.length * 3);
  const nodeSizes = new Float32Array(nodes.length); // Size per node for hierarchy
  const nodeHighlights = new Float32Array(nodes.length); // 1.0 = highlighted, 0.3 = dimmed
  const nodeIndices = new Map();
  const textureSize = Math.ceil(Math.sqrt(nodes.length));
  const nodePositionsData = new Float32Array(textureSize * textureSize * 4);

  // Determine which nodes are highlighted
  const highlightedIds = config?.highlightedNodeIds;
  const hasHighlights = highlightedIds && highlightedIds.size > 0;

  let nodeIndex = 0;
  function forNode(node: Node) {
    const color = getColorForType(node.type);
    nodeColors[nodeIndex * 3 + 0] = color.r;
    nodeColors[nodeIndex * 3 + 1] = color.g;
    nodeColors[nodeIndex * 3 + 2] = color.b;

    // Set highlight state: if no highlights, all at 1.0; if highlights exist, dim non-highlighted
    if (hasHighlights) {
      nodeHighlights[nodeIndex] = highlightedIds!.has(node.id) ? 1.0 : 0.3;
    } else {
      nodeHighlights[nodeIndex] = 1.0; // All visible when no highlights
    }

    // Store size multiplier based on type
    nodeSizes[nodeIndex] = typeSizeMap[node.type] || 1.0;

    nodePositionsData[nodeIndex * 4 + 0] = 0.0;
    nodePositionsData[nodeIndex * 4 + 1] = 0.0;
    nodePositionsData[nodeIndex * 4 + 2] = 0.0;
    nodePositionsData[nodeIndex * 4 + 3] = 1.0;

    nodeIndices.set(node.id, nodeIndex);

    nodeIndex += 1;
  }

  // Node related data
  const edgeIndices = new Float32Array(edges.length * 2);

  let edgeIndex = 0;
  function forEdge(edge: Edge) {
    const fromIndex = nodeIndices.get(edge.source);
    const toIndex = nodeIndices.get(edge.target);
    edgeIndices[edgeIndex * 2 + 0] = fromIndex;
    edgeIndices[edgeIndex * 2 + 1] = toIndex;

    edgeIndex += 1;
  }

  // Graph creation and layout
  const graph = createGraph(nodes, edges, forNode, forEdge);

  // Adaptive layout parameters based on graph size
  const nodeCount = nodes.length;
  const isLargeGraph = nodeCount > 5000;
  const isMassiveGraph = nodeCount > 15000;

  // Apple embedding atlas style: stronger repulsion for clear cluster separation
  const graphLayout = createForceLayout(graph, {
    dragCoefficient: isMassiveGraph ? 0.95 : 0.85,
    springLength: isMassiveGraph ? 120 : isLargeGraph ? 180 : 220, // Longer springs for spacing
    springCoefficient: isMassiveGraph ? 0.12 : isLargeGraph ? 0.15 : 0.18, // Weaker springs
    gravity: isMassiveGraph ? -1200 : isLargeGraph ? -1500 : -1800, // Stronger repulsion
  });

  // Node Mesh
  const nodePositionsTexture = createNodePositionsTexture(
    nodes,
    nodePositionsData
  );

  const nodeSwarmMesh = createNodeSwarmMesh(
    nodes,
    nodePositionsTexture,
    nodeColors,
    nodeSizes,
    nodeHighlights,
    INITIAL_CAMERA_DISTANCE
  );

  const edgeMesh = createEdgeMesh(
    edges,
    nodePositionsTexture,
    edgeIndices,
    INITIAL_CAMERA_DISTANCE
  );

  // Density cloud setup - adaptive resolution for performance
  const densityCloudScene = new Scene();
  const densityResolution = isMassiveGraph ? 256 : isLargeGraph ? 384 : 512;
  const densityCloudTarget = createDensityRenderTarget(densityResolution);

  const densityAccumulatorMesh = createDensityAccumulatorMesh(
    nodes,
    nodeColors,
    nodePositionsTexture,
    INITIAL_CAMERA_DISTANCE
  );

  const metaballMesh = createMetaballMesh(densityCloudTarget);

  // const densityCloudDebugMesh = createDebugViewMesh(densityCloudTarget);
  // Density cloud setup end

  let pickedNodeIndex = -1;
  const lastPickedNodeIndex = -1;
  const pickNodeFromScene = (event: unknown) => {
    pickedNodeIndex = pickNodeIndexFromScene(event as MouseEvent);
  };

  parentElement.addEventListener("mousemove", (event) => {
    const rect = parentElement.getBoundingClientRect();
    mousePosition.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mousePosition.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    pickNodeFromScene(event);
  });

  // Group nodes by type for cluster boundaries
  const nodesByType = new Map<string, Node[]>();
  nodes.forEach(node => {
    if (!nodesByType.has(node.type)) {
      nodesByType.set(node.type, []);
    }
    nodesByType.get(node.type)!.push(node);
  });

  const scene = new Scene();
  // Apple embedding atlas style: pure black background
  scene.background = new Color("#000000");
  const renderer = new WebGLRenderer({ antialias: true });

  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(parentElement.clientWidth, parentElement.clientHeight);
  if (parentElement.children.length === 0) {
    parentElement.appendChild(renderer.domElement);
  }

  // Setup camera
  const aspect = parentElement.clientWidth / parentElement.clientHeight;
  const frustumSize = INITIAL_CAMERA_DISTANCE;

  const camera = new OrthographicCamera(
    (-frustumSize * aspect) / 2,
    (frustumSize * aspect) / 2,
    frustumSize / 2,
    -frustumSize / 2,
    1,
    5000
  );

  camera.position.set(0, 0, INITIAL_CAMERA_DISTANCE);
  camera.lookAt(0, 0, 0);

  // Setup controls
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableRotate = false;
  controls.enablePan = true;
  controls.enableZoom = true;
  controls.screenSpacePanning = true;
  controls.minZoom = 0.5;      // Allow zooming out more
  controls.maxZoom = 6;         // Allow closer zoom for detail
  controls.enableDamping = true;
  controls.dampingFactor = 0.08; // Smoother, more fluid motion
  controls.target.set(0, 0, 0);

  controls.update();

  // Handle resizing
  window.addEventListener("resize", () => {
    const aspect = parentElement.clientWidth / parentElement.clientHeight;
    camera.left = (-frustumSize * aspect) / 2;
    camera.right = (frustumSize * aspect) / 2;
    camera.top = frustumSize / 2;
    camera.bottom = -frustumSize / 2;
    camera.updateProjectionMatrix();
    renderer.setSize(parentElement.clientWidth, parentElement.clientHeight);
  });

  // Node picking setup
  const pickingTarget = new WebGLRenderTarget(
    window.innerWidth,
    window.innerHeight,
    {
      format: RGBAFormat,
      type: UnsignedByteType,
      depthBuffer: true,
      stencilBuffer: false,
    }
  );
  const pickingScene = new Scene();

  function pickNodeIndexFromScene(event: MouseEvent): number {
    pickingScene.add(pickingMesh);

    const pickedNodeIndex = pickNodeIndex(
      event,
      renderer,
      pickingScene,
      camera,
      pickingTarget
    );

    return pickedNodeIndex;
  }

  const pickingMesh = createPickingMesh(
    nodes,
    nodePositionsTexture,
    nodeColors,
    INITIAL_CAMERA_DISTANCE
  );

  renderer.domElement.addEventListener("mousedown", (event) => {
    const pickedNodeIndex = pickNodeIndexFromScene(event);
    console.log("Picked node index: ", pickedNodeIndex);
  });
  // Node picking setup end

  // Adaptive layout iterations based on graph size
  const layoutIterations = isMassiveGraph ? 300 : isLargeGraph ? 500 : 800;
  console.log(`Running ${layoutIterations} layout iterations for ${nodeCount} nodes...`);

  for (let i = 0; i < layoutIterations; i++) {
    graphLayout.step();

    // Progress logging for large graphs
    if (isMassiveGraph && i % 50 === 0) {
      console.log(`Layout progress: ${((i / layoutIterations) * 100).toFixed(0)}%`);
    }
  }
  console.log("Layout complete!");

  let visibleLabels: unknown[] = [];

  // Only create entity type labels for smaller graphs (performance optimization)
  const entityTypeLabels: [string, unknown][] = [];
  if (!isMassiveGraph) {
    for (const node of nodes) {
      if (node.type === "EntityType") {
        const label = createLabel(node.label, config?.fontSize);
        entityTypeLabels.push([node.id, label]);
      }
    }
  }

  // const processingStep = 0;

  // Performance monitoring
  let frameCount = 0;
  let lastFpsUpdate = performance.now();
  let currentFps = 60;

  // Cluster boundaries
  let clusterBoundariesCreated = false;
  const clusterBoundaryMeshes: three.Mesh[] = [];

  function calculateClusterBoundaries(): ClusterInfo[] {
    const clusters: ClusterInfo[] = [];

    nodesByType.forEach((typeNodes, nodeType) => {
      if (typeNodes.length < 3) return; // Skip small clusters

      // Calculate center and radius from actual node positions
      let sumX = 0;
      let sumY = 0;
      typeNodes.forEach(node => {
        const pos = graphLayout.getNodePosition(node.id);
        sumX += pos.x;
        sumY += pos.y;
      });

      const center = {
        x: sumX / typeNodes.length,
        y: sumY / typeNodes.length
      };

      // Calculate radius as max distance from center + padding
      let maxDist = 0;
      typeNodes.forEach(node => {
        const pos = graphLayout.getNodePosition(node.id);
        const dist = Math.sqrt(
          Math.pow(pos.x - center.x, 2) + Math.pow(pos.y - center.y, 2)
        );
        maxDist = Math.max(maxDist, dist);
      });

      clusters.push({
        center,
        radius: maxDist + 350, // Apple style: more spacing between clusters
        color: getColorForType(nodeType)
      });
    });

    return clusters;
  }

  // Render loop
  function render() {
    // Adaptive physics updates - skip for large graphs after stabilization
    if (!isMassiveGraph || frameCount < 100) {
      graphLayout.step();
    } else if (frameCount % 2 === 0) {
      // Update physics every other frame for massive graphs
      graphLayout.step();
    }

    // FPS monitoring
    frameCount++;
    const now = performance.now();
    if (now - lastFpsUpdate > 1000) {
      currentFps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
      frameCount = 0;
      lastFpsUpdate = now;
      if (isMassiveGraph && currentFps < 30) {
        console.warn(`Low FPS detected: ${currentFps} fps`);
      }
    }

    controls.update();

    // Create cluster boundaries after layout stabilizes
    if (!clusterBoundariesCreated && frameCount === 60) {
      const clusters = calculateClusterBoundaries();
      clusters.forEach(cluster => {
        const mesh = createClusterBoundaryMesh(cluster);
        clusterBoundaryMeshes.push(mesh);
        scene.add(mesh);
      });
      clusterBoundariesCreated = true;
    }

    updateNodePositions(
      nodes,
      graphLayout,
      nodePositionsData,
      nodePositionsTexture
    );
    const textScale = Math.max(1, 4 / camera.zoom);

    nodeSwarmMesh.material.uniforms.camDist.value = Math.floor(
      camera.zoom * 500
    );
    nodeSwarmMesh.material.uniforms.mousePos.value.set(
      mousePosition.x,
      mousePosition.y
    );
    // @ts-expect-error uniforms does exist on material
    edgeMesh.material.uniforms.camDist.value = Math.floor(camera.zoom * 500);
    // @ts-expect-error uniforms does exist on material
    edgeMesh.material.uniforms.mousePos.value.set(
      mousePosition.x,
      mousePosition.y
    );

    // @ts-expect-error uniforms does exist on material
    pickingMesh.material.uniforms.camDist.value = Math.floor(camera.zoom * 500);

    // Zoom-level semantics: determine what to show based on zoom
    let zoomLevel: "far" | "mid" | "near" = "mid";
    if (camera.zoom < 1.0) {
      zoomLevel = "far"; // Show clusters, domains, density
    } else if (camera.zoom > 3.0) {
      zoomLevel = "near"; // Show applications, paths, labels
    }

    edgeMesh.renderOrder = 1;
    nodeSwarmMesh.renderOrder = 2;

    // IMPROVEMENT #8: Conditional layer rendering based on config and zoom
    const showEdges = config?.showEdges !== false && zoomLevel !== "far"; // Hide edges when far
    const showNodes = config?.showNodes !== false; // Always show nodes
    const showMetaballs = config?.showMetaballs !== false && zoomLevel === "far"; // Only show at far zoom

    // Path filtering based on hover
    const pathFilterMode = config?.pathFilterMode || "all";
    const shouldShowPath = pathFilterMode === "all" || (pathFilterMode === "hoverOnly" && pickedNodeIndex >= 0);

    if (showEdges) {
      scene.add(edgeMesh);
    }
    if (showNodes) {
      scene.add(nodeSwarmMesh);
    }

    // Metaball rendering - reduce frequency for massive graphs
    const shouldRenderMetaballs = showMetaballs && (!isMassiveGraph || frameCount % 2 === 0);

    if (shouldRenderMetaballs) {
      // Pass 1: draw points into density texture
      renderer.setRenderTarget(densityCloudTarget);
      renderer.clear();
      densityCloudScene.clear();
      densityCloudScene.add(densityAccumulatorMesh);
      renderer.render(densityCloudScene, camera);

      // Pass 2: render density map to screen
      renderer.setRenderTarget(null);
      renderer.clear();
      metaballMesh.renderOrder = 0;
      scene.add(metaballMesh);
    } else {
      renderer.setRenderTarget(null);
      renderer.clear();
    }

    for (const [nodeId, label] of entityTypeLabels) {
      const nodePosition = graphLayout.getNodePosition(nodeId);
      // @ts-expect-error label is Text from troika-three-text
      label.position.set(nodePosition.x, nodePosition.y, 1.0);
      // @ts-expect-error label is Text from troika-three-text
      label.scale.setScalar(textScale);
      // @ts-expect-error label is Text from troika-three-text
      scene.add(label);
    }

    if (pickedNodeIndex >= 0) {
      if (pickedNodeIndex !== lastPickedNodeIndex) {
        for (const label of visibleLabels) {
          // @ts-expect-error label is Text from troika-three-text
          label.visible = false;
        }
        visibleLabels = [];
      }

      const pickedNode = nodes[pickedNodeIndex];

      parentElement.style.cursor = "pointer";

      const pickedNodePosition = graphLayout.getNodePosition(pickedNode.id);

      let pickedNodeLabel = nodeLabelMap.get(pickedNode.id);
      if (!pickedNodeLabel) {
        pickedNodeLabel = createLabel(pickedNode.label, config?.fontSize);
        nodeLabelMap.set(pickedNode.id, pickedNodeLabel);
      }
      pickedNodeLabel.position.set(
        pickedNodePosition.x,
        pickedNodePosition.y,
        1.0
      );
      pickedNodeLabel.scale.setScalar(textScale);

      // Adaptive label display based on graph size and zoom
      const minZoomForLabels = isMassiveGraph ? 4 : isLargeGraph ? 3 : 2;
      const maxLabels = isMassiveGraph ? 5 : isLargeGraph ? 10 : 15;

      if (camera.zoom > minZoomForLabels) {
        graph.forEachLinkedNode(
          pickedNode.id,
          (otherNode: GraphNode, edge: GraphLink) => {
            if (visibleLabels.length > maxLabels) {
              return;
            }

            let otherNodeLabel = nodeLabelMap.get(otherNode.id);
            if (!otherNodeLabel) {
              otherNodeLabel = createLabel(otherNode.data.label, config?.fontSize);
              nodeLabelMap.set(otherNode.id, otherNodeLabel);
            }
            const otherNodePosition = graphLayout.getNodePosition(otherNode.id);
            otherNodeLabel.position.set(
              otherNodePosition.x,
              otherNodePosition.y,
              1.0
            );

            let linkLabel = edgeLabelMap.get(edge.id);
            if (!linkLabel) {
              linkLabel = createLabel(edge.data.label, config?.fontSize);
              edgeLabelMap.set(edge.id, linkLabel);
            }
            const linkPosition = graphLayout.getLinkPosition(edge.id);
            const middleLinkPosition = new Vector2(
              (linkPosition.from.x + linkPosition.to.x) / 2,
              (linkPosition.from.y + linkPosition.to.y) / 2
            );
            linkLabel.position.set(
              middleLinkPosition.x,
              middleLinkPosition.y,
              1.0
            );

            linkLabel.visible = true;
            linkLabel.scale.setScalar(textScale);
            visibleLabels.push(linkLabel);
            otherNodeLabel.visible = true;
            otherNodeLabel.scale.setScalar(textScale);
            visibleLabels.push(otherNodeLabel);

            scene.add(linkLabel);
            scene.add(otherNodeLabel);
          }
        );
      }

      pickedNodeLabel.visible = true;
      visibleLabels.push(pickedNodeLabel);

      scene.add(pickedNodeLabel);
    } else {
      parentElement.style.cursor = "default";

      for (const label of visibleLabels) {
        // @ts-expect-error label is Text from troika-three-text
        label.visible = false;
      }

      visibleLabels = [];
    }

    renderer.render(scene, camera);

    animationFrameId = requestAnimationFrame(render);
  }

  let animationFrameId: number;
  render();

  // Return cleanup function
  return () => {
    if (animationFrameId) {
      cancelAnimationFrame(animationFrameId);
    }
    // Clean up cluster boundaries
    clusterBoundaryMeshes.forEach(mesh => {
      scene.remove(mesh);
      mesh.geometry.dispose();
      if (mesh.material instanceof three.Material) {
        mesh.material.dispose();
      }
    });
    graphLayout.dispose();
    renderer.dispose();
    controls.dispose();
  };
}

function updateNodePositions(
  nodes: Node[],
  graphLayout: Layout<Graph>,
  nodePositionsData: Float32Array,
  nodePositionsTexture: DataTexture
) {
  for (let i = 0; i < nodes.length; i++) {
    const p = graphLayout.getNodePosition(nodes[i].id);
    nodePositionsData[i * 4 + 0] = p.x;
    nodePositionsData[i * 4 + 1] = p.y;
    nodePositionsData[i * 4 + 2] = 0.0;
    nodePositionsData[i * 4 + 3] = 1.0;
  }
  nodePositionsTexture.needsUpdate = true;
}
