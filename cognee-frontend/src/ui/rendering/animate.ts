import { Graph, Node as GraphNode, Link as GraphLink } from "ngraph.graph";
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

const INITIAL_CAMERA_DISTANCE = 2000;

interface Config {
  fontSize: number;
}

export default function animate(
  nodes: Node[],
  edges: Edge[],
  parentElement: HTMLElement,
  config?: Config
): void {
  const nodeLabelMap = new Map();
  const edgeLabelMap = new Map();
  // Enhanced color palette with vibrant, distinguishable colors
  const colorPalette = [
    new Color("#5C10F4"), // Deep Purple - Primary concepts
    new Color("#A550FF"), // Light Purple - Algorithms
    new Color("#0DFF00"), // Neon Green - Architectures
    new Color("#00D9FF"), // Cyan - Technologies
    new Color("#FF6B35"), // Coral - Applications
    new Color("#F7B801"), // Golden Yellow - Data
    new Color("#FF1E56"), // Hot Pink - Optimization
    new Color("#00E5FF"), // Bright Cyan - Additional
    new Color("#7DFF8C"), // Mint Green - Additional
    new Color("#FFB347"), // Peach - Additional
  ];
  let lastColorIndex = 0;
  const colorPerType = new Map();

  function getColorForType(nodeType: string): Color {
    if (colorPerType.has(nodeType)) {
      return colorPerType.get(nodeType);
    }

    const color = colorPalette[lastColorIndex % colorPalette.length];
    colorPerType.set(nodeType, color);
    lastColorIndex += 1;

    return color;
  }

  const mousePosition = new Vector2();

  // Node related data
  const nodeColors = new Float32Array(nodes.length * 3);
  const nodeIndices = new Map();
  const textureSize = Math.ceil(Math.sqrt(nodes.length));
  const nodePositionsData = new Float32Array(textureSize * textureSize * 4);

  let nodeIndex = 0;
  function forNode(node: Node) {
    const color = getColorForType(node.type);
    nodeColors[nodeIndex * 3 + 0] = color.r;
    nodeColors[nodeIndex * 3 + 1] = color.g;
    nodeColors[nodeIndex * 3 + 2] = color.b;

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

  // Improved layout parameters for better visualization
  const graphLayout = createForceLayout(graph, {
    dragCoefficient: 0.8,     // Reduced for smoother movement
    springLength: 180,         // Slightly tighter clustering
    springCoefficient: 0.25,   // Stronger connections
    gravity: -1200,            // Stronger repulsion for better spread
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
    INITIAL_CAMERA_DISTANCE
  );

  const edgeMesh = createEdgeMesh(
    edges,
    nodePositionsTexture,
    edgeIndices,
    INITIAL_CAMERA_DISTANCE
  );

  // Density cloud setup
  const densityCloudScene = new Scene();
  const densityCloudTarget = createDensityRenderTarget(512);

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

  const scene = new Scene();
  // Darker background for better contrast with vibrant colors
  scene.background = new Color("#0a0a0f");
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

  // Setup scene - More layout iterations for better initial positioning
  for (let i = 0; i < 800; i++) {
    graphLayout.step();
  }

  let visibleLabels: unknown[] = [];

  const entityTypeLabels: [string, unknown][] = [];
  for (const node of nodes) {
    if (node.type === "EntityType") {
      const label = createLabel(node.label, config?.fontSize);
      entityTypeLabels.push([node.id, label]);
    }
  }

  // const processingStep = 0;

  // Render loop
  function render() {
    graphLayout.step();

    controls.update();

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

    edgeMesh.renderOrder = 1;
    nodeSwarmMesh.renderOrder = 2;

    scene.add(edgeMesh);
    scene.add(nodeSwarmMesh);

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

      if (camera.zoom > 2) {
        graph.forEachLinkedNode(
          pickedNode.id,
          (otherNode: GraphNode, edge: GraphLink) => {
            // Show more labels when zoomed in further
            if (visibleLabels.length > 15) {
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

    requestAnimationFrame(render);
  }

  render();
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
