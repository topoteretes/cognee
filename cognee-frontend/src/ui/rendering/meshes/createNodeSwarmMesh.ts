import {
  Mesh,
  DataTexture,
  CircleGeometry,
  InstancedBufferAttribute,
  InstancedBufferGeometry,
} from "three";
import { Node } from "../graph/types";
import createNodeSwarmMaterial from "../materials/createNodeSwarmMaterial";

export default function createNodeSwarmMesh(
  nodes: Node[],
  nodePositionsTexture: DataTexture,
  nodeColors: Float32Array,
  nodeSizes: Float32Array,
  nodeHighlights: Float32Array,
  initialCameraDistance: number
) {
  const nodeGeom = new CircleGeometry(2, 16);
  const geom = new InstancedBufferGeometry();
  geom.index = nodeGeom.index;
  geom.instanceCount = nodes.length;

  geom.setAttribute("position", nodeGeom.attributes.position);
  geom.setAttribute("uv", nodeGeom.attributes.uv);
  geom.setAttribute("nodeColor", new InstancedBufferAttribute(nodeColors, 3));
  geom.setAttribute("nodeSize", new InstancedBufferAttribute(nodeSizes, 1));
  geom.setAttribute("nodeHighlight", new InstancedBufferAttribute(nodeHighlights, 1));

  const material = createNodeSwarmMaterial(
    nodePositionsTexture,
    initialCameraDistance
  );

  const nodeSwarmMesh = new Mesh(geom, material);
  nodeSwarmMesh.frustumCulled = false;

  return nodeSwarmMesh;
}
