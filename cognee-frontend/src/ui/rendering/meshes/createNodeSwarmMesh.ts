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
  initialCameraDistance: number
) {
  const nodeGeom = new CircleGeometry(2, 16);
  const geom = new InstancedBufferGeometry();
  geom.index = nodeGeom.index;
  geom.instanceCount = nodes.length;

  geom.setAttribute("position", nodeGeom.attributes.position);
  geom.setAttribute("uv", nodeGeom.attributes.uv);
  geom.setAttribute("nodeColor", new InstancedBufferAttribute(nodeColors, 3));

  const material = createNodeSwarmMaterial(
    nodePositionsTexture,
    initialCameraDistance
  );

  const nodeSwarmMesh = new Mesh(geom, material);
  nodeSwarmMesh.frustumCulled = false;

  return nodeSwarmMesh;
}
