import {
  Mesh,
  DataTexture,
  CircleGeometry,
  InstancedBufferGeometry,
  InstancedBufferAttribute,
} from "three";
import { Node } from "../graph/types";
import createPickingMaterial from "../materials/createPickingMaterial";

export default function createPickingMesh(
  nodes: Node[],
  nodePositionsTexture: DataTexture,
  nodeColors: Float32Array,
  initialCameraDistance: number
): Mesh {
  const nodeGeom = new CircleGeometry(2, 16);
  const geom = new InstancedBufferGeometry();
  geom.index = nodeGeom.index;
  geom.instanceCount = nodes.length;

  geom.setAttribute("position", nodeGeom.attributes.position);
  geom.setAttribute("uv", nodeGeom.attributes.uv);
  geom.setAttribute("nodeColor", new InstancedBufferAttribute(nodeColors, 3));

  const pickingMaterial = createPickingMaterial(
    nodePositionsTexture,
    initialCameraDistance
  );

  const pickingMesh = new Mesh(geom, pickingMaterial);
  pickingMesh.frustumCulled = false;

  return pickingMesh;
}
