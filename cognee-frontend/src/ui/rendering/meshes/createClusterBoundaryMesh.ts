import * as three from "three";
import createClusterBoundaryMaterial from "../materials/createClusterBoundaryMaterial";

export interface ClusterInfo {
  center: { x: number; y: number };
  radius: number;
  color: three.Color;
}

export default function createClusterBoundaryMesh(
  cluster: ClusterInfo
): three.Mesh {
  // Create a circle geometry for the cluster boundary
  const geometry = new three.PlaneGeometry(
    cluster.radius * 2.5, // Make it larger to encompass the cluster
    cluster.radius * 2.5
  );

  const material = createClusterBoundaryMaterial(cluster.color);
  const mesh = new three.Mesh(geometry, material);

  // Position the mesh at the cluster center
  mesh.position.set(cluster.center.x, cluster.center.y, -100); // Behind everything else
  mesh.renderOrder = -1;

  return mesh;
}
