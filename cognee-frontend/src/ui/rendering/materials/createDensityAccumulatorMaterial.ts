import { AdditiveBlending, DataTexture, ShaderMaterial } from "three";

export default function createDensityAccumulatorMaterial(
  nodePositionsTexture: DataTexture,
  initialCameraDistance: number
) {
  const densityCloudMaterial = new ShaderMaterial({
    depthWrite: false,
    depthTest: false,
    transparent: true,
    blending: AdditiveBlending,
    uniforms: {
      nodePositionsTexture: {
        value: nodePositionsTexture,
      },
      textureSize: {
        value: nodePositionsTexture.image.width,
      },
      camDist: {
        value: initialCameraDistance,
      },
      radius: { value: 0.05 },
    },
    vertexShader: `
      uniform sampler2D nodePositionsTexture;
      uniform float textureSize;
      uniform float camDist;
      attribute vec3 nodeColor;
      varying vec3 vColor;
      varying vec2 vUv;
      varying float nodeSize;

      vec3 getNodePos(float idx) {
        float fx = mod(idx, textureSize);
        float fy = floor(idx / textureSize);
        vec2 uv = (vec2(fx, fy) + 0.5) / textureSize;
        return texture2D(nodePositionsTexture, uv).xyz;
      }

      void main() {
        vUv = uv;
        vColor = nodeColor;
        vec3 nodePos = getNodePos(float(gl_InstanceID));

        float baseNodeSize = 8.0;

        // Normalize camera distance into [0,1]
        float t = clamp((camDist - 500.0) / (2000.0 - 500.0), 0.0, 1.0);
        nodeSize = baseNodeSize * mix(10.0, 12.0, t);

        vec3 transformed = nodePos + position * nodeSize;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(transformed, 1.0);
      }
  `,
    fragmentShader: `
      precision highp float;

      varying vec2 vUv;
      varying float nodeSize;
      varying vec3 vColor;

      void main() {
        vec2 pCoord = vUv - 0.5;
        float distSq = dot(pCoord, pCoord) * 4.0;

        if (distSq > 1.0) {
          discard;
        }

        float radiusSq = (nodeSize / 2.0) * (nodeSize / 2.0);
        float falloff = max(0.0, 1.0 - distSq);
        float influence = radiusSq * falloff * falloff;
        vec3 accumulatedColor = vColor * influence;

        gl_FragColor = vec4(accumulatedColor, influence);
      }
  `,
  });

  return densityCloudMaterial;
}
