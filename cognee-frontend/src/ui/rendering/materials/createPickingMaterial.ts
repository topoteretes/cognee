import * as three from "three";

export default function createPickingMaterial(
  nodePositionsTexture: three.DataTexture,
  initialCameraDistance: number
) {
  const pickingMaterial = new three.ShaderMaterial({
    depthTest: true,
    depthWrite: true,
    transparent: false,
    blending: three.NoBlending,
    uniforms: {
      nodePosTex: { value: nodePositionsTexture },
      textureSize: { value: nodePositionsTexture.image.width },
      camDist: { value: initialCameraDistance },
    },
    vertexShader: `
      precision highp float;

      uniform sampler2D nodePosTex;
      uniform float textureSize;
      uniform float camDist;
      varying vec3 vColor;

      vec3 getNodePos(float idx) {
        float size = textureSize;
        float fx = mod(idx, size);
        float fy = floor(idx / size);
        vec2 uv = (vec2(fx, fy) + 0.5) / size;
        return texture2D(nodePosTex, uv).xyz;
      }

      void main() {
        float id = float(gl_InstanceID);
        vec3 nodePos = getNodePos(id);
        vColor = vec3(
          mod(id, 256.0) / 255.0,
          mod(floor(id / 256.0), 256.0) / 255.0,
          floor(id / 65536.0) / 255.0
        );
        // vColor = vec3(fract(sin(id * 12.9898) * 43758.5453));

        float baseNodeSize = 4.0;

        // Normalize camera distance into [0,1]
        float t = clamp((camDist - 500.0) / (2000.0 - 500.0), 0.0, 1.0);
        float nodeSize = baseNodeSize * mix(1.0, 2.0, t);

        vec3 transformed = nodePos + position * nodeSize;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(transformed, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;

      varying vec3 vColor;

      void main() {
        gl_FragColor = vec4(vColor, 1.0);
      }
  `,
  });

  return pickingMaterial;
}
