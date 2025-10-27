import * as three from "three";

export default function createNodeSwarmMaterial(
  nodePositionsTexture: three.DataTexture,
  initialCameraDistance: number
) {
  const material = new three.ShaderMaterial({
    transparent: true,
    uniforms: {
      nodePosTex: { value: nodePositionsTexture },
      textureSize: { value: nodePositionsTexture.image.width },
      camDist: { value: initialCameraDistance },
      mousePos: { value: new three.Vector2(9999, 9999) }, // start offscreen
    },
    vertexShader: `
      precision highp float;

      uniform sampler2D nodePosTex;
      uniform float textureSize;
      uniform float camDist;
      uniform vec2 mousePos;
      attribute vec3 nodeColor;
      varying vec3 vColor;
      varying float vHighlight;

      vec3 getNodePos(float idx) {
        float size = textureSize;
        float fx = mod(idx, size);
        float fy = floor(idx / size);
        vec2 uv = (vec2(fx, fy) + 0.5) / size;
        return texture2D(nodePosTex, uv).xyz;
      }

      void main() {
        vColor = nodeColor;
        vec3 nodePos = getNodePos(float(gl_InstanceID));

        // Project world-space position to clip-space
        vec4 clipPos = projectionMatrix * modelViewMatrix * vec4(nodePos, 1.0);
        vec3 ndc = clipPos.xyz / clipPos.w; // normalized device coordinates [-1,1]

        float distanceFromMouse = length(ndc.xy - mousePos);
        vHighlight = smoothstep(0.2, 0.0, distanceFromMouse);

        float baseNodeSize = 8.0;

        // Normalize camera distance into [0,1]
        float t = clamp((camDist - 500.0) / (2000.0 - 500.0), 0.0, 1.0);
        float nodeSize = baseNodeSize * mix(1.1, 1.3, t);

        vec3 transformed = nodePos + position * nodeSize;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(transformed, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;

      varying vec3 vColor;
      varying float vHighlight;

      void main() {
        vec3 finalColor = mix(vColor, vec3(1.0), vHighlight * 0.3);
        gl_FragColor = vec4(finalColor, 1.0);
        // gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
      }
    `,
  });

  return material;
}
