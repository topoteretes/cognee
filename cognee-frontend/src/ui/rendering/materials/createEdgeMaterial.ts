import * as three from "three";

export default function createEdgeMaterial(
  texture: three.DataTexture,
  initialCameraDistance: number
): three.ShaderMaterial {
  const material = new three.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: three.AdditiveBlending,
    uniforms: {
      nodePosTex: { value: texture },
      textureSize: { value: texture.image.width },
      camDist: { value: initialCameraDistance },
      mousePos: { value: new three.Vector2(9999, 9999) }, // start offscreen
      color: { value: new three.Color(0xffffff) },
    },
    vertexShader: `
      attribute vec2 edgeIndices;
      uniform sampler2D nodePosTex;
      uniform float textureSize;
      uniform float camDist;
      uniform vec2 mousePos;

      varying float vFade;
      varying float vHighlight;

      vec3 getNodePos(float idx) {
        float x = mod(idx, textureSize);
        float y = floor(idx / textureSize);
        vec2 uv = (vec2(x, y) + 0.5) / textureSize;
        return texture2D(nodePosTex, uv).xyz;
      }

      void main() {
        vec3 start = getNodePos(edgeIndices.x);
        vec3 end = getNodePos(edgeIndices.y);
        vec3 nodePos = mix(start, end, position.x);

        // Project world-space position to clip-space
        vec4 clipPos = projectionMatrix * modelViewMatrix * vec4(nodePos, 1.0);
        vec3 ndc = clipPos.xyz / clipPos.w; // normalized device coordinates [-1,1]

        float distanceFromMouse = length(ndc.xy - mousePos);
        vHighlight = smoothstep(0.2, 0.0, distanceFromMouse);

        vFade = smoothstep(500.0, 1500.0, camDist);
        vFade = 0.2 * clamp(vFade, 0.0, 1.0);

        gl_Position = projectionMatrix * modelViewMatrix * vec4(nodePos, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;

      uniform vec3 color;
      varying vec3 vColor;
      varying float vFade;
      varying float vHighlight;

      void main() {
        vec3 finalColor = mix(color, vec3(1.0), vHighlight * 0.8);
        float alpha = mix(vFade, 0.8, vHighlight);
        gl_FragColor = vec4(finalColor, alpha);
      }
    `,
  });

  return material;
}
