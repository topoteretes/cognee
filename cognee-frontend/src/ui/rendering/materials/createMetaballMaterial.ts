import { ShaderMaterial, Texture } from "three";

export function createMetaballMaterial(fieldTexture: Texture) {
  return new ShaderMaterial({
    transparent: true,
    uniforms: {
      fieldTex: { value: fieldTexture },
      threshold: { value: 25000.0 },
      smoothing: { value: 5000.0 },
    },
    vertexShader: `
      varying vec2 vUv;

      void main() {
        vUv = uv;
        gl_Position = vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;

      varying vec2 vUv;
      uniform float threshold;
      uniform float smoothing;
      uniform sampler2D fieldTex;

      void main() {
        vec4 fieldData = texture2D(fieldTex, vUv);
        vec3 accumulatedColor = fieldData.rgb;
        float totalInfluence = fieldData.a;

        vec3 finalColor = vec3(0.0);

        if (totalInfluence > 0.0) {
            finalColor = accumulatedColor / totalInfluence;
        }

        // Apple embedding atlas style: very subtle density clouds
        float alphaEdge = smoothstep(threshold - smoothing, threshold + smoothing, totalInfluence);
        float alpha = alphaEdge * 0.08; // Very subtle for clean Apple aesthetic

        if (alpha < 0.01) {
          discard;
        }

        gl_FragColor = vec4(finalColor, alpha);
      }
    `,
  });
}
