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
      // Apple embedding atlas style: soft pastel edges
      color: { value: new three.Color("#FCD34D") }, // Soft amber for minimalist aesthetic
    },
    vertexShader: `
      attribute vec2 edgeIndices;
      uniform sampler2D nodePosTex;
      uniform float textureSize;
      uniform float camDist;
      uniform vec2 mousePos;

      varying float vFade;
      varying float vHighlight;
      varying float vEdgePosition; // IMPROVEMENT #2: For directional gradient

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

        // IMPROVEMENT #2: Pass edge position for gradient
        vEdgePosition = position.x;

        // Project world-space position to clip-space
        vec4 clipPos = projectionMatrix * modelViewMatrix * vec4(nodePos, 1.0);
        vec3 ndc = clipPos.xyz / clipPos.w; // normalized device coordinates [-1,1]

        float distanceFromMouse = length(ndc.xy - mousePos);
        vHighlight = smoothstep(0.2, 0.0, distanceFromMouse);

        // Apple embedding atlas style: subtle edge opacity
        vFade = smoothstep(500.0, 1500.0, camDist);
        vFade = 0.25 * clamp(vFade, 0.0, 1.0); // Subtle for clean aesthetic

        gl_Position = projectionMatrix * modelViewMatrix * vec4(nodePos, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;

      uniform vec3 color;
      varying float vFade;
      varying float vHighlight;
      varying float vEdgePosition; // IMPROVEMENT #2: For directional gradient

      void main() {
        // IMPROVEMENT #2: Directional gradient from start to end
        // Brighter at start, slightly darker at end for flow direction
        float gradientFactor = 1.0 - (vEdgePosition * 0.3); // 30% dimming from start to end

        // IMPROVEMENT #2: Add subtle glow effect
        vec3 glowColor = vec3(1.0, 0.9, 0.7); // Warm white glow
        vec3 baseColor = color * gradientFactor;
        vec3 finalColor = mix(baseColor, glowColor, vHighlight * 0.9);

        // IMPROVEMENT #2: Increased visibility and glow
        float baseAlpha = vFade * 1.5; // Increased visibility
        float alpha = mix(baseAlpha, 0.95, vHighlight); // Stronger highlight

        gl_FragColor = vec4(finalColor, alpha);
      }
    `,
  });

  return material;
}
