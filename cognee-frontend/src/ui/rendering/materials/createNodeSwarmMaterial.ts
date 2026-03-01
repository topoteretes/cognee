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
      attribute float nodeSize; // Hierarchy-based size multiplier
      attribute float nodeHighlight; // Selection-based highlight (1.0 = selected, 0.3 = dimmed)
      varying vec3 vColor;
      varying float vHighlight;
      varying float vSelectionHighlight;
      varying vec2 vUv; // IMPROVEMENT #4: For radial halo effect

      vec3 getNodePos(float idx) {
        float size = textureSize;
        float fx = mod(idx, size);
        float fy = floor(idx / size);
        vec2 uv = (vec2(fx, fy) + 0.5) / size;
        return texture2D(nodePosTex, uv).xyz;
      }

      void main() {
        vColor = nodeColor;
        vSelectionHighlight = nodeHighlight;
        vec3 nodePos = getNodePos(float(gl_InstanceID));

        // IMPROVEMENT #4: Pass UV coordinates for halo effect
        vUv = position.xy * 0.5 + 0.5; // Convert from [-1,1] to [0,1]

        // Project world-space position to clip-space
        vec4 clipPos = projectionMatrix * modelViewMatrix * vec4(nodePos, 1.0);
        vec3 ndc = clipPos.xyz / clipPos.w; // normalized device coordinates [-1,1]

        float distanceFromMouse = length(ndc.xy - mousePos);
        vHighlight = smoothstep(0.2, 0.0, distanceFromMouse);

        // Hierarchy-based sizing: base size * type size multiplier
        float baseNodeSize = 7.0;

        // Normalize camera distance into [0,1]
        float t = clamp((camDist - 500.0) / (2000.0 - 500.0), 0.0, 1.0);
        float finalSize = baseNodeSize * nodeSize * mix(1.0, 1.2, t); // Apply hierarchy multiplier

        vec3 transformed = nodePos + position * finalSize;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(transformed, 1.0);
      }
    `,
    fragmentShader: `
      precision highp float;

      varying vec3 vColor;
      varying float vHighlight;
      varying float vSelectionHighlight;
      varying vec2 vUv; // IMPROVEMENT #4: For radial halo effect

      void main() {
        // Apple embedding atlas style: subtle radial glow
        vec2 center = vec2(0.5, 0.5);
        float distFromCenter = length(vUv - center) * 2.0;

        // Create sharp node with very subtle glow
        float coreRadius = 0.75; // Slightly larger core
        float haloRadius = 1.0;

        // Core node (solid)
        float core = 1.0 - smoothstep(0.0, coreRadius, distFromCenter);

        // Very subtle outer glow (Apple aesthetic)
        float halo = smoothstep(haloRadius, coreRadius, distFromCenter);

        // Subtle color mixing
        vec3 haloColor = vColor * 1.15; // Subtle brightness increase
        vec3 baseColor = mix(vColor, vec3(1.0), vHighlight * 0.4);
        vec3 finalColor = mix(haloColor, baseColor, core);

        // Alpha with subtle glow
        float alpha = mix(halo * 0.4, 1.0, core); // Reduced halo opacity

        // Apply selection-based dimming (neutral-by-default)
        alpha *= vSelectionHighlight;

        gl_FragColor = vec4(finalColor, alpha);
      }
    `,
  });

  return material;
}
