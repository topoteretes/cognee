// Mantine's baseline sets `font: inherit` on form controls, which beats
// Tailwind font utilities (text-sm, font-medium) on buttons and inputs, so
// design-system buttons would render at the wrong size and weight. Removing
// that one declaration lets the utilities apply; controls still inherit
// their font through Tailwind's own preflight (`font: inherit`), which sits
// in the lower layer and now wins.
const CONTROLS = "input,button,textarea,select";

module.exports = () => ({
  postcssPlugin: "mantine-leaves-control-fonts-to-tailwind",
  Rule(rule) {
    if (rule.selector.replace(/\s+/g, "") !== CONTROLS) return;
    rule.walkDecls("font", (declaration) => {
      if (declaration.value === "inherit") declaration.remove();
    });
    if (rule.nodes?.length === 0) rule.remove();
  },
});

module.exports.postcss = true;
