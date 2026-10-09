import { MantineThemeOverride } from "@mantine/core";

const typography: MantineThemeOverride = {
  fontFamily: '"TWKLausanne", sans-serif',
  fontFamilyMonospace:
    'ui-monospace, Menlo, Monaco, "Cascadia Mono", "Segoe UI Mono", "Roboto Mono", monospace',
  fontSizes: {
    xl: "1.5rem",
    lg: "1.25rem",
    md: "1rem",
    sm: "0.875rem",
    xs: "0.75rem",
  },
  lineHeights: {
    xl: "1.25",
    lg: "1.25",
    md: "1.25",
    sm: "1.25",
    xs: "1.25",
  },
  autoContrast: true,
  headings: {
    fontWeight: "700",
    sizes: {
      h1: { fontSize: "1.75rem", lineHeight: "2.25rem" },
      h2: { fontSize: "1.5rem", lineHeight: "2rem" },
      h3: { fontSize: "1.25rem", lineHeight: "1.75rem" },
      h4: { fontSize: "1.125rem", lineHeight: "1.5rem" },
      h5: { fontSize: "1rem", lineHeight: "1.375rem" },
      h6: { fontSize: "0.875rem", lineHeight: "1.25rem" },
    },
  },
};

export default typography;
