import { fileURLToPath } from 'node:url';

// Turbopack only accepts an absolute path for a local PostCSS plugin.
const mantineControlFonts = fileURLToPath(
  new URL('./postcss/mantineLeavesControlFontsToTailwind.cjs', import.meta.url),
);

const config = {
  plugins: {
    'postcss-import': {},
    '@tailwindcss/postcss': {},
    'autoprefixer': {},
    'postcss-preset-mantine': {},
    [mantineControlFonts]: {},
    'postcss-simple-vars': {
      variables: {
        'mantine-breakpoint-xs': '36em',
        'mantine-breakpoint-sm': '48em',
        'mantine-breakpoint-md': '62em',
        'mantine-breakpoint-lg': '75em',
        'mantine-breakpoint-xl': '88em',
      },
    },
  },
};

export default config;
