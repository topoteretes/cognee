import { resolve } from 'node:path';

// Turbopack only accepts an absolute path for a local PostCSS plugin.
// Its PostCSS worker runs in the app root; import.meta.url points into .next.
const mantineControlFonts = resolve(
  process.cwd(),
  'postcss/mantineLeavesControlFontsToTailwind.cjs',
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
