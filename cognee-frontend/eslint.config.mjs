import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  // `next lint` used to apply these ignores implicitly; plain `eslint .`
  // does not, so without this it also lints build output and coverage
  // reports. `scripts/**` is standalone public-repo sync tooling (see
  // scripts/sync-to-public.sh) — not part of the app build, and not
  // something `next lint` ever covered either.
  { ignores: [".next/**", "out/**", "build/**", "coverage/**", "scripts/**"] },
  ...compat.extends("next/core-web-vitals", "next/typescript"),
];

export default eslintConfig;
