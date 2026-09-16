// ESLint 9 flat config. `eslint-config-next` v16 ships its presets as flat-config
// arrays already, so they're spread in directly -- no FlatCompat shim needed.
import nextCoreWebVitals from "eslint-config-next/core-web-vitals";
import nextTypescript from "eslint-config-next/typescript";

const config = [
  {
    ignores: [".next/**", "out/**", "build/**", "node_modules/**", "next-env.d.ts"],
  },
  ...nextCoreWebVitals,
  ...nextTypescript,
];

export default config;
