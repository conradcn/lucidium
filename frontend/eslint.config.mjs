// ESLint flat config. Replaces the old ``.eslintrc.cjs``: ESLint 9
// defaults to flat config and ESLint 10 drops eslintrc support
// entirely, so the rc file no longer applies.
//
// ``.mjs`` rather than ``.js``: frontend/package.json deliberately has
// no ``"type": "module"`` (the Electron main process loads
// ``dist-electron/main.js`` as CommonJS — see tsconfig.electron.json),
// so a plain ``.js`` config here would be parsed as CommonJS and the
// ESM syntax below would fail.
import js from "@eslint/js";
import react from "eslint-plugin-react";
import reactHooks from "eslint-plugin-react-hooks";
import prettier from "eslint-config-prettier";
import globals from "globals";
import tseslint from "typescript-eslint";

export default [
  {
    // ``release`` is electron-builder's output. It contains the unpacked
    // app plus the whole PyInstaller backend, and on Linux-unpacked
    // output eslint's directory walk dies with EACCES on the bundled
    // ``.so`` files — so ``npm run lint`` fails for anyone who has run a
    // package locally, with an error that looks nothing like a lint error.
    ignores: [
      "dist/**",
      "dist-electron/**",
      "node_modules/**",
      "release/**",
      "src/shared/generated/**",
    ],
  },

  js.configs.recommended,
  ...tseslint.configs.recommended,
  react.configs.flat.recommended,
  // Turns off ``react/react-in-jsx-scope``: the automatic JSX runtime
  // (tsconfig ``"jsx": "react-jsx"``) means React need not be in scope.
  react.configs.flat["jsx-runtime"],
  reactHooks.configs.flat["recommended-latest"],

  {
    files: ["**/*.{ts,tsx,js,jsx,mjs,cjs}"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: { ...globals.browser, ...globals.node },
    },
    settings: { react: { version: "detect" } },
    rules: {
      // ``^_`` is the project-wide "deliberately unused" convention: it
      // covers unused parameters, unused bindings (``for (const _step of
      // steps)`` loops that only need the iteration count) and caught
      // errors that are intentionally swallowed.
      "@typescript-eslint/no-unused-vars": [
        "warn",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],

      // Prose-heavy UI copy: apostrophes in ordinary English sentences
      // ("you're", "don't", "the guide's") are not a defect, and JSX
      // renders them correctly. Off permanently, not triage.
      "react/no-unescaped-entities": "off",
    },
  },

  // The TRIAGE BASELINE that used to live here — demoting
  // ``react-hooks/rules-of-hooks``, ``set-state-in-effect``, ``refs``,
  // ``globals``, ``no-explicit-any`` and ``no-this-alias`` to warnings
  // while the flat-config backlog was worked off — is GONE, because
  // the backlog is empty. Every one of those findings is fixed, so the
  // rules run at their plugin default severity (error) and a
  // regression fails the build instead of scrolling past in a warning
  // list. Do not re-add a blanket demotion; suppress a genuinely
  // intentional site with a targeted, commented
  // ``eslint-disable-next-line`` instead.

  // Must stay last: switches off every rule that would fight prettier.
  prettier,
];
