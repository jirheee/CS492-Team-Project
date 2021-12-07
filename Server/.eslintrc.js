// .eslintrc.js
module.exports = {
  env: {
    node: true,
    es6: true
  },
  extends: ['airbnb-base', 'eslint:recommended', 'plugin:prettier/recommended'],
  globals: {
    Atomics: 'readonly',
    SharedArrayBuffer: 'readonly'
  },
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2018,
    sourceType: 'module'
  },
  plugins: ['@typescript-eslint', 'prettier'],
  rules: {
    'prettier/prettier': 'error',
    'arrow-body-style': 'off',
    'prefer-arrow-callback': 'off',
    'no-shadow': 'off',
    'no-unused-vars': 'off',
    '@typescript-eslint/no-shadow': ['error'],
    '@typescript-eslint/no-unused-vars': ['error'],
    'comma-dangle': ['error', 'never'],
    'import/extensions': ['off'],
    'import/no-unresolved': 'error'
  },
  settings: {
    'import/parsers': { '@typescript-eslint/parser': ['.ts', '.tsx'] },
    'import/resolver': {
      // 아래부터는 자기 코드의 상황에 맞는 설정을 골라서 추가하면 된다.
      // 1. use <root>/tsconfig.json
      typescript: {
        alwaysTryTypes: true // always try to resolve types under `<roo/>@types` directory even it doesn't contain any source code, like `@types/unist`
      }
      // 2. use <root>/path/to/folder/tsconfig.json
      // typescript: { directory: './' }
    }
  }
};
