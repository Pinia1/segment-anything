import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import { resolve } from "path";
import copy from "rollup-plugin-copy";
// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
});
