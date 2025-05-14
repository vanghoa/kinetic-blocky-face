import { defineConfig } from 'vite';
import mkcert from 'vite-plugin-mkcert';

export default defineConfig({
    plugins: [mkcert()],
    base: '/kinetic-blocky-face/', // replace with your actual repo name
});
