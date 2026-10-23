// The Explain with AI proxy's entry module (the Workers runtime takes every export of it as an entry point, so the
// constants and helpers live in proxy.js).
import { handler } from "./proxy.js";
export default handler;
