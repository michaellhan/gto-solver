// The page's service worker: it makes the page cross-origin isolated, which the solver's threads need
// (SharedArrayBuffer: worker.js's core-mt.wasm and its helpers) and which takes two response headers GitHub Pages can't
// send. It adds them to this site's pages and worker scripts as they come from the network; every other request goes
// to the network untouched (the page's cross-origin loads, Pyodide from jsdelivr, send CORP or are CORS requests, as
// COEP's require-corp asks). index.html registers it and, the first time, loads once more under it. Its scope is
// the folder it's served from (gto-solver/).
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));
self.addEventListener("fetch", (e) => {
  const r = e.request;
  if (r.mode !== "navigate" && r.destination !== "worker") return;
  if (new URL(r.url).origin !== self.location.origin) return;
  e.respondWith(fetch(r).then((res) => {
    if (res.status === 0) return res;   // (a redirect to follow: as it came)
    const h = new Headers(res.headers);
    h.set("Cross-Origin-Opener-Policy", "same-origin");
    h.set("Cross-Origin-Embedder-Policy", "require-corp");
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
  }));
});
