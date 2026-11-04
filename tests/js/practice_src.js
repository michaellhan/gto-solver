// practice.js as the page loads it (loadPractice): explain.js first, whose hand strength and words it reads. The tests
// of those two modules read it here, so the page's harness (page.js) doesn't need either file.
"use strict";
const fs = require("node:fs");
const path = require("node:path");
const { HTML_PATH } = require("./page.js");

const PRACTICE = ["explain.js", "practice.js"].map(f => fs.readFileSync(path.join(path.dirname(HTML_PATH), f), "utf8")).join("\n");
module.exports = { PRACTICE };
