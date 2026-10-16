# Jiaoyang Ruan

Personal homepage, served at https://ruanjiaoyang.github.io

## Publishing a blog post

1. Copy the post's Markdown (with its front matter) into `content/<series>/`, e.g. `content/generative-models/02-vae.md`.
2. Run `python3 tools/build_blog.py` (needs `pip install markdown pyyaml`).
   It writes `blog/<series>/<slug>/index.html` and refreshes the Blog list in `index.html`.
3. Commit and push. Math is rendered in the browser with KaTeX (`$...$` inline, `$$...$$` display).
