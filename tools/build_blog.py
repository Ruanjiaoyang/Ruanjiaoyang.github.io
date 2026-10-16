"""Build blog pages from Markdown.

Every content/<series>/<NN-slug>.md becomes blog/<series>/<slug>/index.html,
and the Blog section of index.html (between the BLOG markers) is regenerated.

To publish a new post: copy its .md into content/<series>/ and run
    python3 tools/build_blog.py
Requires: pip install markdown pyyaml
"""
import html
import re
import shutil
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify_unicode
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"
OUT = ROOT / "blog"
INDEX = ROOT / "index.html"
START, END = "<!-- BLOG:START -->", "<!-- BLOG:END -->"

# Series shown on the Blog list, in this order: folder name -> (English name, Chinese name).
# A folder not listed here still publishes, after these, under its front-matter series name.
SERIES = {
    "generative-models": ("From GANs to Diffusion", "从 GAN 到 Diffusion"),
    "diffusion-lm": ("Diffusion Language Models", "扩散语言模型"),
    "transformer": ("Transformer Architecture", "Transformer 架构"),
    "rl": ("Reinforcement Learning for LLMs", "LLM 强化学习"),
    "opd": ("On-Policy Distillation", "On-Policy 蒸馏"),
}


def read_post(path):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    meta = yaml.safe_load(m.group(1)) if m else {}
    body = m.group(2) if m else text
    slug = re.sub(r"^\d+-", "", path.stem)
    meta.update(series_dir=path.parent.name, slug=slug, body=body,
                url=f"blog/{path.parent.name}/{slug}/")
    return meta


def protect_math(md):
    """Swap code and math for placeholders so Markdown leaves them alone."""
    store = []

    def keep(s):
        store.append(s)
        return f"\x00{len(store) - 1}\x00"

    # fenced code and inline code stay as Markdown, but hide them from the math scan
    code = []

    def keep_code(m):
        code.append(m.group(0))
        return f"\x01{len(code) - 1}\x01"

    md = re.sub(r"^```.*?^```[ \t]*$", keep_code, md, flags=re.S | re.M)
    md = re.sub(r"`[^`\n]+`", keep_code, md)
    md = re.sub(r"\$\$(.+?)\$\$", lambda m: keep(m.group(0)), md, flags=re.S)
    md = re.sub(r"(?<![\\$])\$(?!\s)([^\n$]+?)(?<!\s)\$(?!\d)", lambda m: keep(m.group(0)), md)
    md = re.sub(r"\x01(\d+)\x01", lambda m: code[int(m.group(1))], md)
    return md, store


def render(md):
    """Return (body_html, toc_html)."""
    md, store = protect_math(md)
    conv = markdown.Markdown(extensions=["extra", "sane_lists", "toc"],
                             extension_configs={"toc": {"permalink": False, "slugify": slugify_unicode,
                                                        "toc_depth": "2-3"}})
    out = conv.convert(md)
    restore = lambda t: re.sub(r"\x00(\d+)\x00", lambda m: html.escape(store[int(m.group(1))], quote=False), t)
    return restore(out), restore(conv.toc)


def figures(body):
    """An image paragraph followed by an italic paragraph becomes <figure> with <figcaption>.
    Images live in content/<series>/images/ and are copied to blog/<series>/images/."""
    body = body.replace('src="images/', 'src="../images/')
    def fig(m):
        rest = m.group(3).strip()
        return (f'<figure>{m.group(1)} loading="lazy"><figcaption>{m.group(2)}</figcaption></figure>'
                + (f"\n<p>{rest}</p>" if rest else ""))
    # a caption line followed directly by text shares its paragraph; that text becomes its own paragraph
    return re.sub(r'<p>(<img [^>]*?) ?/?></p>\s*<p><em>(.*?)</em>(.*?)</p>', fig, body, flags=re.S)


def reading_minutes(md):
    cjk = len(re.findall(r"[\u4e00-\u9fff]", md))
    words = len(re.findall(r"[A-Za-z]+", md))
    return max(1, round(cjk / 400 + words / 220))


def nice_date(d):
    try:
        return f"{d:%B} {d.day}, {d.year}"
    except (TypeError, ValueError):
        return str(d)


def meta_line(p):
    return (f'Date: {nice_date(p.get("date"))} | Estimated Reading Time: {p["minutes"]} min'
            f' | Author: Jiaoyang Ruan')


def page(p, body_html, toc_html):
    title = html.escape(str(p.get("title", p["slug"])))
    title_en = html.escape(str(p.get("title_en", "")))
    desc = html.escape(" ".join(str(p.get("summary_en", "")).split()))
    tags = "".join(f'<span class="tag">{html.escape(str(t))}</span>' for t in p.get("tags", []))
    series = html.escape(str(p.get("series", "")))
    order = p.get("order")
    kicker = (f"{series} · 系列导读" if order == 0 else f"{series} · 第 {order} 篇") if series and order is not None else series
    toc_block = (f'<details class="toc"><summary>目录 · Table of Contents</summary>{toc_html}</details>'
                 if "<li>" in toc_html else "")
    fields = dict(title=title, title_en=title_en, desc=desc, tags=tags, kicker=kicker,
                  date=str(p.get("date", "")), toc=toc_block, body=body_html)
    out = TEMPLATE
    for k, v in fields.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def entry(p):
    summary = html.escape(" ".join(str(p.get("summary_en", "")).split()))
    title_en = html.escape(str(p.get("title_en") or p.get("title")))
    return f'''        <article class="entry">
          <h4><span lang="en">{title_en}</span><span lang="zh">{html.escape(str(p.get("title")))}</span></h4>
          <p class="entry-sum">{summary}</p>
          <p class="entry-meta">{meta_line(p)}</p>
          <a class="entry-link" href="{p["url"]}" aria-label="{title_en}"></a>
        </article>'''


def blog_list(posts):
    if not posts:
        return '<p class="empty"><span lang="en">Posts coming soon.</span><span lang="zh">文章即将上线。</span></p>'
    groups = {}
    for p in posts:
        groups.setdefault(p["series_dir"], []).append(p)
    order = [d for d in SERIES if d in groups] + sorted(d for d in groups if d not in SERIES)
    out = []
    for d in order:
        items = sorted(groups[d], key=lambda p: p.get("order") or 0)
        fallback = str(items[0].get("series") or d)
        en, zh = (html.escape(x) for x in SERIES.get(d, (fallback, fallback)))
        n = len(items)
        out.append(f'''      <details class="series">
        <summary><h3 class="series-name"><span lang="en">{en}</span><span lang="zh">{zh}</span></h3>
          <span class="series-count"><span lang="en">{n} post{"s" if n != 1 else ""}</span><span lang="zh">{n} 篇</span></span></summary>
        <div class="entries">
{chr(10).join(entry(p) for p in items)}
        </div>
      </details>''')
    return '<div class="series-list">\n' + "\n".join(out) + "\n      </div>"


def main():
    posts = [read_post(f) for f in sorted(CONTENT.glob("*/*.md"))]
    for p in posts:
        p["minutes"] = reading_minutes(p["body"])
    for p in posts:
        dest = OUT / p["series_dir"] / p["slug"]
        dest.mkdir(parents=True, exist_ok=True)
        body_html, toc_html = render(p["body"])
        body_html = figures(body_html)
        (dest / "index.html").write_text(page(p, body_html, toc_html), encoding="utf-8")
        print("built", dest.relative_to(ROOT))
    for images in CONTENT.glob("*/images"):
        shutil.copytree(images, OUT / images.parent.name / "images", dirs_exist_ok=True)
    s = INDEX.read_text(encoding="utf-8")
    i, j = s.index(START) + len(START), s.index(END)
    INDEX.write_text(s[:i] + "\n      " + blog_list(posts) + "\n      " + s[j:], encoding="utf-8")
    print("updated index.html blog list:", len(posts), "post(s)")


TEMPLATE = (Path(__file__).parent / "post_template.html").read_text(encoding="utf-8")

if __name__ == "__main__":
    main()
