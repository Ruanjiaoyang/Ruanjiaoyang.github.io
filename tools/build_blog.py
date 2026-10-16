"""Build blog pages from Markdown.

Every content/<series>/<NN-slug>.md becomes blog/<series>/<slug>/index.html,
and the Blog section of index.html (between the BLOG markers) is regenerated.

To publish a new post: copy its .md into content/<series>/ and run
    python3 tools/build_blog.py
Requires: pip install markdown pyyaml
"""
import html
import re
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify_unicode
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"
OUT = ROOT / "blog"
INDEX = ROOT / "index.html"
START, END = "<!-- BLOG:START -->", "<!-- BLOG:END -->"


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
    kicker = f"{series} · 第 {order} 篇" if series and order is not None else series
    toc_block = (f'<details class="toc"><summary>目录 · Table of Contents</summary>{toc_html}</details>'
                 if "<li>" in toc_html else "")
    fields = dict(title=title, title_en=title_en, desc=desc, tags=tags, kicker=kicker,
                  date=str(p.get("date", "")), toc=toc_block, body=body_html)
    out = TEMPLATE
    for k, v in fields.items():
        out = out.replace("{{" + k + "}}", v)
    return out


def blog_list(posts):
    if not posts:
        return '<p class="empty"><span lang="en">Posts coming soon.</span><span lang="zh">文章即将上线。</span></p>'
    rows = []
    for p in posts:
        summary = html.escape(" ".join(str(p.get("summary_en", "")).split()))
        rows.append(f'''        <article class="entry">
          <h3><span lang="en">{html.escape(str(p.get("title_en") or p.get("title")))}</span><span lang="zh">{html.escape(str(p.get("title")))}</span></h3>
          <p class="entry-sum">{summary}</p>
          <p class="entry-meta">{meta_line(p)}</p>
          <a class="entry-link" href="{p["url"]}" aria-label="{html.escape(str(p.get("title_en") or p.get("title")))}"></a>
        </article>''')
    return '<div class="entries">\n' + "\n".join(rows) + "\n      </div>"


def main():
    posts = [read_post(f) for f in sorted(CONTENT.glob("*/*.md"))]
    for p in posts:
        p["minutes"] = reading_minutes(p["body"])
    for p in posts:
        dest = OUT / p["series_dir"] / p["slug"]
        dest.mkdir(parents=True, exist_ok=True)
        body_html, toc_html = render(p["body"])
        (dest / "index.html").write_text(page(p, body_html, toc_html), encoding="utf-8")
        print("built", dest.relative_to(ROOT))
    posts.sort(key=lambda p: (str(p.get("date", "")), p.get("order") or 0), reverse=True)
    s = INDEX.read_text(encoding="utf-8")
    i, j = s.index(START) + len(START), s.index(END)
    INDEX.write_text(s[:i] + "\n      " + blog_list(posts) + "\n      " + s[j:], encoding="utf-8")
    print("updated index.html blog list:", len(posts), "post(s)")


TEMPLATE = (Path(__file__).parent / "post_template.html").read_text(encoding="utf-8")

if __name__ == "__main__":
    main()
