"""Stage the allowlisted dashboard, readable report and PDF for publication."""

import argparse
import shutil
from pathlib import Path
from urllib.parse import quote, urlsplit

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/NikitaMeshDS/sberindex-task2/blob/main/"
ALLOWED = {
    "index.html",
    "app.js",
    "style.css",
    "data.js",
    "presentation.pdf",
    "REPORT.html",
    "GLOSSARY.html",
    "SOLUTION.html",
    ".nojekyll",
}


def public_link(href, source):
    parsed = urlsplit(href)
    if parsed.scheme or href.startswith("#"):
        return href
    resolved = (source.parent / parsed.path).resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError:
        raise ValueError("Document link escapes the published project")
    return (
        REPOSITORY
        + quote(str(relative))
        + (("#" + parsed.fragment) if parsed.fragment else "")
    )


def render_document(source):
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = parser.parse(source.read_text())
    for token in tokens:
        for child in token.children or []:
            if child.type == "link_open":
                child.attrs["href"] = public_link(child.attrs["href"], source)
    body = parser.renderer.render(tokens, parser.options, {})
    return (
        '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><link rel="stylesheet" href="style.css"><title>Исследование СберИндекса</title><style>main{max-width:1100px;padding-top:35px;padding-bottom:50px}h1{max-width:none;font-size:38px}h2{margin-top:35px}table{display:block;overflow:auto;margin:20px 0}th{position:static}td,th{text-align:left;padding:10px}p,li{overflow-wrap:anywhere}pre{overflow:auto}</style><header><a href="index.html">← Интерфейс расходов МО</a><a href="presentation.pdf">Презентация PDF</a></header><main>'
        + body
        + "</main></html>"
    )


def main(destination):
    destination.mkdir(parents=True, exist_ok=True)
    unexpected = {p.name for p in destination.iterdir()} - ALLOWED
    if unexpected:
        raise ValueError(
            f"Publication folder contains unreviewed files: {sorted(unexpected)}"
        )
    for name in ["index.html", "app.js", "style.css", "data.js"]:
        shutil.copyfile(ROOT / "dashboard" / name, destination / name)
    documents = {
        "../docs/research/SUBMISSION_REPORT.md": (
            "REPORT.html",
            ROOT / "docs/research/SUBMISSION_REPORT.md",
        ),
        "../docs/research/GLOSSARY.md": (
            "GLOSSARY.html",
            ROOT / "docs/research/GLOSSARY.md",
        ),
        "../docs/research/SOLUTION_IN_FIVE_SENTENCES.md": (
            "SOLUTION.html",
            ROOT / "docs/research/SOLUTION_IN_FIVE_SENTENCES.md",
        ),
    }
    html = (
        (destination / "index.html")
        .read_text()
        .replace("../artifacts/presentation.pdf", "presentation.pdf")
    )
    shutil.copyfile(
        ROOT / "artifacts/presentation.pdf", destination / "presentation.pdf"
    )
    for link, (target, source) in documents.items():
        html = html.replace(link, target)
        (destination / target).write_text(render_document(source))
    (destination / "index.html").write_text(html)
    (destination / ".nojekyll").write_text("")
    assert {p.name for p in destination.iterdir()} == ALLOWED
    print(f"Staged {len(ALLOWED)} reviewed static files in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=ROOT / ".presentation-build/pages-final-site",
    )
    main(parser.parse_args().destination)
