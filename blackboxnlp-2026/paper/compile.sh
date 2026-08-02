#!/bin/bash
# Compile the BlackboxNLP paper from any machine.
# Auto-cd to this script's directory so it works from anywhere.

cd "$(dirname "$0")"

if command -v tectonic >/dev/null 2>&1; then
    tectonic main.tex
else
    pdflatex -interaction=nonstopmode main.tex
    bibtex main
    pdflatex -interaction=nonstopmode main.tex
    pdflatex -interaction=nonstopmode main.tex
fi

# main.pdf is the final artifact
