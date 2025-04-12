import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import pandas as pd
    import pdfplumber
    import re
    import json
    return json, pd, pdfplumber, re


@app.cell
def _():
    path_who = r"C:\Users\Dee\root\Projects\dev\unpatternedAi\unstructureparser\docs\world_health_statistics_2024.pdf"
    return (path_who,)


@app.cell
def _(path_who, pdfplumber):
    with pdfplumber.open(path_who) as pdf:
        page = pdf.pages[2]
        text = page.extract_text()
    
        # print the extracted text
        for line in text.split('\n'):
            print(line)
    return line, page, pdf, text


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
