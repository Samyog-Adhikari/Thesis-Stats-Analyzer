"""In-memory Excel and chart archive exports."""
from __future__ import annotations

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd


def _safe_sheet(title: str, used: set[str]) -> str:
    """Return a unique Excel-safe worksheet name."""
    name = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)[:31] or "Result"
    base, i = name, 2
    while name in used: name = f"{base[:28]}_{i}"; i += 1
    used.add(name); return name


def results_to_excel(results: list[dict]) -> bytes:
    """Export result tables into one in-memory workbook."""
    output = BytesIO(); used = set()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for result in results:
            sheet = _safe_sheet(result.get("title", "Result"), used); row = 0
            pd.DataFrame([{"Title": result.get("title", ""), "N": result.get("n", ""), "Interpretation": result.get("interpretation", "")}]).to_excel(writer, sheet_name=sheet, index=False, startrow=row); row += 3
            for i, table in enumerate(result.get("tables", []), 1):
                if isinstance(table, pd.DataFrame): table.to_excel(writer, sheet_name=sheet, index=True, startrow=row, startcol=0); row += len(table) + 3
    return output.getvalue()


def figures_to_zip(results: list[dict]) -> bytes:
    """Export every result figure as a 300-DPI PNG in a ZIP archive."""
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for ri, result in enumerate(results, 1):
            for fi, fig in enumerate(result.get("figures", []), 1):
                image = BytesIO(); fig.savefig(image, format="png", dpi=300, bbox_inches="tight", facecolor="white"); archive.writestr(f"result_{ri}_figure_{fi}.png", image.getvalue())
    return output.getvalue()


def figure_to_png(figure) -> bytes:
    """Export one matplotlib figure as a 300-DPI PNG in memory."""
    image = BytesIO()
    figure.savefig(image, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    return image.getvalue()
