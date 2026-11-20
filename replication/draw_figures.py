"""Draw the journal figures from the archived machine-readable estimates."""
import csv
import math
from pathlib import Path

import reportlab
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "output/final_paper_20261103/tables"
FONT = "Vera"
pdfmetrics.registerFont(TTFont(FONT, str(Path(reportlab.__file__).parent / "fonts/Vera.ttf")))
PRIMARY, FIXED, BOOST = "#155866", "#654B85", "#8F4B22"


def rows(name):
    with (TABLES / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def label(c, x, y, value, size=8, color="#20252A", align="left"):
    c.setFont(FONT, size)
    c.setFillColor(HexColor(color))
    writer = {"left": c.drawString, "right": c.drawRightString, "center": c.drawCentredString}
    writer[align](x, y, str(value))


def interval(c, fn, y, low, point, high, color, shape):
    c.setStrokeColor(HexColor(color))
    c.setFillColor(HexColor(color))
    c.setLineWidth(1.3)
    c.line(fn(low), y, fn(high), y)
    for end in (low, high):
        c.line(fn(end), y - 2, fn(end), y + 2)
    x = fn(point)
    if shape == "circle":
        c.circle(x, y, 2.6, fill=1)
    elif shape == "square":
        c.rect(x-2.5, y-2.5, 5, 5, fill=1)
    else:
        p = c.beginPath()
        p.moveTo(x, y+3)
        p.lineTo(x+3.2, y-2.6)
        p.lineTo(x-3.2, y-2.6)
        p.close()
        c.drawPath(p, fill=1)


def grid(c, fn, ticks, lower, upper, reference):
    for tick in ticks:
        c.setStrokeColor(HexColor("#757B82" if tick == reference else "#DDE0E1"))
        c.setLineWidth(0.8 if tick == reference else 0.4)
        c.line(fn(tick), lower, fn(tick), upper)
        label(c, fn(tick), lower-12, f"{tick:g}", align="center")


def prediction():
    c = Canvas(str(ROOT / "journal/Fig1.pdf"), pagesize=(465, 196), invariant=True)
    c.setTitle("Incremental average precision by checkpoint")
    x = lambda d: 95 + (d+3)*31
    grid(c, x, [-2, 0, 2, 4, 6, 8], 27, 162, 0)
    label(c, 9, 172, "Round")
    label(c, 9, 175+10, "Comparison", size=7)
    legend = [(91, "Logistic", PRIMARY, "circle"), (177, "Logistic, reached 30", FIXED, "square"),
              (316, "Gradient boosting", BOOST, "triangle")]
    for at, text, color, shape in legend:
        interval(c, lambda n: at+n, 187, -6, 0, 6, color, shape)
        label(c, at+10, 184, text)
    all_rows = rows("checkpoint_metrics.csv")
    for i,k in enumerate((5,10,15,20,30)):
        y = 150-i*27
        label(c, 9, y-3, f"{k}" + (" (primary)" if k == 15 else ""))
        for r in all_rows:
            if int(r["k"]) != k or r["model"] != "M2":
                continue
            if r["kind"] == "hist_gradient_boosting":
                color,shape,offset = BOOST,"triangle",-7
            elif r["population"] == "fixed30":
                color,shape,offset = FIXED,"square",0
            else:
                color,shape,offset = PRIMARY,"circle",7
            interval(c,x,y+offset,*(1000*float(r[key]) for key in ("delta_ap_low","delta_ap","delta_ap_high")),color,shape)
    label(c, 263, 1, "Change in average precision (thousandths)", align="center")
    c.save()


def association():
    c = Canvas(str(ROOT / "journal/Fig2.pdf"), pagesize=(465, 167), invariant=True)
    c.setTitle("Command presence and odds of recorded failure")
    x = lambda v: 143 + 160 * math.log2(v / 0.65)
    grid(c,x,[0.7,0.8,1,1.25,1.5,1.8],23,141,1)
    label(c, 5, 149, "Round / adjustment")
    label(c, 456, 149, "Valid refits", align="right")
    estimates = [r for r in rows("within_issue_odds_ratios.csv")
                 if r["feature"] == "ran_tests" and r["analysis"] in ("A","B","C")]
    names = {"A":"Pooled","B":"Within issue","C":"Within issue/config."}
    for i,r in enumerate(sorted(estimates,key=lambda r:(int(r["k"]),r["analysis"]))):
        y = 132-i*18-(8 if i>=3 else 0)
        label(c,5,y-3,f'{r["k"]}    {names[r["analysis"]]}')
        label(c,456,y-3,f'{r["bootstrap_valid"]}/500',align="right")
        interval(c,x,y,*(float(r[key]) for key in ("or_low","odds_ratio","or_high")),PRIMARY,"circle")
    label(c,253,1,"Odds ratio for failure (log scale)",align="center")
    c.save()


if __name__ == "__main__":
    prediction()
    association()
