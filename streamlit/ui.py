"""Look and feel only: theme (CSS + embedded font), HTML building blocks and charts. No app logic here."""
import base64
from html import escape

import altair as alt
import pandas as pd
import streamlit as st

from config import ASSETS_DIR, COLORS

FONT_WEIGHTS = (400, 600, 800)


# ---- theme -------------------------------------------------------------------------
@st.cache_resource
def _theme_css():
    faces = []
    for weight in FONT_WEIGHTS:
        data = base64.b64encode((ASSETS_DIR / 'fonts' / f'plus-jakarta-sans-{weight}.woff2').read_bytes()).decode()
        faces.append("@font-face{font-family:'Plus Jakarta Sans';font-style:normal;font-display:swap;"
                     f"font-weight:{weight};src:url(data:font/woff2;base64,{data}) format('woff2');}}")
    return ''.join(faces) + (ASSETS_DIR / 'style.css').read_text(encoding='utf-8')


def apply_theme():
    st.html(f'<style>{_theme_css()}</style>')


def show(html):
    st.html(html)


# ---- building blocks ----------------------------------------------------------------
RADAR = '<div class="radar"><i></i><i></i><i></i><span class="sweep"></span><b></b><b></b><b></b></div>'


def hero(eyebrow, title, subtitle):
    show(f'<div class="hero"><div><span class="hero-eyebrow">{escape(eyebrow)}</span>'
         f'<h1 class="hero-title">{escape(title)}</h1><p class="hero-sub">{escape(subtitle)}</p></div>{RADAR}</div>')


def section(title):
    show(f'<div class="section-title">{escape(title)}</div>')


def empty_state(icon, message):
    show(f'<div class="empty"><b>{icon}</b>{escape(message)}</div>')


def chips(items):
    """items: {name: bool} - green when available, amber when not."""
    html = ''.join(f'<span class="chip {"ok" if ok else "off"}">{"●" if ok else "▲"} {escape(n)}</span>'
                   for n, ok in items.items())
    st.markdown(f'<div class="chips">{html}</div>', unsafe_allow_html=True)


def pill(text, kind='', alert=False):
    return f'<span class="pill {kind} {"alert" if alert else ""}">{escape(text)}</span>'


def kpi_cards(cards):
    """cards: [(label, value, sub, kind, unit)] - numeric values count up from 0."""
    html = []
    for i, (label, value, sub, kind, unit) in enumerate(cards):
        if isinstance(value, (int, float)) and abs(value) >= 1000:
            value = f'{value:,.0f}'          # big numbers keep their thousands separator (no count-up)
        if isinstance(value, (int, float)):
            value_html = f'<div class="kpi-value count" style="--to:{int(round(value))};--unit:\'{unit}\'"></div>'
        else:
            value_html = f'<div class="kpi-value">{escape(str(value))}</div>'
        html.append(f'<div class="kpi {kind}" style="--i:{i}"><div class="kpi-label">{escape(label)}</div>'
                    f'{value_html}<div class="kpi-sub">{escape(sub)}</div></div>')
    show(f'<div class="kpis">{"".join(html)}</div>')


def verdict_card(p_not, threshold, flagged, risk, title, text, pills):
    """Animated gauge (chance of NOT recommending) with the verdict next to it."""
    kind = 'bad' if flagged else 'ok'
    if not flagged and risk == 'Medium':
        kind = 'mid'
    degrees = -90 + 180 * threshold
    show(f'''<div class="verdict {kind}">
      <div class="gauge"><div class="arc-wrap"><div class="arc" style="--gv:{p_not * 180:.1f}"></div>
        <div class="gtick" style="--d:{degrees:.1f}deg"></div></div>
        <div class="gauge-num count" style="--to:{round(p_not * 100)};--unit:'%'"></div>
        <div class="gauge-cap">chance of not recommending</div></div>
      <div><div class="v-eyebrow">Verdict</div><h2 class="v-title">{escape(title)}</h2>
        <p class="v-text">{escape(text)}</p><div class="pills">{pills}</div></div></div>''')


def meters(rows, kind_for):
    """rows: [(label, value)] shown as animated bars. kind_for(value) -> 'bad' | 'good' | 'blue'."""
    if not rows:
        return
    top = max(abs(v) for _, v in rows) or 1
    html = ''.join(f'<div class="meter {kind_for(v)}" style="--i:{i}"><span class="m-label">{escape(str(l))}</span>'
                   f'<span class="m-bar"><i style="--w:{abs(v) / top * 100:.0f}%"></i></span><span class="m-val">{v:+.2f}</span></div>'
                   for i, (l, v) in enumerate(rows))
    show(f'<div class="card">{html}</div>')


def highlighted_review(highlights):
    """The review with words coloured by how strongly they pushed the verdict."""
    top = max((abs(e) for _, e in highlights if e), default=1) or 1
    parts, shown = [], 0
    for piece, effect in highlights:
        if effect and abs(effect) / top > 0.08:
            kind = 'bad' if effect > 0 else 'good'
            alpha = 0.18 + 0.62 * abs(effect) / top
            parts.append(f'<mark class="{kind}" style="--a:{alpha:.2f};--i:{shown}">{escape(piece)}</mark>')
            shown += 1
        else:
            parts.append(escape(piece))
    show('<div class="hl-text">' + ''.join(parts).replace('\n', '<br>') + '</div>'
         '<div class="legend"><span><i style="background:rgba(240,68,56,.55)"></i>pushes towards “not recommended”</span>'
         '<span><i style="background:rgba(18,183,106,.55)"></i>pushes towards “recommends”</span></div>')


def confusion_matrix(tp, fp, fn, tn):
    """2x2 grid: really-unhappy / really-happy customers vs flagged / not flagged."""
    def cell(label, n, color, i):
        return f'<div class="c" style="--cc:{color};--i:{i}"><b>{n:,}</b><span>{escape(label)}</span></div>'
    show('<div class="card"><div class="cm"><div></div><div class="h">Flagged</div><div class="h">Not flagged</div>'
         '<div class="h">Really not<br>recommended</div>'
         + cell('Caught', tp, COLORS['good'], 0) + cell('Missed unhappy customer', fn, COLORS['bad'], 1)
         + '<div class="h">Really<br>recommended</div>'
         + cell('False alarm', fp, COLORS['warn'], 2) + cell('Correctly left alone', tn, COLORS['primary'], 3)
         + '</div></div>')


def timeline(steps):
    """steps: [(title, text)] as an animated vertical timeline."""
    html = ''.join(f'<div class="step" style="--i:{i}"><span class="n">{i + 1}</span><h5>{escape(t)}</h5><p>{escape(x)}</p></div>'
                   for i, (t, x) in enumerate(steps))
    show(f'<div class="timeline">{html}</div>')


# ---- charts (Altair, styled to match) ---------------------------------------------------
def _style(chart, height=300):
    return (chart.properties(height=height, background='transparent')
            .configure_view(stroke=None)
            .configure_axis(labelFont='Plus Jakarta Sans', titleFont='Plus Jakarta Sans', labelColor=COLORS['muted'],
                            titleColor=COLORS['navy'], gridColor='#E6EEFB', domainColor='#D0DBEE', tickColor='#D0DBEE')
            .configure_legend(labelFont='Plus Jakarta Sans', titleFont='Plus Jakarta Sans', orient='bottom')
            .configure_title(font='Plus Jakarta Sans', color=COLORS['navy'], fontSize=14))


def show_chart(chart, height=300):
    st.altair_chart(_style(chart, height), width='stretch')


def bar_chart(df, category, value, group=None, horizontal=False, height=300, sort=None, colors=None, color=None):
    """Rounded bars. `group` is a column that splits the bars by colour; otherwise one `color` is used."""
    whole_numbers = pd.api.types.is_integer_dtype(df[value]) if value in df.columns else False
    axis_args = {'tickCount': 5, **({'tickMinStep': 1} if whole_numbers else {})}
    category_axis = alt.Axis(labelAngle=0, labelLimit=170) if not horizontal else alt.Axis(labelLimit=170)
    category_enc = (alt.Y if horizontal else alt.X)(category, sort=sort, title=None, axis=category_axis)
    value_enc = (alt.X if horizontal else alt.Y)(value, title=None, axis=alt.Axis(**axis_args))
    encoding = {('y' if horizontal else 'x'): category_enc, ('x' if horizontal else 'y'): value_enc,
                'tooltip': [alt.Tooltip(c) for c in df.columns]}
    mark_args = {'cornerRadiusEnd': 7}
    if group:
        scale = alt.Scale(domain=list(colors), range=list(colors.values())) if colors else alt.Undefined
        encoding['color'] = alt.Color(group, scale=scale, title=None)
        encoding['yOffset' if horizontal else 'xOffset'] = (alt.YOffset if horizontal else alt.XOffset)(group)
    else:
        mark_args['color'] = color or COLORS['primary']
    show_chart(alt.Chart(df).mark_bar(**mark_args).encode(**encoding), height)


def line_chart(df, x, y, color, height=300, colors=None):
    scale = alt.Scale(domain=list(colors), range=list(colors.values())) if colors else alt.Undefined
    chart = alt.Chart(df).mark_line(strokeWidth=3, interpolate='monotone').encode(
        x=alt.X(x), y=alt.Y(y), color=alt.Color(color, scale=scale, title=None, legend=alt.Legend(columns=2, labelLimit=240)),
        tooltip=[alt.Tooltip(c) for c in df.columns])
    show_chart(chart, height)


def word_map(df, height=420):
    base = alt.Chart(df)
    points = base.mark_circle(size=170, opacity=.9).encode(
        x=alt.X('x', axis=None), y=alt.Y('y', axis=None),
        color=alt.Color('kind', scale=alt.Scale(domain=['searched word', 'neighbour'], range=[COLORS['bad'], COLORS['primary']]), title=None),
        tooltip=['word', 'kind'])
    labels = base.mark_text(dy=-16, font='Plus Jakarta Sans', fontWeight=700, fontSize=13, color=COLORS['navy']).encode(x='x', y='y', text='word')
    show_chart(points + labels, height)
