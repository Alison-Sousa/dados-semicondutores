# 1) Importações de bibliotecas
import math
import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LinearSegmentedColormap, Normalize

plt.switch_backend("Agg")

# 2) Caminhos e constantes globais
ROOT = Path(__file__).resolve().parents[1]
EXCEL = ROOT / "Bases" / "base_de_dados.xlsx"
SHAPE = ROOT / "BR_Municipios_2024" / "BR_Municipios_2024.shp"
FIG = ROOT / "Figuras"

UF_ALVO = "São Paulo"
ANO_INICIO = 2000
ANO_FIM = 2025

# 3) Paletas de cores por gráfico
COR_EXP = "#7fbce6"
COR_IMP = "#4a90c4"
COR_SALDO = "#1a3a5c"
COR_BARRA = "#C62828"
COR_BORDA = "#8E1B1B"

# 4) Mapeamento UF → região para mapas de custo
UF_TO_REGIAO = {
    "AC": "Norte",  "AM": "Norte",  "AP": "Norte",  "PA": "Norte",
    "RO": "Norte",  "RR": "Norte",  "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste", "SE": "Nordeste",
    "DF": "C.-Oeste", "GO": "C.-Oeste", "MS": "C.-Oeste", "MT": "C.-Oeste",
    "ES": "Sudeste",  "MG": "Sudeste",  "RJ": "Sudeste",  "SP": "Sudeste",
    "PR": "Sul",     "RS": "Sul",     "SC": "Sul",
}


# 5) Funções auxiliares de transformação de dados BC
def extrair_mes(v):
    m = re.search(r"(\d+)", str(v))
    if not m:
        raise ValueError(f"Mês inválido: {v}")
    return int(m.group(1))


def transformar_series(df_uf):
    d = df_uf.copy()
    d["Mes_Num"] = d["Mês"].apply(extrair_mes)
    exp_c = [c for c in d.columns if c.startswith("Exportação - ") and "Valor US$ FOB" in c]
    imp_c = [c for c in d.columns if c.startswith("Importação - ") and "Valor US$ FOB" in c]

    def melt(cols, nome):
        t = d[["Mes_Num"] + cols].melt(id_vars=["Mes_Num"], value_vars=cols, var_name="Col", value_name=nome)
        t["Ano"] = t["Col"].str.extract(r"(\d{4})").astype(int)
        return t.drop(columns=["Col"])

    e = melt(exp_c, "Exp")
    i = melt(imp_c, "Imp")
    m = pd.merge(e, i, on=["Ano", "Mes_Num"], how="outer").fillna(0)
    m["Data"] = pd.to_datetime(dict(year=m["Ano"], month=m["Mes_Num"], day=1))
    mo = m.groupby("Data", as_index=False)[["Exp", "Imp"]].sum().sort_values("Data")
    mo["Saldo"] = mo["Exp"] - mo["Imp"]
    idx = pd.date_range(f"{ANO_INICIO}-01-01", f"{ANO_FIM}-12-01", freq="MS")
    mo = mo.set_index("Data").reindex(idx, fill_value=0).rename_axis("Data").reset_index()
    return mo


def gerar_indicadores(df_m):
    d25 = df_m[df_m["Data"].dt.year == 2025]
    rows = []
    for label, d in [("2025", d25), ("Total", df_m)]:
        te, ti = d["Exp"].sum(), d["Imp"].sum()
        rows.append({"Periodo": label, "Imp": ti, "Exp": te, "Saldo": te - ti, "Razao": ti / te if te else 0})
    return pd.DataFrame(rows)


# 6) Balança comercial SP + PIB por setor
def plotar_bc_sp(df_m, df_pib, path, oculto=False):
    T = 18
    plt.rcParams.update({"font.size": T, "axes.titlesize": T + 2, "axes.labelsize": T,
                          "xtick.labelsize": T, "ytick.labelsize": T, "legend.fontsize": T - 2})
    bilhoes = FuncFormatter(lambda x, _: f"{x / 1e9:.0f}")

    fig = plt.figure(figsize=(36, 14), facecolor="white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.8, 1.0], wspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    ax_pie = fig.add_subplot(gs[0, 1])

    dp = df_m.copy()
    dp["Ano"] = dp["Data"].dt.year
    da = dp.groupby("Ano", as_index=False)[["Exp", "Imp", "Saldo"]].sum()
    x = da["Ano"].to_numpy()
    w = 0.42
    ax.bar(x - w / 2, da["Exp"], width=w, color=COR_EXP, label="Exportação", edgecolor="none", zorder=3)
    ax.bar(x + w / 2, da["Imp"], width=w, color=COR_IMP, label="Importação", edgecolor="none", zorder=2)

    ax2 = ax.twinx()
    sp = da["Saldo"].where((da["Exp"] > 0) & (da["Imp"] > 0), other=pd.NA)
    ax2.plot(da["Ano"], sp, color=COR_SALDO, lw=2.4, ls="-", marker="o", ms=3.8,
             markerfacecolor=COR_SALDO, markeredgewidth=0, label="Saldo comercial", zorder=4)

    ax.set_ylabel("Valor (US$ bilhões)", fontsize=T)
    ax2.set_ylabel("Saldo (US$ bilhões)", fontsize=T)
    ax.set_xlabel("Ano", fontsize=T)
    ax.yaxis.set_major_formatter(bilhoes)
    ax2.yaxis.set_major_formatter(bilhoes)
    ms = sp.min(skipna=True)
    if pd.notna(ms) and ms < 0:
        ax2.set_ylim(0, ms * 1.05)
    ax2.axhline(0, color="#888", lw=0.9, ls=":", zorder=1)
    ax.set_xlim(ANO_INICIO - 0.6, ANO_FIM + 0.6)
    ax.set_xticks(da["Ano"])
    ax.set_xticklabels(da["Ano"].astype(str), rotation=45, ha="right", fontsize=T - 4)
    ax.grid(visible=False); ax2.grid(visible=False)
    ax.set_title("Balança Comercial da Indústria", fontsize=T + 4, fontweight="bold", pad=20)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg_ax = ax.legend(h1 + h2, l1 + l2, loc="upper right", bbox_to_anchor=(0.75, 1.0),
                       frameon=False, ncol=3, fontsize=T - 2)
    for s in list(ax.spines.values()) + list(ax2.spines.values()):
        s.set_visible(True); s.set_edgecolor("#BBB"); s.set_linewidth(1.2)

    # 6a) Donut PIB por setor
    col_s = next((c for c in df_pib.columns if "setor" in c.lower()), df_pib.columns[0])
    col_p = next((c for c in df_pib.columns if "part" in c.lower() or "pib" in c.lower() or "%" in c), df_pib.columns[1])
    dp2 = df_pib[[col_s, col_p]].copy()
    dp2[col_p] = pd.to_numeric(dp2[col_p], errors="coerce").fillna(0)
    dp2 = dp2[dp2[col_p] > 0].sort_values(col_p, ascending=False)
    dp2["Lbl"] = dp2[col_s]
    cores_b = ["#1a3a5c", "#2a5f8f", "#4a90c4", "#6db3d6", "#8fcce0", "#b4dde8", "#d6eef4"]
    vals = dp2[col_p].values
    nomes = dp2["Lbl"].values
    n_setores = len(nomes)
    cores = [cores_b[int(round(i * (len(cores_b) - 1) / max(n_setores - 1, 1)))]
             for i in range(n_setores)]

    r_d = 1.58 if oculto else 1.36; cy = 0.0 if oculto else -0.20
    wedges, _ = ax_pie.pie(vals, labels=None, startangle=90, counterclock=False,
                           colors=cores[:len(vals)], center=(0, cy), radius=r_d,
                           wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 2.5})
    ax_pie.set_aspect("auto")
    lh = [Patch(facecolor=cores[i], edgecolor="white", label=nomes[i]) for i in range(len(nomes))]
    leg_pie = ax_pie.legend(handles=lh, loc="upper center", bbox_to_anchor=(0.5, 0.985),
                            frameon=False,
                            ncol=3, fontsize=T - 8, columnspacing=0.8, handletextpad=0.4, alignment="center")

    def cor_txt(hx):
        h = hx.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return "white" if 0.299 * r + 0.587 * g + 0.114 * b < 145 else "black"

    txts = []
    for i, (wd, v, c) in enumerate(zip(wedges, vals, cores)):
        ang = (wd.theta2 + wd.theta1) / 2
        ar = math.radians(ang)
        vr = int(round(float(v)))
        if vr >= 2:
            rr = 0.83 * r_d
            xp = rr * math.cos(ar); yp = cy + rr * math.sin(ar)
        else:
            rr = 0.73 * r_d
            xp = rr * math.cos(ar) + 0.05; yp = cy + rr * math.sin(ar) - 0.16
        txts.append(ax_pie.text(xp, yp, f"{vr}%", ha="center", va="center",
                                fontsize=T - 1 if v >= 5 else T - 2, fontweight="bold", color=cor_txt(c)))

    ax_pie.set_title("PIB por Setor Paulista", fontsize=T + 4, fontweight="bold", pad=20)
    ax_pie.set_xlim(-1.7, 1.7); ax_pie.set_ylim(-1.7, 1.7)
    ax_pie.set_xticks([]); ax_pie.set_yticks([])
    ax_pie.set_frame_on(True); ax_pie.patch.set_facecolor("white"); ax_pie.patch.set_edgecolor("none")
    for s in ax_pie.spines.values():
        s.set_visible(True); s.set_edgecolor("#BBB"); s.set_linewidth(1.2)
    ax_pie.add_patch(Rectangle((0, 0), 1, 1, transform=ax_pie.transAxes, fill=False,
                                edgecolor="#BBB", linewidth=1.2, zorder=20, clip_on=False))

    if oculto:
        ax.title.set_visible(False); ax_pie.title.set_visible(False)
        ax.xaxis.label.set_visible(False); ax.yaxis.label.set_visible(False); ax2.yaxis.label.set_visible(False)
        for lb in ax.get_xticklabels() + ax.get_yticklabels() + ax2.get_yticklabels():
            lb.set_visible(False)
        for t in txts:
            t.set_visible(False)
        for t in leg_ax.get_texts():
            t.set_visible(False)
        leg_pie.set_visible(False)
        ax_pie.set_xlim(-1.9, 1.9); ax_pie.set_ylim(-1.9, 1.9)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight", format="jpg", facecolor="white")
    plt.close(fig)


# 7) Empregos no setor + donut por setor econômico
def plotar_empregos(df_emp, df_setor, path, oculto=False):
    sns.set_style("white")
    T = 18
    plt.rcParams.update({"font.size": T, "axes.titlesize": T + 2, "axes.labelsize": T,
                          "xtick.labelsize": T, "ytick.labelsize": T, "legend.fontsize": T - 4})

    fig = plt.figure(figsize=(36, 14), facecolor="white")
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.8, 1.0], wspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    ax_pie = fig.add_subplot(gs[0, 1])

    tons = ["#7A1111", "#8E1B1B", "#A32222", "#B92A2A", "#C62828",
            "#D43A3A", "#E05555", "#EA7474", "#F09494", "#F6B5B5"]
    cores_b = [tons[i % len(tons)] for i in range(len(df_emp))]
    barras = ax.bar(df_emp["Ano"], df_emp["Valor"], color=cores_b, edgecolor=COR_BORDA,
                    linewidth=0.8, width=0.7, label="Participação", zorder=3)
    leg_ax = ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=1,
                       frameon=False)
    ax.set_title("Empregos no Setor", fontsize=T + 4, fontweight="bold", pad=18)
    ax.set_ylabel("Participação (%)", fontsize=T); ax.set_xlabel("Ano", fontsize=T)
    ax.set_xlim(df_emp["Ano"].min() - 0.6, df_emp["Ano"].max() + 0.6)
    ax.set_xticks(df_emp["Ano"])
    ax.set_xticklabels(df_emp["Ano"].astype(str), rotation=45, ha="right", fontsize=T - 4)
    ax.set_yticks([]); ax.grid(axis="y", ls="-", lw=0.5, alpha=0.25); ax.grid(axis="x", visible=False)

    txts_b = []
    for b, v in zip(barras, df_emp["Valor"]):
        txts_b.append(ax.text(b.get_x() + b.get_width() / 2, v + max(df_emp["Valor"].max() * 0.01, 0.05),
                              f"{v:.1f}".replace(".", ","), ha="center", va="bottom", fontsize=T - 4, color="#7A1111"))
    for s in ax.spines.values():
        s.set_visible(True); s.set_edgecolor("#BBB"); s.set_linewidth(1.2)

    # 7a) Donut empregos por setor
    tons_d = ["#7A1111", "#A32222", "#C62828", "#D95B5B", "#E98D8D", "#F2B6B6", "#F8DCDC"]
    vals = df_setor["Valor"].values
    nomes = df_setor["Setor"].values
    n_d = len(vals)
    cores_d = [tons_d[min(int(round(i * (len(tons_d) - 1) / max(n_d - 1, 1))), len(tons_d) - 1)]
               for i in range(n_d)]
    r_emp = 1.15 if oculto else 1.0; cy = 0.0 if oculto else -0.20
    wedges, _ = ax_pie.pie(vals, labels=None, startangle=90, counterclock=False, colors=cores_d,
                           center=(0, cy), radius=r_emp, wedgeprops={"width": 0.45, "edgecolor": "white", "linewidth": 2.0})
    lh = [Patch(facecolor=cores_d[i], edgecolor="white", label=nomes[i]) for i in range(len(nomes))]
    leg_pie = ax_pie.legend(handles=lh, loc="upper center", bbox_to_anchor=(0.5, 0.985),
                            frameon=False,
                            ncol=3, fontsize=T - 5, columnspacing=1.0, handletextpad=0.5, alignment="center")

    txts_d = []
    for wd, v, c, n in zip(wedges, vals, cores_d, nomes):
        ang = (wd.theta2 + wd.theta1) / 2
        ar = math.radians(ang)
        rr = 0.78 if v >= 8 else 0.84 if v >= 2 else 0.90 if v >= 1 else 0.94
        xp = rr * math.cos(ar); yp = cy + rr * math.sin(ar)
        if v < 2:
            yp -= 0.03
        txt = "1%" if v < 1.5 else "2%" if v < 2 else f"{int(round(v))}%"
        if v < 1.5:
            xp, yp = 0.04, cy + 0.52
        cl = "white" if c in ["#7A1111", "#A32222", "#C62828", "#D95B5B"] else "black"
        txts_d.append(ax_pie.text(xp, yp, txt, ha="center", va="center", fontsize=T - 3, fontweight="bold", color=cl))

    ax_pie.set_title("Empregos por Setor", fontsize=T + 4, fontweight="bold", pad=20)
    ax_pie.set_aspect("equal")
    ax_pie.set_xlim(-1.2, 1.2); ax_pie.set_ylim(-1.35, 1.2)
    ax_pie.set_xticks([]); ax_pie.set_yticks([])
    for s in ax_pie.spines.values():
        s.set_visible(True); s.set_edgecolor("#BBB"); s.set_linewidth(1.2)
    ax_pie.add_patch(Rectangle((0, 0), 1, 1, transform=ax_pie.transAxes, fill=False,
                                edgecolor="#BBB", linewidth=1.2, zorder=20, clip_on=False))

    if oculto:
        ax.title.set_visible(False); ax_pie.title.set_visible(False)
        ax.xaxis.label.set_visible(False); ax.yaxis.label.set_visible(False)
        for lb in ax.get_xticklabels() + ax.get_yticklabels():
            lb.set_visible(False)
        for t in txts_b + txts_d:
            t.set_visible(False)
        for t in leg_ax.get_texts():
            t.set_visible(False)
        leg_pie.set_visible(False)
        ax_pie.set_aspect("auto")
        ax_pie.set_xlim(-1.4, 1.4); ax_pie.set_ylim(-1.4, 1.4)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.12)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight", format="jpg", facecolor="white")
    plt.close(fig)


# 8) Funções auxiliares de texto e abreviação
def _normalize_text(series):
    return series.astype(str).str.upper().str.strip().str.replace(r"\s+", " ", regex=True)


def _normalize_tipo(series):
    t = _normalize_text(series)
    is_t = t.str.contains(r"TESE|DOUTOR", regex=True, na=False)
    is_d = t.str.contains(r"DISSERT|MESTRADO", regex=True, na=False)
    return pd.Series(np.where(is_t, "TESE", np.where(is_d, "DISSERTAÇÃO", "OUTROS")), index=series.index)


def _abbrev_ies(name):
    t = str(name).upper().strip()
    sw = {"DE", "DO", "DA", "DAS", "DOS", "E"}
    words = [w for w in t.replace("-", " ").split() if w and w not in sw]
    ac = "".join(w[0] for w in words)
    return ac if 2 <= len(ac) <= 8 else t[:12]


def _short_label(t, mx=30):
    v = str(t).upper().strip()
    if len(v) <= mx:
        return v
    sw = {"DE", "DO", "DA", "DAS", "DOS", "E"}
    words = [w for w in v.replace(",", " ").replace("-", " ").split() if w and w not in sw]
    ac = "".join(w[0] for w in words)
    return ac if 2 <= len(ac) <= 10 else v[:mx - 1] + "…"


def _abbrev_empresa(name: str) -> str:
    t = str(name).strip()
    sw = {"DE", "DO", "DA", "DAS", "DOS", "E", "SA", "S.A.", "S/A", "LTDA", "AND", "THE", "OF"}
    words = [w for w in re.split(r'[\s\-/\.]+', t.upper()) if w and w not in sw and len(w) > 1]
    ac = "".join(w[0] for w in words if w.isalpha())
    if 2 <= len(ac) <= 6:
        return ac
    return " ".join(words[:2])[:10]


def _abbrev_depositante(name: str) -> str:
    t = str(name).strip()
    m_cc = re.search(r'-?\[([A-Z]{2})\]', t)
    country = f" [{m_cc.group(1)}]" if m_cc else ""
    t = re.sub(r';.*$', '', t)
    t = re.sub(r'-?\[.*?\]', '', t).strip().rstrip('-').strip()
    m_ac = re.search(
        r'\b(USP|UNICAMP|UFRJ|UFMG|UFSC|UFPR|UFBA|UFRN|UFC|UFG|INPI|INPE|UNB|UFSM|UFAM|PUC)\b',
        t.upper())
    if m_ac:
        return m_ac.group(1) + country
    sw2 = {"DE", "DO", "DA", "DAS", "DOS", "E", "OF", "THE", "AND", "PARA", "EM", "LA", "LOS"}
    words = [w for w in re.split(r'[\s\-]+', t.upper()) if w and w not in sw2 and w.isalpha()]
    ac = "".join(w[0] for w in words)
    if 2 <= len(ac) <= 8:
        return ac + country
    return t[:16].strip() + country


def _add_frame(ax):
    ax.add_patch(mpatches.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                                     fill=False, linewidth=1.6, edgecolor="#B3A7B9"))


# 9) Teses e dissertações — barras por ano + donuts região/IES/área
def plotar_teses(df, path, oculto=False):
    years = list(range(2013, 2025))
    tipo_s = _normalize_tipo(df["tipo"])
    ct = tipo_s.value_counts()
    sel = [t for t in ["DISSERTAÇÃO", "TESE"] if t in ct.index]
    if len(sel) < 2:
        sel = list(ct.head(2).index)

    ybt = (df.assign(tn=tipo_s).groupby(["ano_base", "tn"], as_index=False).size()
           .pivot(index="ano_base", columns="tn", values="size").reindex(years, fill_value=0))

    fig = plt.figure(figsize=(28, 14), dpi=120)
    grid = fig.add_gridspec(2, 3, height_ratios=[2.0, 2.0], hspace=0.24, wspace=0.28)
    ax_top = fig.add_subplot(grid[0, :])
    ax_reg = fig.add_subplot(grid[1, 0])
    ax_uni = fig.add_subplot(grid[1, 1])
    ax_area = fig.add_subplot(grid[1, 2])

    pd_, pm, pl, pll = "#4A254F", "#6F4B75", "#BCA2C4", "#D7C6DD"
    bw = 0.34; x = np.arange(len(years))
    ta, tb = sel[0], sel[1]
    ya = ybt[ta].values; yb = ybt[tb].values
    ba = ax_top.bar(x - bw / 2, ya, width=bw, color=pd_, alpha=0.95, label=ta.title())
    bb = ax_top.bar(x + bw / 2, yb, width=bw, color=pm, alpha=0.95, label=tb.title())

    # 9a) Rótulos numéricos acima das barras (apenas versão normal)
    txts_bar = []
    if not oculto:
        vmax_t = max(ya.max(), yb.max()) if len(ya) else 1
        for bars, vals_b in [(ba, ya), (bb, yb)]:
            for bar_r, v in zip(bars, vals_b):
                if v > 0:
                    txts_bar.append(ax_top.text(
                        bar_r.get_x() + bar_r.get_width() / 2,
                        v + vmax_t * 0.015,
                        str(int(v)), ha="center", va="bottom",
                        fontsize=8, fontweight="bold", color="#4A254F"))

    ax_top.set_xlim(-0.6, len(years) - 0.4); ax_top.set_xticks(x); ax_top.set_xticklabels(years)
    ax_top.grid(axis="y", ls="--", lw=0.6, alpha=0.35); ax_top.spines[["top", "right"]].set_visible(False)
    tl = [ta.title(), tb.title()] if not oculto else ["", ""]
    ax_top.legend([ba, bb], tl, ncol=2, loc="upper left", bbox_to_anchor=(-0.01, 1.13),
                  frameon=False, fontsize=9, handlelength=1.8, columnspacing=1.2,
                  handletextpad=0.0 if oculto else 0.5)

    # 9b) Donuts: região, IES, área de conhecimento
    def pie_panel(ax, counts, use_abbrev=False):
        labels = list(counts.index)
        vals = counts.values
        colors = [pd_, pm, "#8E6A95", pl, pll][:len(labels)]
        pct = (lambda p: f"{p:.1f}%" if p >= 2.0 else "") if not oculto else None
        pie = ax.pie(vals, labels=None, autopct=pct, startangle=90, pctdistance=0.8,
                     colors=colors, radius=1.52 if oculto else 1.12,
                     wedgeprops={"width": 0.45, "edgecolor": "white"},
                     textprops={"fontsize": 9})
        wdg = pie[0]
        if use_abbrev:
            ll = [_abbrev_ies(v) for v in labels]
        else:
            ll = [_short_label(v, 34) for v in labels]
        ll = ll if not oculto else [""] * len(ll)
        lg = ax.legend(wdg, ll, loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False,
                  fontsize=8.3, handlelength=1.2, handletextpad=0.0 if oculto else 0.5, borderaxespad=0.0)
        if oculto:
            lg.set_visible(False)
        ax.set(aspect="equal")
        if oculto:
            ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.75, 1.75)
        if not oculto:
            _add_frame(ax)

    pie_panel(ax_reg, _normalize_text(df["regiao"]).value_counts())
    pie_panel(ax_uni, _normalize_text(df["ies"]).value_counts().head(5), use_abbrev=True)
    pie_panel(ax_area, _normalize_text(df["area"]).value_counts().head(5))

    for axis, dx in zip((ax_reg, ax_uni, ax_area), (-0.12, -0.09, -0.06)):
        pos = axis.get_position(); axis.set_position([pos.x0 + dx, pos.y0, pos.width, pos.height])

    if oculto:
        ax_top.tick_params(axis="both", labelleft=False, labelbottom=False)
        for a in (ax_reg, ax_uni, ax_area):
            a.set_xticks([]); a.set_yticks([])

    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.01, right=0.92, top=0.90, bottom=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="jpg", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 10) Mapa cadeia produtiva — projeto de CI e encapsulamento/teste
def norm_geo(v):
    t = str(v).strip().upper()
    t = unicodedata.normalize("NFKD", t).encode("ASCII", "ignore").decode("ASCII")
    return " ".join(t.split())


def spread_pts(df):
    df = df.copy()
    for _, idx in df.groupby(["uf", "local_norm", "etapa"]).groups.items():
        idx = list(idx)
        if len(idx) <= 1:
            continue
        angs = np.linspace(0, 2 * np.pi, len(idx), endpoint=False)
        for i, ri in enumerate(idx):
            df.at[ri, "x"] += 0.38 * np.cos(angs[i])
            df.at[ri, "y"] += 0.38 * np.sin(angs[i])
    return df


def plotar_mapa(estados, empresas, path, oculto=False):
    cor = "#7B5C9A"
    fig, axes = plt.subplots(1, 2, figsize=(24, 12), dpi=130)

    for ax, etapa, titulo in zip(axes, ["SFP", "TMT"],
                                  ["Projeto de circuito integrado", "Encapsulamento e teste"]):
        dados = empresas[empresas["etapa"] == etapa].copy()
        uf_set = sorted(dados["uf"].unique().tolist())
        estados_sel = estados[estados["SIGLA_UF"].isin(uf_set)].copy()
        estados.plot(ax=ax, color="#E7E7E9" if not oculto else "#ECECEF",
                     edgecolor="#F6F6F6" if not oculto else "#F8F8F8", linewidth=0.7)
        if not estados_sel.empty:
            estados_sel.plot(ax=ax, color=cor, edgecolor="#F7F7F7", linewidth=0.9)
        ax.scatter(dados["x"], dados["y"], s=24 if not oculto else 22, marker="s", color="#1D1A1A", zorder=5)

        if not oculto:
            mx_map = (estados.total_bounds[0] + estados.total_bounds[2]) / 2
            for _, r in dados.iterrows():
                abbrev = _abbrev_empresa(r["empresa"])
                ha = "left" if r["x"] >= mx_map else "right"
                dx = 0.55 if r["x"] >= mx_map else -0.55
                ax.text(r["x"] + dx, r["y"] + 0.28, abbrev, fontsize=7.8, fontweight="semibold",
                        color="#171717", ha=ha, va="center", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                                  edgecolor="none", alpha=0.75))

        for _, r in estados.iterrows():
            ax.text(r["x"], r["y"], r["SIGLA_UF"], fontsize=7.7, color="#243447",
                    ha="center", va="center", fontweight="semibold", zorder=4)

        if not oculto:
            ax.set_title(titulo, fontsize=12, fontweight="bold", loc="left", pad=6)
        ax.set_axis_off()
        mnx, mny, mxx, mxy = estados.total_bounds
        ax.set_xlim(mnx - 1.4, mxx + 1.4); ax.set_ylim(mny - 0.8, mxy + 0.8)

    fig.patch.set_facecolor("white")
    top = 0.93 if not oculto else 0.98
    fig.subplots_adjust(left=0.01, right=0.995, top=top, bottom=0.02 if not oculto else 0.02, wspace=-0.16)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250, format="jpg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 11) Preparação de dados de empregos e setor
def preparar_empregos(df):
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    col_ano = next((c for c in df2.columns if "ano" in c.lower()), None)
    if not col_ano:
        raise ValueError("Coluna de ano não encontrada")
    others = [c for c in df2.columns if c != col_ano]
    pri = [c for c in others if re.search(r"percent|particip|%", c, flags=re.IGNORECASE)]
    col_val = None
    for c in pri + others:
        if pd.to_numeric(df2[c], errors="coerce").notna().sum() > 0:
            col_val = c; break
    if not col_val:
        raise ValueError("Coluna de valores não encontrada")
    out = pd.DataFrame({"Ano": pd.to_numeric(df2[col_ano], errors="coerce"),
                         "Valor": pd.to_numeric(df2[col_val], errors="coerce")}).dropna()
    out["Ano"] = out["Ano"].astype(int)
    return out.sort_values("Ano")


def preparar_setor(df):
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    col_s = next((c for c in df2.columns if "setor" in c.lower()), None)
    if not col_s:
        col_s = next((c for c in df2.columns if df2[c].dtype == object), None)
    if not col_s:
        raise ValueError("Coluna de setor não encontrada")
    others = [c for c in df2.columns if c != col_s]
    pri = [c for c in others if re.search(r"percent|particip|%", c, flags=re.IGNORECASE)]
    col_val = None
    for c in pri + others:
        if pd.to_numeric(df2[c], errors="coerce").notna().sum() > 0:
            col_val = c; break
    if not col_val:
        raise ValueError("Coluna de valores não encontrada")
    out = pd.DataFrame({"Setor": df2[col_s].astype(str).str.strip(),
                         "Valor": pd.to_numeric(df2[col_val], errors="coerce")}).dropna()
    out = out[out["Valor"] > 0].sort_values("Valor", ascending=False)
    return out[~out["Setor"].str.lower().str.contains(r"extrat|mineral", na=False)]


# 12) Patentes por setor — barras, depositantes e donuts
TONS_G = ["#064E3B", "#065F46", "#047857", "#059669", "#10B981", "#34D399", "#6EE7B7"]


def _donut_patentes(ax, counts, title, oculto):
    labels = list(counts.index)
    vals = counts.values
    n = len(vals)
    cols = [TONS_G[int(round(i * (len(TONS_G) - 1) / max(n - 1, 1)))] for i in range(n)]
    pct_fn = (lambda p: f"{p:.1f}%" if p >= 3.0 else "") if not oculto else None
    pie = ax.pie(vals, labels=None, autopct=pct_fn, startangle=90, pctdistance=0.78,
                 colors=cols, radius=1.52 if oculto else 1.12,
                 wedgeprops={"width": 0.48, "edgecolor": "white", "linewidth": 2.2},
                 textprops={"fontsize": 10})
    wdg = pie[0]
    ll = [_short_label(v, 32) for v in labels] if not oculto else [""] * len(labels)
    lg = ax.legend(wdg, ll, loc="center left", bbox_to_anchor=(1.04, 0.5),
                   frameon=False, fontsize=9.5, handlelength=1.2,
                   handletextpad=0.5 if not oculto else 0.0, borderaxespad=0.0)
    if oculto:
        lg.set_visible(False)
    ax.set(aspect="equal")
    if not oculto:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=14)
        _add_frame(ax)
    else:
        ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.75, 1.75)
    return lg


def plotar_patentes(df, path, oculto=False):
    T = 15
    plt.rcParams.update({"font.size": T, "axes.titlesize": T + 2, "axes.labelsize": T,
                          "xtick.labelsize": T - 2, "ytick.labelsize": T - 2,
                          "legend.fontsize": T - 3})

    anos = sorted(df["Ano"].dropna().unique().astype(int).tolist())
    cont_ano = df.groupby("Ano").size().reindex(anos, fill_value=0)
    yoy = cont_ano.pct_change() * 100

    dep_counts = df["Depositante"].dropna().value_counts().head(14)
    dep_labels = [_abbrev_depositante(n) for n in dep_counts.index]
    dep_vals   = dep_counts.values

    status_c = df["Status"].dropna().value_counts()
    natjur_c = df["Natureza Jurídica"].dropna().value_counts()
    cat_c    = df["Categoria"].dropna().value_counts()

    fig = plt.figure(figsize=(32, 22), facecolor="white")
    gs  = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.45, 1.0],
                             hspace=0.35, wspace=0.32)
    ax_bar  = fig.add_subplot(gs[0, 0:2])
    ax_dep  = fig.add_subplot(gs[0, 2])
    ax_s    = fig.add_subplot(gs[1, 0])
    ax_nj   = fig.add_subplot(gs[1, 1])
    ax_cat  = fig.add_subplot(gs[1, 2])

    # 12a) Barras verticais — contagem por ano
    vmax_c = cont_ano.max() if cont_ano.max() > 0 else 1
    bar_cols = [TONS_G[min(int((v / vmax_c) * (len(TONS_G) - 1)), len(TONS_G) - 1)]
                for v in cont_ano.values]
    barras = ax_bar.bar(range(len(anos)), cont_ano.values, color=bar_cols,
                        edgecolor="white", linewidth=0.7, width=0.72, zorder=3)

    ax2 = ax_bar.twinx()
    valid_mask = ~np.isnan(yoy.values)
    ax2.plot(np.array(range(len(anos)))[valid_mask], yoy.values[valid_mask],
             color="#064E3B", lw=2.4, ls="-", marker="o", ms=5,
             markerfacecolor="#064E3B", markeredgewidth=0, label="Var. % a.a.", zorder=4)
    ax2.axhline(0, color="#AAA", lw=0.9, ls=":", zorder=1)
    ax2.set_ylabel("Variação % a.a.", fontsize=T - 1)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.grid(visible=False)
    leg_line = ax2.legend(loc="upper right", frameon=False, fontsize=T - 3)

    ax_bar.set_xticks(range(len(anos)))
    ax_bar.set_xticklabels([str(a) for a in anos], rotation=45, ha="right", fontsize=T - 3)
    ax_bar.set_ylabel("Nº de Patentes", fontsize=T)
    ax_bar.set_xlabel("Ano", fontsize=T)
    ax_bar.set_title("Evolução de Patentes por Ano", fontsize=T + 4, fontweight="bold", pad=16)
    ax_bar.grid(axis="y", ls="--", lw=0.5, alpha=0.3)
    ax_bar.grid(axis="x", visible=False)
    ax_bar.spines[["top"]].set_visible(False)
    for s in list(ax_bar.spines.values()) + list(ax2.spines.values()):
        s.set_edgecolor("#CCC"); s.set_linewidth(1.1)

    txts_bar = []
    for b, v in zip(barras, cont_ano.values):
        if v > 0:
            txts_bar.append(ax_bar.text(
                b.get_x() + b.get_width() / 2, v + vmax_c * 0.012,
                str(int(v)), ha="center", va="bottom", fontsize=T - 5, color="#047857"))

    # 12b) Barras horizontais — principais depositantes
    vmax_d = dep_vals[0] if len(dep_vals) else 1
    h_cols = [TONS_G[min(int((v / vmax_d) * (len(TONS_G) - 1)), len(TONS_G) - 1)]
              for v in reversed(dep_vals)]
    dep_vals_r  = list(reversed(dep_vals))
    dep_labels_r = list(reversed(dep_labels))
    barras_h = ax_dep.barh(range(len(dep_labels_r)), dep_vals_r, color=h_cols,
                            edgecolor="white", linewidth=0.7, height=0.72, zorder=3)
    ax_dep.set_yticks(range(len(dep_labels_r)))
    ax_dep.set_yticklabels(dep_labels_r, fontsize=T - 4)
    ax_dep.set_xlabel("Nº de Patentes", fontsize=T - 1)
    ax_dep.set_title("Principais Depositantes", fontsize=T + 2, fontweight="bold", pad=16)
    ax_dep.grid(axis="x", ls="--", lw=0.5, alpha=0.3)
    ax_dep.grid(axis="y", visible=False)
    ax_dep.spines[["top", "right"]].set_visible(False)
    for s in ax_dep.spines.values():
        s.set_edgecolor("#CCC"); s.set_linewidth(1.1)

    txts_dep = []
    for b, v in zip(barras_h, dep_vals_r):
        txts_dep.append(ax_dep.text(
            v + vmax_d * 0.012, b.get_y() + b.get_height() / 2,
            str(int(v)), ha="left", va="center", fontsize=T - 5, color="#047857"))

    # 12c) Donuts — Status, Natureza Jurídica, Categoria
    lgs_d = []
    lgs_d.append(_donut_patentes(ax_s,   status_c,  "Status",            oculto))
    lgs_d.append(_donut_patentes(ax_nj,  natjur_c,  "Natureza Jurídica", oculto))
    lgs_d.append(_donut_patentes(ax_cat, cat_c,     "Categoria",         oculto))

    if oculto:
        for a in (ax_bar, ax_dep):
            a.title.set_visible(False)
            a.xaxis.label.set_visible(False)
            a.yaxis.label.set_visible(False)
            for lb in a.get_xticklabels() + a.get_yticklabels():
                lb.set_visible(False)
        ax2.yaxis.label.set_visible(False)
        for lb in ax2.get_yticklabels():
            lb.set_visible(False)
        for t in txts_bar + txts_dep:
            t.set_visible(False)
        for t in leg_line.get_texts():
            t.set_visible(False)

    fig.subplots_adjust(left=0.05, right=0.88, top=0.93, bottom=0.08)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="jpg", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 13) Mapas coropletos de custo médio por região (2×2)
CMAP_CUSTO = LinearSegmentedColormap.from_list(
    "custo", ["#FFF3E0", "#FFB347", "#E05C00", "#7F1D00"])


def plotar_mapa_custo(df_custo, municipios, path, oculto=False):
    T = 13

    # 13a) Dissolve municípios em regiões e obtém pontos representativos
    mun = municipios.copy()
    mun["regiao_p"] = mun["SIGLA_UF"].map(UF_TO_REGIAO)
    regioes_geo = mun.dissolve(by="regiao_p", as_index=False)
    rp = regioes_geo.geometry.representative_point()
    regioes_geo["cx"] = rp.x
    regioes_geo["cy"] = rp.y
    adj = {"Norte": (-2.5, 0.5)}

    df = df_custo.copy()
    df["regiao_p"] = df["estado"].map(UF_TO_REGIAO)
    df["preco"] = pd.to_numeric(df["valorTotalCompra"], errors="coerce")

    items = df["Categoria_Equipamento"].dropna().unique()[:4]

    fig, axes = plt.subplots(2, 2, figsize=(26, 22), facecolor="white")
    axes_flat = axes.flatten()
    mnx, mny, mxx, mxy = regioes_geo.total_bounds

    for ax, item in zip(axes_flat, items):
        # 13b) Média de preço por região para cada item
        sub = (df[df["Categoria_Equipamento"] == item]
               .groupby("regiao_p")["preco"].mean().reset_index())
        sub.columns = ["regiao_p", "valor"]
        geo = regioes_geo.merge(sub, on="regiao_p", how="left")

        has_data = geo["valor"].notna()
        vmax_i = geo.loc[has_data, "valor"].max() if has_data.any() else 1
        norm_i = Normalize(vmin=0, vmax=vmax_i)

        geo_d = geo[has_data].copy()

        # 13c) Base cinza + camada colorida
        regioes_geo.plot(ax=ax, color="#E4E4E4", edgecolor="white", linewidth=1.2, zorder=1)
        if not geo_d.empty:
            geo_d.plot(ax=ax, column="valor", cmap=CMAP_CUSTO,
                       vmin=0, vmax=vmax_i,
                       edgecolor="white", linewidth=1.2, legend=False, zorder=2)

        # 13d) Rótulos de região com valor médio
        if not oculto:
            for _, row in geo.iterrows():
                off = adj.get(str(row["regiao_p"]), (0, 0))
                lx, ly = row["cx"] + off[0], row["cy"] + off[1]
                v = row["valor"]
                rname = str(row["regiao_p"])
                if pd.notna(v):
                    v_str = (f"R$ {v:,.2f}"
                             .replace(",", "X").replace(".", ",").replace("X", "."))
                    lbl = f"{rname}\n{v_str}"
                    txt_col = "white" if norm_i(v) > 0.55 else "#222222"
                else:
                    lbl = rname; txt_col = "#555555"
                ax.text(lx, ly, lbl, ha="center", va="center",
                        fontsize=T - 1, fontweight="bold", color=txt_col, zorder=5)

        item_title = str(item).title()
        if not oculto:
            ax.set_title(f"Valor Médio — {item_title}",
                         fontsize=T + 1, fontweight="bold", pad=8)
        ax.set_axis_off()
        ax.set_xlim(mnx - 1.5, mxx + 1.5)
        ax.set_ylim(mny - 0.8, mxy + 0.8)

        # 13e) Barra de cores (escuro = mais caro)
        sm = plt.cm.ScalarMappable(cmap=CMAP_CUSTO, norm=norm_i)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical",
                            fraction=0.028, pad=0.02, aspect=22, shrink=0.72)
        if not oculto:
            cbar.ax.yaxis.set_major_formatter(
                FuncFormatter(lambda x, _: (f"R$ {x:,.0f}"
                                            .replace(",", "X").replace(".", ",").replace("X", "."))))
            cbar.ax.tick_params(labelsize=T - 3)
            cbar.set_label("Valor (R$)", fontsize=T - 2)
        else:
            cbar.ax.set_yticks([])

    if not oculto:
        fig.suptitle("Valor Médio de Equipamentos Semicondutores por Região (R$)",
                     fontsize=T + 5, fontweight="bold", y=1.005)

    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.01, right=0.90, top=0.96, bottom=0.02, wspace=0.22, hspace=0.14)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, format="jpg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 14) Função principal — leitura de dados e geração de todas as figuras
def main():
    FIG.mkdir(parents=True, exist_ok=True)

    # 14a) Balança comercial SP
    df_bc = pd.read_excel(EXCEL, sheet_name="BC_SP")
    df_sp = df_bc[df_bc["UF do Produto"] == UF_ALVO].copy()
    df_pib = pd.read_excel(EXCEL, sheet_name="PIB_Setor_SP")
    df_month = transformar_series(df_sp)
    plotar_bc_sp(df_month, df_pib, FIG / "bc_sp.jpg")
    plotar_bc_sp(df_month, df_pib, FIG / "bc_sp_oculto.jpg", oculto=True)
    ind = gerar_indicadores(df_month)
    ind.to_csv(FIG / "ind_bc.csv", index=False, encoding="utf-8-sig")
    print("BC_SP: OK")

    # 14b) Empregos SP
    df_emp = preparar_empregos(pd.read_excel(EXCEL, sheet_name="Empregos_SP"))
    df_set = preparar_setor(pd.read_excel(EXCEL, sheet_name="Empregos_Setor_SP"))
    plotar_empregos(df_emp, df_set, FIG / "emp_sp.jpg")
    plotar_empregos(df_emp, df_set, FIG / "emp_sp_oculto.jpg", oculto=True)
    print("Empregos: OK")

    # 14c) Teses e dissertações
    df_teses = pd.read_excel(EXCEL, sheet_name="Dissertacoes_Teses")
    df_teses["ano_base"] = pd.to_numeric(df_teses["ano_base"], errors="coerce")
    df_teses = df_teses[df_teses["ano_base"].between(2013, 2024)].copy()
    df_teses["ano_base"] = df_teses["ano_base"].astype(int)
    plotar_teses(df_teses, FIG / "teses.jpg")
    plotar_teses(df_teses, FIG / "teses_oculto.jpg", oculto=True)
    print("Teses: OK")

    # 14d) Mapa cadeia produtiva
    local_fixes = {"SANTO AGOSTINHO": "IPOJUCA", "ANTONIO POSSE": "SANTO ANTONIO DE POSSE"}
    cadeia = pd.read_excel(EXCEL, sheet_name="Cadeia_Setor")
    cadeia = cadeia[["Estado", "Etapa", "Empresa", "Local"]].dropna().copy()
    cadeia["uf"] = cadeia["Estado"].map(norm_geo)
    cadeia["etapa"] = cadeia["Etapa"].map(norm_geo)
    cadeia["empresa"] = cadeia["Empresa"].astype(str).str.strip()
    cadeia["local_norm"] = cadeia["Local"].map(norm_geo).replace(local_fixes)

    municipios = gpd.read_file(SHAPE)
    municipios["uf"] = municipios["SIGLA_UF"].map(norm_geo)
    municipios["mun_norm"] = municipios["NM_MUN"].map(norm_geo)
    merged = cadeia.merge(municipios[["uf", "mun_norm", "geometry"]], how="left",
                          left_on=["uf", "local_norm"], right_on=["uf", "mun_norm"])
    nf = merged[merged["geometry"].isna()][["Estado", "Empresa", "Local"]]
    if not nf.empty:
        print("Locais não encontrados:", nf.to_string(index=False))
    merged = merged.dropna(subset=["geometry"]).copy()
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=municipios.crs)
    merged["geometry"] = merged.geometry.representative_point()
    merged["x"] = merged.geometry.x; merged["y"] = merged.geometry.y
    merged = spread_pts(merged)
    estados = municipios.dissolve(by="SIGLA_UF", as_index=False)
    estados["x"] = estados.geometry.representative_point().x
    estados["y"] = estados.geometry.representative_point().y

    plotar_mapa(estados, merged, FIG / "mapa.jpg")
    plotar_mapa(estados, merged, FIG / "mapa_oculto.jpg", oculto=True)
    print("Mapa: OK")

    # 14e) Patentes do setor
    df_patentes = pd.read_excel(EXCEL, sheet_name="Patentes_Setor")
    plotar_patentes(df_patentes, FIG / "patentes.jpg")
    plotar_patentes(df_patentes, FIG / "patentes_oculto.jpg", oculto=True)
    print("Patentes: OK")

    # 14f) Mapas de custo de equipamentos
    df_custo_eq = pd.read_excel(EXCEL, sheet_name="Custo_Equipamentos")
    plotar_mapa_custo(df_custo_eq, municipios, FIG / "mapa_custo.jpg")
    plotar_mapa_custo(df_custo_eq, municipios, FIG / "mapa_custo_oculto.jpg", oculto=True)
    print("Custo Equipamentos: OK")

    print("Todas as figuras geradas em:", FIG)


if __name__ == "__main__":
    main()
