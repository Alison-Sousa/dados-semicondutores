"""Extract all data from Excel files and export as JSON for the website."""
import json, pandas as pd, numpy as np, re

EXCEL = "Bases/base_de_dados.xlsx"
TESES = "Bases/teses_dissertacoes.xlsx"

out = {}

# 1) BC_SP - Balança Comercial
df_bc = pd.read_excel(EXCEL, sheet_name="BC_SP")
df_sp = df_bc[df_bc["UF do Produto"] == "São Paulo"].copy()

def extrair_mes(v):
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else 1

df_sp["Mes_Num"] = df_sp["Mês"].apply(extrair_mes)
exp_c = [c for c in df_sp.columns if c.startswith("Exportação - ") and "Valor US$ FOB" in c]
imp_c = [c for c in df_sp.columns if c.startswith("Importação - ") and "Valor US$ FOB" in c]

def melt(df, cols, nome):
    t = df[["Mes_Num"] + cols].melt(id_vars=["Mes_Num"], value_vars=cols, var_name="Col", value_name=nome)
    t["Ano"] = t["Col"].str.extract(r"(\d{4})").astype(int)
    return t.drop(columns=["Col"])

e = melt(df_sp, exp_c, "Exp")
i = melt(df_sp, imp_c, "Imp")
m = pd.merge(e, i, on=["Ano", "Mes_Num"], how="outer").fillna(0)
yearly = m.groupby("Ano")[["Exp", "Imp"]].sum().reset_index()
yearly["Saldo"] = yearly["Exp"] - yearly["Imp"]
yearly = yearly[(yearly["Ano"] >= 2000) & (yearly["Ano"] <= 2025)].sort_values("Ano")

out["bc_sp"] = {
    "anos": yearly["Ano"].tolist(),
    "exportacao": [round(v/1e9, 2) for v in yearly["Exp"].tolist()],
    "importacao": [round(v/1e9, 2) for v in yearly["Imp"].tolist()],
    "saldo": [round(v/1e9, 2) for v in yearly["Saldo"].tolist()]
}

# 2) PIB_Setor_SP
df_pib = pd.read_excel(EXCEL, sheet_name="PIB_Setor_SP")
col_s = next((c for c in df_pib.columns if "setor" in c.lower()), df_pib.columns[0])
col_p = next((c for c in df_pib.columns if "part" in c.lower() or "pib" in c.lower() or "%" in c), df_pib.columns[1])
df_pib[col_p] = pd.to_numeric(df_pib[col_p], errors="coerce").fillna(0)
df_pib = df_pib[df_pib[col_p] > 0].sort_values(col_p, ascending=False)
out["pib_setor"] = {
    "setores": df_pib[col_s].tolist(),
    "valores": df_pib[col_p].tolist()
}

# 3) Empregos_SP
df_emp = pd.read_excel(EXCEL, sheet_name="Empregos_SP")
c_ano = df_emp.columns[0]
c_val = df_emp.columns[1]
df_emp[c_val] = pd.to_numeric(df_emp[c_val], errors="coerce")
df_emp = df_emp.dropna().sort_values(c_ano)
out["empregos_sp"] = {
    "anos": [int(x) for x in df_emp[c_ano].tolist()],
    "valores": df_emp[c_val].tolist()
}

# 4) Empregos_Setor_SP
df_set = pd.read_excel(EXCEL, sheet_name="Empregos_Setor_SP")
c_s = df_set.columns[0]
c_v = df_set.columns[1]
df_set[c_v] = pd.to_numeric(df_set[c_v], errors="coerce")
df_set = df_set.dropna().sort_values(c_v, ascending=False)
out["empregos_setor"] = {
    "setores": df_set[c_s].tolist(),
    "valores": df_set[c_v].tolist()
}

# 5) Cadeia_Setor
df_cad = pd.read_excel(EXCEL, sheet_name="Cadeia_Setor")
cadeia = []
for _, r in df_cad.iterrows():
    cadeia.append({
        "estado": str(r.get("Estado", "")),
        "etapa": str(r.get("Etapa", "")),
        "empresa": str(r.get("Empresa", "")),
        "sede": str(r.get("Sede", "")),
        "local": str(r.get("Local", "")),
        "modelo": str(r.get("Modelo de Negócio", "")),
        "empregados": str(r.get("Total de Empregados", "")),
        "superior": str(r.get("Com Ensino Superior", "")),
        "pd": str(r.get("P&D", "")),
        "area_total": str(r.get("Área Total (m²)", "")),
        "sala_limpa": str(r.get("Sala Limpa (m²)", "")),
        "comentarios": str(r.get("Comentários", ""))
    })
out["cadeia"] = cadeia

# 6) Incentivos
df_inc = pd.read_excel(EXCEL, sheet_name="Incentivos")
incentivos = []
for _, r in df_inc.iterrows():
    pais = str(r.get("País", ""))
    if pais == "nan" or not pais.strip():
        continue
    incentivos.append({
        "pais": pais,
        "lei": str(r.get("Lei / Base Legal", "")),
        "beneficio": str(r.get("Benefício Fiscal Principal (%)", "")),
        "estrategia": str(r.get("Estratégia", "")),
        "fonte": str(r.get("Fonte", ""))
    })
out["incentivos"] = incentivos

# 7) Centros
df_cen = pd.read_excel(EXCEL, sheet_name="Centros")
centros = []
for _, r in df_cen.iterrows():
    centros.append({
        "nome": str(r.get("Centro de P&D", "")),
        "vinculacao": str(r.get("Vinculação", "")),
        "unidade": str(r.get("Unidade Principal", "")),
        "area_total": str(r.get("Área Total (m²)", "")),
        "sala_limpa": str(r.get("Sala Limpa (m²)", "")),
        "pesquisadores": str(r.get("Pesquisadores (Total)", "")),
        "doutores": str(r.get("Doutores (PhD)", "")),
        "mestres": str(r.get("Mestres (MSc)", "")),
        "receita": str(r.get("Receita (US$ mi)", ""))
    })
out["centros"] = centros

# 8) Laboratórios
df_lab = pd.read_excel(EXCEL, sheet_name="Laboratórios")
labs = []
for _, r in df_lab.iterrows():
    labs.append({
        "universidade": str(r.get("Universidade", "")),
        "laboratorio": str(r.get("Laboratório", "")),
        "areas": str(r.get("Principais Áreas", "")),
        "sala_limpa": str(r.get("Sala Limpa (m²)", "")),
        "inct": str(r.get("Membro INCT", ""))
    })
out["laboratorios"] = labs

# 9) Patentes_Setor
df_pat = pd.read_excel(EXCEL, sheet_name="Patentes_Setor")
# Yearly counts
pc = df_pat.groupby("Ano").size().reset_index(name="count")
pc = pc.sort_values("Ano")
out["patentes_ano"] = {
    "anos": [int(x) for x in pc["Ano"].tolist()],
    "contagem": pc["count"].tolist()
}
# Top depositantes
dep = df_pat["Depositante"].value_counts().head(14)
dep_abbrevs = []
for name in dep.index:
    t = str(name).strip()
    m_cc = re.search(r'-?\[([A-Z]{2})\]', t)
    country = f" [{m_cc.group(1)}]" if m_cc else ""
    t2 = re.sub(r';.*$', '', t)
    t2 = re.sub(r'-?\[.*?\]', '', t2).strip().rstrip('-').strip()
    m_ac = re.search(r'\b(USP|UNICAMP|UFRJ|UFMG|UFSC|UFPR|UFBA|UFRN|UFC|UFG|INPI|INPE|UNB|UFSM|UFAM|PUC)\b', t2.upper())
    if m_ac:
        dep_abbrevs.append(m_ac.group(1) + country)
    else:
        sw = {"DE","DO","DA","DAS","DOS","E","OF","THE","AND","PARA","EM","LA","LOS"}
        words = [w for w in re.split(r'[\s\-]+', t2.upper()) if w and w not in sw and w.isalpha()]
        ac = "".join(w[0] for w in words)
        dep_abbrevs.append((ac if 2 <= len(ac) <= 8 else t2[:16].strip()) + country)

out["patentes_depositantes"] = {
    "nomes": dep_abbrevs,
    "valores": dep.values.tolist()
}
# Status, Natureza, Categoria
out["patentes_status"] = {
    "labels": df_pat["Status"].value_counts().index.tolist(),
    "valores": df_pat["Status"].value_counts().values.tolist()
}
out["patentes_natureza"] = {
    "labels": df_pat["Natureza Jurídica"].value_counts().index.tolist(),
    "valores": df_pat["Natureza Jurídica"].value_counts().values.tolist()
}
out["patentes_categoria"] = {
    "labels": df_pat["Categoria"].value_counts().index.tolist(),
    "valores": df_pat["Categoria"].value_counts().values.tolist()
}

# 10) Custo_Equipamentos
df_custo = pd.read_excel(EXCEL, sheet_name="Custo_Equipamentos")
UF_TO_REGIAO = {
    "AC":"Norte","AM":"Norte","AP":"Norte","PA":"Norte","RO":"Norte","RR":"Norte","TO":"Norte",
    "AL":"Nordeste","BA":"Nordeste","CE":"Nordeste","MA":"Nordeste","PB":"Nordeste",
    "PE":"Nordeste","PI":"Nordeste","RN":"Nordeste","SE":"Nordeste",
    "DF":"Centro-Oeste","GO":"Centro-Oeste","MS":"Centro-Oeste","MT":"Centro-Oeste",
    "ES":"Sudeste","MG":"Sudeste","RJ":"Sudeste","SP":"Sudeste",
    "PR":"Sul","RS":"Sul","SC":"Sul"
}
df_custo["regiao"] = df_custo["estado"].map(UF_TO_REGIAO)
df_custo["preco"] = pd.to_numeric(df_custo["valorTotalCompra"], errors="coerce")
items = df_custo["Categoria_Equipamento"].dropna().unique()[:4].tolist()
custo_data = {}
for item in items:
    sub = df_custo[df_custo["Categoria_Equipamento"] == item].groupby("regiao")["preco"].mean().round(2)
    custo_data[item] = sub.to_dict()
out["custo_equipamentos"] = {"items": items, "data": custo_data}

# 11) Teses e Dissertações
df_teses = pd.read_excel(EXCEL, sheet_name="Dissertacoes_Teses")
df_teses["ano_base"] = pd.to_numeric(df_teses["ano_base"], errors="coerce")
df_teses = df_teses[df_teses["ano_base"].between(2013, 2024)].copy()
t = df_teses["tipo"].str.upper().str.strip()
t = t.replace({"MESTRADO":"DISSERTAÇÃO","MESTRADO PROFISSIONAL":"DISSERTAÇÃO","DOUTORADO":"TESE"})
df_teses["tipo_n"] = t

ct = df_teses.groupby(["ano_base","tipo_n"]).size().unstack(fill_value=0)
out["teses_ano"] = {
    "anos": [int(x) for x in ct.index.tolist()],
    "dissertacao": ct.get("DISSERTAÇÃO", pd.Series(0, index=ct.index)).tolist(),
    "tese": ct.get("TESE", pd.Series(0, index=ct.index)).tolist()
}

# Region counts
reg = df_teses["regiao"].str.upper().str.strip().value_counts()
out["teses_regiao"] = {"labels": reg.index.tolist(), "valores": reg.values.tolist()}

# IES top 5
ies = df_teses["ies"].str.upper().str.strip().value_counts().head(5)
ies_abbr = []
for name in ies.index:
    m = re.search(r'\b(USP|UNICAMP|UFRJ|UFMG|UFSC|UFPR|UFBA|UFRN|UFC|UFG|UFRGS)\b', name)
    if m:
        ies_abbr.append(m.group(1))
    else:
        sw = {"DE","DO","DA","DAS","DOS","E"}
        words = [w for w in name.replace("-"," ").split() if w and w not in sw]
        ies_abbr.append("".join(w[0] for w in words)[:8])
out["teses_ies"] = {"labels": ies_abbr, "valores": ies.values.tolist()}

# Area top 5
area = df_teses["area"].str.upper().str.strip().value_counts().head(5)
out["teses_area"] = {"labels": area.index.tolist(), "valores": area.values.tolist()}

# 12) Empregos RAIS (national data by sector, year, state)
df_emp_rais = pd.read_excel(EXCEL, sheet_name="Empregos")
df_emp_rais["Empregados"] = pd.to_numeric(df_emp_rais["Empregados"], errors="coerce").fillna(0)
# Aggregate by year (nacional)
emp_nac = df_emp_rais.groupby("Ano")["Empregados"].sum().reset_index().sort_values("Ano")
out["empregos_nacional"] = {
    "anos": [int(x) for x in emp_nac["Ano"].tolist()],
    "valores": emp_nac["Empregados"].tolist()
}
# Aggregate by UF
emp_uf = df_emp_rais.groupby("UF")["Empregados"].sum().sort_values(ascending=False)
out["empregos_por_uf"] = {"labels": emp_uf.index.tolist(), "valores": emp_uf.values.tolist()}
# Aggregate by Setor Sebrae
emp_setor = df_emp_rais.groupby("Setores Sebrae")["Empregados"].sum().sort_values(ascending=False)
out["empregos_por_setor"] = {"labels": emp_setor.index.tolist(), "valores": emp_setor.values.tolist()}

# 13) Detalhamento (sources)
df_det = pd.read_excel(EXCEL, sheet_name="Detalhamento")
sources = {}
for _, r in df_det.iterrows():
    aba = str(r.get("Abas", ""))
    ref = str(r.get("Referências Bibliográficas (Links)", ""))
    if aba != "nan" and ref != "nan":
        # Extract first URL (references may start with "1. ")
        url_match = re.search(r'https?://\S+', ref)
        if url_match:
            sources[aba] = url_match.group(0)
out["fontes"] = sources

# 14) Indicadores BC
out["indicadores_bc"] = {
    "total_imp_bilhoes": round(yearly["Imp"].sum() / 1e9, 1),
    "total_exp_bilhoes": round(yearly["Exp"].sum() / 1e9, 1),
    "saldo_bilhoes": round((yearly["Exp"].sum() - yearly["Imp"].sum()) / 1e9, 1),
    "razao": round(yearly["Imp"].sum() / yearly["Exp"].sum(), 1) if yearly["Exp"].sum() > 0 else 0
}

# Serialize
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, (np.ndarray,)): return obj.tolist()
        return super().default(obj)

with open("site_data.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2, cls=NpEncoder)

# ── Generate full CSV files for downloads ──
import os
csv_dir = os.path.join("semi-br", "data")
os.makedirs(csv_dir, exist_ok=True)

# Patentes (all 689 rows, all columns)
df_pat_full = pd.read_excel(EXCEL, sheet_name="Patentes_Setor")
df_pat_full.to_csv(os.path.join(csv_dir, "patentes_semicondutores.csv"), index=False, sep=";", encoding="utf-8-sig")
print(f"  CSV: patentes_semicondutores.csv ({len(df_pat_full)} rows)")

# Teses (all rows, all columns)
df_tes_full = pd.read_excel(EXCEL, sheet_name="Dissertacoes_Teses")
df_tes_full.to_csv(os.path.join(csv_dir, "teses_dissertacoes.csv"), index=False, sep=";", encoding="utf-8-sig")
print(f"  CSV: teses_dissertacoes.csv ({len(df_tes_full)} rows)")

# Empregos (all rows, all columns)
df_emp_full = pd.read_excel(EXCEL, sheet_name="Empregos")
df_emp_full.to_csv(os.path.join(csv_dir, "empregos_industria.csv"), index=False, sep=";", encoding="utf-8-sig")
print(f"  CSV: empregos_industria.csv ({len(df_emp_full)} rows)")

# Balança Comercial
bc_out = pd.DataFrame({
    "Ano": out["bc_sp"]["anos"],
    "Exportação (US$ bi)": out["bc_sp"]["exportacao"],
    "Importação (US$ bi)": out["bc_sp"]["importacao"],
    "Saldo (US$ bi)": out["bc_sp"]["saldo"]
})
bc_out.to_csv(os.path.join(csv_dir, "balanca_comercial_semicondutores.csv"), index=False, sep=";", encoding="utf-8-sig")
print(f"  CSV: balanca_comercial_semicondutores.csv ({len(bc_out)} rows)")

# Cadeia Produtiva
df_cad_full = pd.read_excel(EXCEL, sheet_name="Cadeia_Setor")
df_cad_full.to_csv(os.path.join(csv_dir, "cadeia_produtiva_brasil.csv"), index=False, sep=";", encoding="utf-8-sig")
print(f"  CSV: cadeia_produtiva_brasil.csv ({len(df_cad_full)} rows)")

# Incentivos
df_inc_full = pd.read_excel(EXCEL, sheet_name="Incentivos")
df_inc_full.to_csv(os.path.join(csv_dir, "incentivos_internacionais.csv"), index=False, sep=";", encoding="utf-8-sig")
print(f"  CSV: incentivos_internacionais.csv ({len(df_inc_full)} rows)")

# Centros
df_cen_full = pd.read_excel(EXCEL, sheet_name="Centros")
df_cen_full.to_csv(os.path.join(csv_dir, "centros_pd.csv"), index=False, sep=";", encoding="utf-8-sig")
print(f"  CSV: centros_pd.csv ({len(df_cen_full)} rows)")

# Laboratórios
df_lab_full = pd.read_excel(EXCEL, sheet_name="Laboratórios")
df_lab_full.to_csv(os.path.join(csv_dir, "laboratorios_universitarios.csv"), index=False, sep=";", encoding="utf-8-sig")
print(f"  CSV: laboratorios_universitarios.csv ({len(df_lab_full)} rows)")

# Equipamentos
df_equip_full = pd.read_excel(EXCEL, sheet_name="Custo_Equipamentos")
df_equip_full.to_csv(os.path.join(csv_dir, "custo_equipamentos.csv"), index=False, sep=";", encoding="utf-8-sig")
print(f"  CSV: custo_equipamentos.csv ({len(df_equip_full)} rows)")

print("\nData exported to site_data.json")
print("CSV files exported to semi-br/data/")
print("Keys:", list(out.keys()))
