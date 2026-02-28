# 1) Importações
from pathlib import Path
import pandas as pd

# 2) Caminhos e constantes
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "Bases" / "capes_raw"
OUTPUT = ROOT / "Bases" / "teses_dissertacoes.xlsx"

# 3) Palavras-chave para filtragem de títulos
KEYWORDS = [
    "semicondutor", "semiconductor",
    "circuito integrado", "integrated circuit",
    "microeletrônica", "microeletronica", "microelectronics",
    "transistor", "MOSFET", "CMOS", "VLSI",
    "chip", "wafer", "silício", "silicon",
    "fotolitografia", "litografia", "lithography",
    "nanoeletr", "nanoelectr",
    "dispositivo eletrônico", "electronic device",
    "heterojunção", "heterojuncao",
    "junção pn", "p-n junction",
    "bandgap", "band gap",
    "dopagem", "doping",
    "epitaxia", "epitaxy",
    "sala limpa", "cleanroom",
    "GaAs", "InP", "SiC", "GaN",
]

# 4) Colunas finais do xlsx de saída
FINAL_COLUMNS = [
    "ano_base", "tipo", "titulo", "ies", "area", "regiao",
]


# 5) Leitura e filtragem de um arquivo CSV bruto da CAPES
def load_and_filter(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", encoding="latin-1", dtype=str, on_bad_lines="skip")
    df.columns = df.columns.str.strip().str.lower()

    col_map = {}
    for c in df.columns:
        low = c.lower()
        if "ano" in low:
            col_map["ano_base"] = c
        elif "tipo" in low:
            col_map["tipo"] = c
        elif "titulo" in low or "nome" in low:
            col_map["titulo"] = c
        elif "ies" in low or "instituicao" in low or "instituição" in low:
            col_map["ies"] = c
        elif "area" in low or "área" in low:
            col_map["area"] = c
        elif "regiao" in low or "região" in low:
            col_map["regiao"] = c

    df = df.rename(columns={v: k for k, v in col_map.items()})
    present = [c for c in FINAL_COLUMNS if c in df.columns]
    df = df[present].copy()

    if "titulo" in df.columns:
        titulo_up = df["titulo"].fillna("").str.upper()
        mask = pd.Series(False, index=df.index)
        for kw in KEYWORDS:
            mask |= titulo_up.str.contains(kw.upper(), na=False, regex=False)
        df = df[mask]

    return df


# 6) Normalização para remoção de duplicatas
def normalize_for_dedup(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "titulo" in out.columns:
        out["_tit_norm"] = (out["titulo"].fillna("")
                            .str.upper().str.strip()
                            .str.replace(r"\s+", " ", regex=True))
    return out


# 7) Execução principal — consolida CSVs e gera xlsx final
def main():
    if not RAW_DIR.exists():
        print(f"Pasta {RAW_DIR} não encontrada. Execute xlsx_capes.r primeiro.")
        return

    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        print("Nenhum CSV encontrado em", RAW_DIR)
        return

    frames = []
    for p in csvs:
        try:
            chunk = load_and_filter(p)
            if not chunk.empty:
                frames.append(chunk)
                print(f"  {p.name}: {len(chunk)} registros filtrados")
        except Exception as e:
            print(f"  ERRO em {p.name}: {e}")

    if not frames:
        print("Nenhum registro encontrado.")
        return

    all_df = pd.concat(frames, ignore_index=True)
    all_df = normalize_for_dedup(all_df)

    if "_tit_norm" in all_df.columns:
        before = len(all_df)
        all_df = all_df.drop_duplicates(subset=["_tit_norm"])
        print(f"Duplicatas removidas: {before - len(all_df)}")
        all_df = all_df.drop(columns=["_tit_norm"])

    present = [c for c in FINAL_COLUMNS if c in all_df.columns]
    all_df = all_df[present]

    all_df.to_excel(OUTPUT, index=False, engine="openpyxl")
    print(f"Salvo {len(all_df)} registros em {OUTPUT}")


if __name__ == "__main__":

    main()

    # --- NOVO: Processamento de Tabela23.xls e RAIS.json ---
    import json
    import re
    from pandas import ExcelWriter

    BASES_DIR = ROOT / "Bases"
    # 1. Processa Tabela23.xls
    tabela23_path = BASES_DIR / "Tabela23.xls"
    setores = {
        '1': 'Total das Atividades',
        '2': 'Agropecuária',
        '3': 'Indústrias extrativas',
        '4': 'Indústrias de transformação',
        '5': 'Eletricidade e gás, água, esgoto',
        '6': 'Construção',
        '7': 'Comércio e reparação de veículos',
        '8': 'Transporte, armazenagem e correio',
        '9': 'Informação e comunicação',
        '10': 'Atividades financeiras e seguros',
        '11': 'Atividades imobiliárias',
        '12': 'Administração, saúde e educação públicas',
        '13': 'Outros serviços'
    }
    somas_setores = {}
    if tabela23_path.exists():
        xls = pd.ExcelFile(tabela23_path)
        for sheet_name in xls.sheet_names:
            match = re.search(r'23\.(\d+)', sheet_name)
            if match:
                num_tabela = match.group(1)
                nome_setor = setores.get(num_tabela, f"Setor {num_tabela}")
                df_temp = pd.read_excel(xls, sheet_name=sheet_name, skiprows=5)
                if df_temp.shape[1] >= 6:
                    df_temp = df_temp.iloc[:, [0, 5]].copy()
                    df_temp.columns = ['Ano', 'Valor']
                    df_temp = df_temp.dropna(subset=['Ano'])
                    df_temp['Ano'] = pd.to_numeric(df_temp['Ano'], errors='coerce')
                    df_temp = df_temp.dropna(subset=['Ano'])
                    df_temp['Valor'] = pd.to_numeric(df_temp['Valor'], errors='coerce')
                    soma_total = df_temp['Valor'].sum()
                    somas_setores[nome_setor] = soma_total
        df_tabela23 = pd.DataFrame(list(somas_setores.items()), columns=['Setor', 'Valor Total (2010-2023)'])
        valor_total_geral = df_tabela23[df_tabela23['Setor'] == 'Total das Atividades']['Valor Total (2010-2023)'].values[0]
        df_tabela23['Participação (%)'] = (df_tabela23['Valor Total (2010-2023)'] / valor_total_geral) * 100
        df_tabela23 = df_tabela23[df_tabela23['Setor'] != 'Total das Atividades'].copy()
        df_tabela23 = df_tabela23.sort_values(by='Participação (%)', ascending=False).reset_index(drop=True)
    else:
        df_tabela23 = pd.DataFrame()

    # 2. Processa RAIS.json
    rais_path = BASES_DIR / "RAIS.json"
    if rais_path.exists():
        with open(rais_path, encoding="utf-8") as f:
            data = json.load(f)
        df_rais = pd.DataFrame(data['data'], columns=data['headers'])
        # Aba 1: Participação anual da Indústria de Transformação em SP
        df_sp = df_rais[df_rais['State'] == 'São Paulo']
        df_agg = df_sp.groupby(['Year', 'Sector'])['Workers'].sum().reset_index()
        total_por_ano = df_agg.groupby('Year')['Workers'].sum().reset_index().rename(columns={'Workers': 'Total_Geral'})
        df_industria = df_agg[df_agg['Sector'] == 'Industria Transformação'].copy()
        df_industria = df_industria[['Year', 'Workers']].rename(columns={'Workers': 'Valor_Industria'})
        tabela_industria = pd.merge(df_industria, total_por_ano, on='Year')
        tabela_industria['Participação (%)'] = (tabela_industria['Valor_Industria'] / tabela_industria['Total_Geral']) * 100
        tabela_industria = tabela_industria.rename(columns={'Year': 'Ano'})
        tabela_industria = tabela_industria[['Ano', 'Valor_Industria', 'Total_Geral', 'Participação (%)']]
        # Aba 2: Participação por setor em SP no ano mais recente
        ano_recente = df_sp['Year'].max()
        df_sp_ano = df_sp[df_sp['Year'] == ano_recente].copy()
        df_setor = df_sp_ano.groupby('Sector')['Workers'].sum().reset_index()
        total_trabalhadores = df_setor['Workers'].sum()
        df_setor['Participação no Emprego (%)'] = (df_setor['Workers'] / total_trabalhadores) * 100
        df_setor = df_setor.rename(columns={'Sector': 'Setor Econômico'})
        tabela_setor = df_setor[['Setor Econômico', 'Participação no Emprego (%)']].copy()
        tabela_setor = tabela_setor.sort_values(by='Participação no Emprego (%)', ascending=False).reset_index(drop=True)
    else:
        tabela_industria = pd.DataFrame()
        tabela_setor = pd.DataFrame()

    # 3. Salva cada resultado como CSV na pasta Figuras
    figuras_dir = ROOT / "Figuras"
    figuras_dir.mkdir(parents=True, exist_ok=True)
    if not df_tabela23.empty:
        df_tabela23.to_csv(figuras_dir / "setores_2010_2023.csv", index=False)
    if not tabela_industria.empty:
        tabela_industria.to_csv(figuras_dir / "industria_sp_ano.csv", index=False)
    if not tabela_setor.empty:
        tabela_setor.to_csv(figuras_dir / "setores_sp_ano_recente.csv", index=False)
    print(f"Arquivos CSV gerados em {figuras_dir}")
