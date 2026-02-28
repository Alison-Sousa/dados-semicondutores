# dados-semicondutores

Projeto de Pesquisa FAPESP — Indústria de Semicondutores no Brasil.

> **Site do projeto:** O dashboard interativo (**SemiBrasil**) está em um repositório separado.  
> Acesse em: **[github.com/Alison-Sousa/semi-br](https://github.com/Alison-Sousa/semi-br)**

## Estrutura

```
Bases/                         Dados de entrada
  base_de_dados.xlsx           Planilha principal (abas: BC_SP, PIB_Setor_SP, Empregos_SP, Empregos_Setor_SP, Cadeia_Setor, Patentes_Setor, Custo_Equipamentos, Dissertacoes_Teses, Incentivos, Centros, Laboratórios, Empregos, Detalhamento)
  capes_raw/                   CSVs brutos da CAPES (pesado, não sobe no GitHub)

BR_Municipios_2024/            Shapefile dos municípios brasileiros (pesado, não sobe no GitHub)

Códigos/                       Scripts de processamento e visualização
  api.py                       Extrai dados de compras governamentais via API ComprasNet
  dados.py                     Consolida CSVs CAPES, atualiza teses_dissertacoes.xlsx e gera CSVs de setores/emprego (Tabela23.xls, RAIS.json)
  graficos.py                  Gera todas as figuras, versões ocultas e indicadores
  xlsx_capes.r                 Baixa dados CAPES via pacote capesR e filtra por palavras-chave

Figuras/                       Saídas geradas por graficos.py
  bc_sp.jpg                    Balança comercial de semicondutores em SP + PIB por setor
  bc_sp_oculto.jpg             Versão sem textos (apresentação)
  emp_sp.jpg                   Empregos no setor em SP + empregos por setor econômico
  emp_sp_oculto.jpg            Versão sem textos
  teses.jpg                    Teses e dissertações por ano (com contagem), região, IES e área
  teses_oculto.jpg             Versão sem textos
  mapa.jpg                     Mapa da cadeia produtiva (projeto de CI e encapsulamento/teste)
  mapa_oculto.jpg              Versão sem textos
  patentes.jpg                 Patentes por ano + depositantes + donuts Status/Natureza/Categoria
  patentes_oculto.jpg          Versão sem textos
  mapa_custo.jpg               4 mapas 2×2 com valor médio por região (escuro = mais caro)
  mapa_custo_oculto.jpg        Versão sem textos
  ind_bc.csv                   Indicadores da balança comercial

  setores_2010_2023.csv        Participação dos setores (Tabela23.xls, 2010-2023)
  industria_sp_ano.csv         Participação anual da indústria de transformação em SP (RAIS.json)
  setores_sp_ano_recente.csv   Participação por setor em SP no ano mais recente (RAIS.json)

extract_data.py                Extrai dados do Excel e exporta site_data.json para o site
site_data.json                 JSON com todos os dados usados pelo dashboard
```

## Scripts

### `extract_data.py`

Extrai todos os dados de `base_de_dados.xlsx` e exporta como `site_data.json`, o JSON consumido pelo dashboard SemiBrasil. Processa balança comercial, empregos, PIB por setor, cadeia produtiva, patentes, custos de equipamentos e teses/dissertações.

**Uso:** `python extract_data.py`

### `api.py`

Extrai dados de compras governamentais de equipamentos médicos com semicondutores via API de Dados Abertos (ComprasNet). Busca catálogo de itens por PDM/classe e coleta histórico de preços. Gera `Base_Consolidada_Equipamentos_Medicos.xlsx` com colunas como `Categoria_Equipamento`, `precoUnitario`, `quantidade` e `valorTotalCompra`.

**Uso:** `python Códigos/api.py`

### `dados.py`

Consolida CSVs brutos da CAPES (pasta `Bases/capes_raw/`) em uma única planilha `teses_dissertacoes.xlsx`. Filtra por palavras-chave de semicondutores (ex: semicondutor, CMOS, VLSI, wafer, fotolitografia), remove duplicatas e mantém as colunas: `ano_base`, `tipo`, `titulo`, `ies`, `area`, `regiao`.

**Uso:** `python Códigos/dados.py`

### `graficos.py`

Gera todas as figuras do projeto, incluindo as versões "oculto" (sem textos/legendas). Lê `base_de_dados.xlsx` e o shapefile de municípios. Produz 12 JPGs (6 normais + 6 oculto) e 1 CSV de indicadores.

| Figura | Fonte | Descrição |
|--------|-------|-----------|
| `bc_sp.jpg` | BC_SP + PIB_Setor_SP | Barras exportação/importação + linha de saldo + donut PIB |
| `emp_sp.jpg` | Empregos_SP + Empregos_Setor_SP | Barras de participação + donut por setor |
| `teses.jpg` | Dissertacoes_Teses | Barras teses vs dissertações (com contagem) + 3 donuts |
| `mapa.jpg` | Cadeia_Setor + shapefile | Mapa do Brasil com empresas de CI e teste |
| `patentes.jpg` | Patentes_Setor | Barras + variação % + depositantes + 3 donuts |
| `mapa_custo.jpg` | Custo_Equipamentos | 4 mapas coropléticos por região — usa valorTotalCompra (escuro = mais caro) |
| `ind_bc.csv` | BC_SP | Exportação, importação, saldo e razão |
| `setores_2010_2023.csv` | Tabela23.xls | Participação dos setores econômicos (2010-2023) |
| `industria_sp_ano.csv` | RAIS.json | Participação anual da indústria de transformação em SP |
| `setores_sp_ano_recente.csv` | RAIS.json | Participação por setor em SP no ano mais recente |

**Uso:** `python Códigos/graficos.py`

### `xlsx_capes.r`

Script R que baixa microdados CAPES (2013–2026) via pacote `capesR`, filtra por palavras-chave e salva em xlsx. Serve como fonte primária; `dados.py` atualiza incrementalmente.

**Uso:** `Rscript Códigos/xlsx_capes.r`

## Observações

- Arquivos pesados (shapefile, CSVs CAPES) não são versionados no GitHub.
- Versões "oculto" omitem textos e legendas — para apresentações com textos manuais.
- Nos mapas de custo, regiões em cinza não possuem dados; a cor mais escura indica o maior valor médio.
- Centróides dos mapas usam projeção SIRGAS 2000 (EPSG:5880) para evitar distorções.

## Download dos dados de origem

- **Shapefile de Municípios (IBGE 2024):** [Malhas Territoriais — IBGE](https://www.ibge.gov.br/geociencias/organizacao-do-territorio/malhas-territoriais/15774-malhas.html)  
  Baixar a malha de **Municípios** em nível Brasil.

- **Catálogo de Teses e Dissertações CAPES:** [Dados Abertos CAPES](https://dadosabertos.capes.gov.br/dataset/2021-a-2024-catalogo-de-teses-e-dissertacoes-brasil)  
  Colocar os CSVs em `Bases/capes_raw/`.
