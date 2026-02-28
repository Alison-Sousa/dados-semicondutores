import requests
import pandas as pd
import time
import os

# ==============================================================================
# 1) CONFIGURAÇÃO DOS GRUPOS DE MATERIAIS (PDMs E CLASSES)
# ==============================================================================
# Lista com todos os equipamentos médicos mapeados para a pesquisa
equipamentos = [
    {"pdm": 13552, "classe": 6515, "nome": "Monitor Multiparâmetro"},
    {"pdm": 11696, "classe": 6515, "nome": "Ventilador Artificial Eletrônico"},
    {"pdm": 7145,  "classe": 6515, "nome": "Aparelho Ultrassonografia"},
    {"pdm": 30160, "classe": 6525, "nome": "Equipamento De Ressonância Magnética"},
    {"pdm": 30198, "classe": 6530, "nome": "Equipamento Cardioversor Externo"},
    {"pdm": 30167, "classe": 6530, "nome": "Desfibrilador Externo"},
    {"pdm": 10879, "classe": 6525, "nome": "Peça / Componente Ressonância Magnética"}
]

BASE_URL_CATALOGO = "https://dadosabertos.compras.gov.br/modulo-material/4_consultarItemMaterial"
BASE_URL_PRECOS = "https://dadosabertos.compras.gov.br/modulo-pesquisa-preco/1_consultarMaterial"

todos_resultados_compras = []

print("Iniciando o pipeline de extração de dados de compras governamentais...\n")

# ==============================================================================
# 2) LOOP PRINCIPAL: VARREDURA POR EQUIPAMENTO
# ==============================================================================
for equip in equipamentos:
    print(f"--- [ INICIANDO EQUIPAMENTO: {equip['nome']} | PDM: {equip['pdm']} ] ---")
    
    # --------------------------------------------------------------------------
    # 2.1) BUSCAR TODOS OS CÓDIGOS ESPECÍFICOS NO CATÁLOGO PARA ESTE PDM
    # --------------------------------------------------------------------------
    params_catalogo = {
        "codigoClasse": equip["classe"],
        "codigoPdm": equip["pdm"],
        "pagina": 1
    }
    
    codigos_dos_itens = set()
    paginas_restantes_cat = 1
    
    print("  -> Passo A: Mapeando códigos específicos no catálogo...")
    while paginas_restantes_cat > 0:
        try:
            resp_cat = requests.get(BASE_URL_CATALOGO, params=params_catalogo)
            resp_cat.raise_for_status()
            dados_cat = resp_cat.json()
            
            for item in dados_cat.get("resultado", []):
                codigo = item.get("codigoItem") or item.get("codigo") or item.get("codigoMaterial")
                if codigo:
                    codigos_dos_itens.add(codigo)
            
            paginas_restantes_cat = dados_cat.get("paginasRestantes", 0)
            if paginas_restantes_cat > 0:
                params_catalogo["pagina"] += 1
                time.sleep(1) # Pausa de segurança
        except Exception as e:
            print(f"  [!] Erro no catálogo (PDM {equip['pdm']}): {e}")
            break

    lista_codigos = list(codigos_dos_itens)
    print(f"  -> {len(lista_codigos)} códigos específicos encontrados.")

    # --------------------------------------------------------------------------
    # 2.2) BUSCAR TODAS AS COMPRAS (PREÇOS) PARA CADA CÓDIGO ENCONTRADO
    # --------------------------------------------------------------------------
    print("  -> Passo B: Extraindo histórico de compras (Isso pode demorar)...")
    for codigo_item in lista_codigos:
        params_preco = {
            "codigoItemCatalogo": codigo_item, 
            "pagina": 1,
            "tamanhoPagina": 100 
        }
        
        paginas_restantes_preco = 1
        
        while paginas_restantes_preco > 0:
            try:
                resp_preco = requests.get(BASE_URL_PRECOS, params=params_preco)
                resp_preco.raise_for_status()
                dados_preco = resp_preco.json()
                
                resultados = dados_preco.get("resultado", [])
                if resultados:
                    # Injeta a Categoria e o PDM no resultado para facilitar a filtragem depois
                    for res in resultados:
                        res["Categoria_Equipamento"] = equip["nome"]
                        res["PDM_Origem"] = equip["pdm"]
                    
                    todos_resultados_compras.extend(resultados)
                
                paginas_restantes_preco = dados_preco.get("paginasRestantes", 0)
                if len(resultados) < params_preco["tamanhoPagina"]:
                    paginas_restantes_preco = 0 
                
                if paginas_restantes_preco > 0:
                    params_preco["pagina"] += 1
                    time.sleep(0.5)
                    
            except Exception as e:
                break # Segue para o próximo código se houver erro
                
        time.sleep(0.5) # Respiro entre os itens do catálogo
    
    print("  -> Extração concluída para este equipamento.\n")

# ==============================================================================
# 3) CONSOLIDAÇÃO, LIMPEZA DE DADOS E FORMATAÇÃO
# ==============================================================================
print("--- [ INICIANDO PROCESSAMENTO DOS DADOS ] ---")
if todos_resultados_compras:
    df = pd.DataFrame(todos_resultados_compras)
    
    # --------------------------------------------------------------------------
    # 3.1) CRIAR COLUNA DE VALOR TOTAL
    # --------------------------------------------------------------------------
    if 'quantidade' in df.columns and 'precoUnitario' in df.columns:
        df['quantidade'] = pd.to_numeric(df['quantidade'], errors='coerce').fillna(0)
        df['precoUnitario'] = pd.to_numeric(df['precoUnitario'], errors='coerce').fillna(0)
        df['valorTotalCompra'] = df['quantidade'] * df['precoUnitario']
        print("  -> Coluna 'valorTotalCompra' calculada com sucesso.")
    
    # --------------------------------------------------------------------------
    # 3.2) REORDENAR COLUNAS (Categoria e Descrição primeiro)
    # --------------------------------------------------------------------------
    cols = df.columns.tolist()
    
    # Puxando as colunas mais importantes para o início da planilha
    colunas_prioritarias = ['Categoria_Equipamento', 'descricaoItem', 'PDM_Origem']
    for col in reversed(colunas_prioritarias):
        if col in cols:
            cols.insert(0, cols.pop(cols.index(col)))
            
    df = df[cols]
        
    # --------------------------------------------------------------------------
    # 3.3) FORMATAR DATAS PARA O PADRÃO BRASILEIRO (DD/MM/YYYY)
    # --------------------------------------------------------------------------
    colunas_de_data = [col for col in df.columns if 'data' in col.lower()]
    for col in colunas_de_data:
        try:
            df[col] = pd.to_datetime(df[col]).dt.strftime('%d/%m/%Y')
        except:
            pass 
    print("  -> Datas formatadas para o padrão brasileiro.")

    # ==============================================================================
    # 4) EXPORTAÇÃO PARA EXCEL (AMBIENTE LOCAL - VS CODE)
    # ==============================================================================
    nome_arquivo = "Base_Consolidada_Equipamentos_Medicos.xlsx"
    caminho_absoluto = os.path.abspath(nome_arquivo)
    
    df.to_excel(nome_arquivo, index=False)
    
    print("\n======================================================================")
    print(f"EXTRAÇÃO GERAL CONCLUÍDA! Total de compras encontradas: {len(df)}")
    print(f"Arquivo gerado com sucesso em: {caminho_absoluto}")
    print("======================================================================")

else:
    print("\nNenhuma compra foi encontrada para a lista de equipamentos fornecida.")