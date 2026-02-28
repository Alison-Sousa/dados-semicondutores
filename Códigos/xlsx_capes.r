# 1) Instalação de pacotes necessários
pacotes <- c("capesR", "openxlsx")
faltando <- pacotes[!sapply(pacotes, requireNamespace, quietly = TRUE)]
if (length(faltando) > 0) install.packages(faltando)

suppressPackageStartupMessages({
  library(capesR)
  library(openxlsx)
})

# 2) Caminho de saída do xlsx
saida_xlsx <- "C:/Users/PC GAMER/Downloads/dados-semicondutores/Base/teses_dissertacoes_semicondutores.xlsx"
dir.create(dirname(saida_xlsx), recursive = TRUE, showWarnings = FALSE)

# 3) Download e leitura dos dados CAPES (2013–2026)
files <- download_capes_data(2013:2026)
dados <- read_capes_data(files)

# 4) Palavras-chave de filtragem
keywords <- c(
  "semicondutor", "semicondutores", "microeletronica", "microeletrônica",
  "microeletronico", "microeletrônico", "microchip", "chip", "chips",
  "circuito integrado", "circuitos integrados", "silicio", "silício",
  "transistor", "transistores", "wafer", "wafers", "fab", "foundry",
  "fotolitografia", "litografia", "litografia ultravioleta", "euv",
  "processo cmos", "cmos", "vlsi", "microfabricacao", "microfabricação",
  "nanofabricacao", "nanofabricação", "encapsulamento eletronico",
  "encapsulamento eletrônico", "design de chip", "projeto de chip",
  "projeto de circuitos", "ic design", "integrated circuit", "semiconductor"
)
padrao <- paste(keywords, collapse = "|")

# 5) Localização das colunas de título e resumo
col_titulo <- intersect(c("titulo", "title", "nm_producao", "NM_PRODUCAO"), names(dados))
col_resumo <- intersect(c("resumo", "abstract", "ds_resumo", "DS_RESUMO"), names(dados))

texto_titulo <- if (length(col_titulo) > 0) as.character(dados[[col_titulo[1]]]) else ""
texto_resumo <- if (length(col_resumo) > 0) as.character(dados[[col_resumo[1]]]) else ""

# 6) Filtragem por palavras-chave e exportação
texto_busca <- paste(texto_titulo, texto_resumo)
idx <- grepl(padrao, texto_busca, ignore.case = TRUE)
res <- dados[idx, , drop = FALSE]

write.xlsx(res, saida_xlsx, overwrite = TRUE)

message("Arquivo salvo em: ", saida_xlsx)
message("Total de registros filtrados: ", nrow(res))
