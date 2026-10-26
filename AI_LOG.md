# AI_LOG.md — Registo do uso de IA

**Projeto:** P2 · Tema F1 · Assistente de doenças da vinha (DL_P2_TP2-G2)
**Membro:** Francisco Rafael Pires Veiga al82066 — F1 (Dados, protocolo e integração)
**Ferramenta de IA:** Claude Sonnet 5.5 (Anthropic), interface de chat
**Data da sessão:** 23/10/2026
**Tarefa:** notebook de preparação de dados do FieldVitis (`preparacao_dados_fieldvitis.ipynb`)

> Regra do protocolo (secção 8): a IA ajuda a programar, depurar e escrever, mas **a verificação é humana**. Os campos ☐ são para eu preencher depois de correr o notebook nos dados reais.

---

## 1. O que foi pedido

Um notebook que faça **apenas o tratamento de dados** do FieldVitis: verificar se batem certo com o protocolo, eliminar duplicados e gerar o `splits.json`. Coerente com o `baseline_v0_cnn_fieldvitis.ipynb` já existente (criado anteriormente pelo aluno).

## 2. O que a IA fez

1. Leu o baseline para manter coerência (caminhos, formato do `splits.json`, hash SHA-256, regra de ≥ 6 imagens de black rot no teste).
2. Escreveu o notebook (sem treino nem modelos) com: inventário de ficheiros, contagens por classe vs. tabela 2.1, duplicados exatos (MD5), limpeza com motivo registado, inspeção visual, `splits.json` estratificado (3 seeds) com hash e verificação anti-fuga.
3. Testou-o num dataset sintético com problemas plantados (ficheiro corrompido, `.DS_Store`, imagem 32×32, duplicado na mesma classe, duplicado entre classes).
4. Numa primeira versão, o notebook incluía também deteção de quase-duplicados (dHash), um teste de *shortcut* com metadados e divisão por grupos. **Foi retirado** por ser mais complexo do que o necessário; fica registado como limitação (ver secção 4).

## 3. Registo de propostas da IA e verificação

| # | Proposta da IA (código ou afirmação) | Funcionou / estava correta? (S/N) | O que estava errado ou em falta | Como verificou (output, shape, métrica) |
|---|---|---|---|---|
| 1 | Notebook de preparação de dados (inventário, contagens, MD5, limpeza, splits, verificação) | **S** em dados sintéticos; ☐ por confirmar nos dados reais | Testado só com imagens sintéticas, não com o FieldVitis real | Execução completa com `nbconvert` sem erros; os 5 casos plantados foram apanhados e listados em `removidas.csv`; ☐ _correr nos dados reais e anotar contagens_ |
| 2 | Política: duplicado exato entre classes → removem-se todas as cópias; na mesma classe → fica a 1.ª | S (decisão de desenho) | É uma **escolha**, não um facto. Pode reduzir o black rot (39 imagens) | Contagens antes/depois por classe e aviso automático se a classe rara ficar < 25; ☐ _decidir e justificar no D1_ |
| 3 | `splits.json` estratificado por classe, 3 seeds, ≥ 6 black rot no teste, SHA-256 | S (sintético) | Divide por imagem; não protege contra quase-duplicados | Asserts: sem sobreposição treino/val/teste, sem MD5 repetido entre splits, hash recalculado a partir do ficheiro, ≥ 6 da classe rara no teste nas 3 seeds |
| 4 | Formato do `splits.json` compatível com o baseline (`classes`, `seeds`, `sha256`; caminhos relativos a `DATA_DIR`) | S (por leitura do código do baseline) | A compatibilidade **não foi executada** com o baseline | ☐ _correr o baseline com o novo `splits.json` e confirmar_ |
| 5 | Afirmação: "o baseline assume 201 imagens em esca, e não 200 como o protocolo" | S | — | Lido no `assert ... [39, 158, 158, 201]` do baseline; ☐ _confirmar a contagem real e corrigir a tabela 2.1_ |

## 4. Limitações

- O notebook **nunca foi corrido nas imagens reais** do FieldVitis pela IA; os testes são de dados sintéticos.
- **Quase-duplicados não são procurados** (só duplicados exatos por MD5). Foram observadas visualmente fotos que parecem ser da mesma folha em Leaf_blight; não foram medidas nem separadas no split (decisão: aceitar e declarar, dado que o dataset não tem identificador de planta). O Macro-F1 em Leaf_blight pode estar inflacionado.
- **Risco de shortcut por fundo e resolução:** a maioria das fotos de Leaf_blight tem uma mão ao fundo e as resoluções diferem entre classes. A CNN pode aprender estas pistas em vez da doença. A verificar com Grad-CAM e a mitigar com augmentation (crop, resize, blur).
- As decisões de limpeza (tratamento de pares entre classes, remoção na mesma classe) são propostas e devem ser decididas após inspeção visual.
- Nenhum conjunto de teste foi tocado: o notebook só prepara dados e divisões.

## 5. Decisões a preencher após correr nos dados reais

☑ Contagens reais encontradas (tabela 2.1 do D1): Black_rot	39; Esca 201; Healthy 158; Leaf_blight 158; TOTAL 556; (Há mais uma na esca)
☑ Contagens após limpeza e nº de imagens de black rot restantes: Saudável 158, Esca 201, Leaf blight 158, Black rot 39. Black rot: 39 imagens, 6 no teste (seed 0).
☑ Política para pares entre classes: Não aplicável. Não foram encontrados duplicados exatos (MD5), nem entre classes nem na mesma e não houve ficheiros corrompidos nem imagens < 64 px (removidas: nenhuma). Quase-duplicados não foram procurados.
☑ Observações visuais: O fundo varia (mãos, céu, terreno); a maioria das fotos de Leaf_blight tem uma mão ao fundo, o que cria risco de shortcut (a CNN pode associar a mão à classe). A iluminação parece constante entre classes. As resoluções diferem entre algumas classes, o que pode refletir origens diferentes. Em Leaf_blight, algumas fotos parecem ser da mesma folha. O split é feito por imagem e não as separa, por isso podem existir fotos da mesma folha em treino e teste. Estas observações são visuais, não quantificadas.
☑ Política para fotos da mesma folha (Leaf_blight): aceitar e declarar como limitação, sem deteção nem agrupamento no split. Decidido em 26/10.
☐ Hash do `splits.json` registado no PROTOCOLO.md: _______________________

## 6. Ficheiros produzidos com apoio da IA

- `preparacao_dados_fieldvitis.ipynb`
- Saídas ao correr o notebook: `data/splits.json`, `data/manifest_clean.csv`, `data/qualidade/{manifest_raw.csv, removidas.csv}`
- Este ficheiro (`AI_LOG.md`)