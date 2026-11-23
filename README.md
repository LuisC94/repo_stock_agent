# Stock Agent - Retail Eureka 📦

O Stock Agent é um agente de aprendizagem por reforço (PPO) que decide **o que colocar à venda e quando**. O objetivo é **maximizar o lucro e minimizar o desperdício**.

Há dois agentes, que partilham o mesmo núcleo:

| Agente | Decisão | Frequência |
|---|---|---|
| **Agente de Lotes** | Coloca à venda 0, 1 ou vários lotes (`Harvest`), escolhidos entre todos os armazéns do perfil | Diária |
| **Agente de Armazéns** | Abre no máximo 1 armazém; todo o stock desse armazém vai para venda | Semanal |

Treina-se **um modelo por perfil (produtor) × cultura × tipo de agente**.

---

## 📂 Estrutura

```text
Stock_Agent/
├── app.py                     # Interface Streamlit (Dados / Treinar / Simulação & Inferência)
├── service/main.py            # Web Service FastAPI (porta 8005) para a plataforma Django
├── stock_agent/
│   ├── schema.py              # Formato do Excel (5 folhas) + validação
│   ├── synthetic.py           # Gerador de dados sintéticos
│   ├── simulator.py           # Simulador de temporada (mercado, envelhecimento, desperdício, recompensa)
│   ├── lc_kernel.py           # Motor Lifecycle LC v0.2.0 em numba (cópia do buyer_agent_service)
│   ├── lifecycle/lc_v0_2_0.py # Presets biológicos (cópia do Lifecycle Web Service)
│   ├── policy.py              # Rede Actor-Critic para conjuntos (Deep Sets)
│   ├── heuristics.py          # Políticas de referência: FEFO + Procura, Venda Imediata
│   ├── training.py            # Imitação FEFO + PPO, validação, melhor modelo
│   ├── inference.py           # Simulação de temporada, explicações, decisão a partir do estado atual
│   └── model_io.py            # Modelo em ZIP (config.json + policy.pt + historico_treino.csv)
├── data/dados_sinteticos_stock_agent.xlsx   # Excel de exemplo
└── models_storage/            # Modelos publicados: <perfil>/<cultura>/<lot|warehouse>/modelo.zip
```

## 🚀 Executar

```bash
pip install -r requirements.txt
streamlit run app.py                 # interface      -> http://localhost:8501
python service/main.py               # web service    -> http://localhost:8005/docs
python -m stock_agent.synthetic      # (re)gerar data/dados_sinteticos_stock_agent.xlsx
```

Com Docker:

```bash
docker compose up --build            # UI na porta 8501 + Web Service na porta 8005
```

A interface e o serviço partilham a pasta `models_storage/`. O botão **"Publicar no repositório do Web Service"** coloca o modelo treinado na interface à disposição do serviço.

## ☁️ Link público (Streamlit Community Cloud)

Repositório: [github.com/LuisC94/repo_stock_agent](https://github.com/LuisC94/repo_stock_agent).

1. Entrar em [share.streamlit.io](https://share.streamlit.io) com a conta GitHub. Se for a primeira vez, autorizar o acesso ao GitHub.
2. **Create app** → **Deploy a public app from GitHub**:
   - Repository: este repositório
   - Branch: `main`
   - Main file path: `app.py`
   - App URL: o subdomínio pretendido, por exemplo `stock-agent-eureka`
3. Em **Advanced settings**, escolher **Python 3.12**.
4. **Deploy**. A primeira instalação demora alguns minutos. O `requirements.txt` instala o PyTorch só para CPU.
5. Para o link ficar acessível sem login, confirmar em **Settings → Sharing** que a app está como **Public**.

Cada `git push` para `main` atualiza a app automaticamente.

**Notas para a versão na cloud:**
- **Recursos limitados:** os treinos de 400 a 600 episódios demoram mais do que localmente. Para demonstrações, usar menos episódios ou carregar um modelo ZIP já treinado.
- **Armazenamento temporário:** os modelos publicados em `models_storage/` perdem-se quando a app reinicia. Descarregar o ZIP de cada modelo.
- **Inatividade:** a app "adormece" após alguns dias sem uso e acorda no primeiro acesso.

---

## 📊 Excel de entrada

Cada folha corresponde a uma entidade que a plataforma já guarda:

| Folha | Entidade Django | Colunas |
|---|---|---|
| `Armazens` | `Warehouse` | `armazem_id`, `perfil`, `localizacao`, `regiao`, `tipo_controlo` (`Controlled` / `Non-Controlled`), `capacidade_kg` |
| `Lotes` | `Harvest` | `lote_id`, `perfil`, `armazem_id`, `cultura`, `tipo_fruta` (chave Lifecycle, ex.: `kiwi_hayward`, `pear`), `temporada`, `data_colheita`, `data_entrada`, `quantidade_kg`, `firmeza`, `brix`, `acidez`, `calibre_mm`, `score_qualidade` |
| `Leituras` | `WarehouseSensorReading` | `armazem_id`, `data`, `temperatura`, `humidade`, `etileno` |
| `Mercado` | `HistoricalSalesData` + `DemandForecast` | `perfil`, `cultura`, `data`, `preco_kg`, `vendas_kg` (procura real), `procura_prevista_kg` |

A folha `LEIA-ME` do Excel gerado descreve todas as colunas e indica quais são opcionais.

O **gerador sintético** cria os dados desta forma:

- Armazéns com e sem atmosfera controlada (AC). A fruta de guarda longa vai preferencialmente para AC.
- Várias temporadas por cultura. As culturas podem estar misturadas no mesmo armazém.
- Leituras de sensores diárias, com excursões ocasionais de temperatura.
- Mercado sazonal: o preço é baixo na colheita e sobe com a escassez, com ruído e picos. A procura de cada temporada é calibrada para o stock dessa temporada.

---

## ⚙️ Como funciona a simulação

Em cada dia da temporada acontece o seguinte:

1. Os lotes entram nos armazéns na `data_entrada`.
2. O agente decide o que colocar à venda.
3. O mercado absorve até `vendas_kg` do stock à venda, começando pelo lote mais urgente. O preço recebido é `preço do dia × fator de qualidade`.
4. Todos os lotes envelhecem 1 dia com o **kernel Lifecycle LC v0.2.0**, usando as leituras do armazém:
   - **AC:** fator de respiração 0.10 (o mesmo que o LC usa para MAP) e etileno removido. Um kiwi dura cerca de 270 dias.
   - **Frio normal:** um kiwi dura cerca de 100 dias.
   - **À venda:** o lote sai da AC e fica em condições de cais de expedição.
5. **Desperdício:** o lote estraga quando a qualidade cai abaixo de 30 **depois de ter atingido o pico**. Isto evita classificar como estragado o kiwi colhido verde, que amadurece em armazém.

**Recompensa (€):**

```
receita − peso_desperdício × kg_desperdiçados × preço_médio − custo_armazenagem
```

No fim do horizonte, todo o stock por vender conta como desperdício. Todos os parâmetros são configuráveis na interface.

## 🧠 Modelo e treino

- **Rede Deep Sets:** cada lote ou armazém é avaliado pela mesma rede, juntamente com o contexto de mercado (preço face às médias de 7 e 30 dias, procura prevista a 7 e 14 dias, stock já à venda, cobertura do stock, época do ano…). Um resumo do conjunto permite que cada decisão tenha em conta os outros candidatos. A rede funciona com **qualquer número de lotes ou armazéns** e não depende da ordem em que aparecem.
  - Lotes: uma probabilidade Bernoulli por lote, o que permite vender vários no mesmo dia.
  - Armazéns: uma distribuição categórica sobre `[ESPERAR, armazém_1..k]`.
- **Fase 1, imitação da heurística FEFO + Procura:** dá ao agente um ponto de partida sensato.
- **Fase 2, PPO:** o treino usa aumento de dados (preço, procura e conjunto de lotes perturbados) e valida periodicamente nas temporadas de teste. Guarda-se o melhor modelo.
- **Comparação:** se o modelo final não superar a FEFO na validação, a interface mostra um aviso.

Resultados nos dados de exemplo (temporada 2024, não vista no treino):

| Agente | Lucro (agente vs FEFO) | Desperdício (agente vs FEFO) | Tempo de treino |
|---|---|---|---|
| Pera, Lotes | 178,9 k€ vs 170,8 k€ | 12,2% vs 14,2% | ≈ 2,5 min |
| Kiwi, Armazéns | 281,9 k€ vs 276,6 k€ | 4,2% vs 4,6% | ≈ 1 min |

Em ambos os casos, a venda imediata desperdiça entre 43 e 55%.

---

## 🌐 Web Service (porta 8005)

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/health` | Estado do serviço |
| GET | `/models` | Modelos disponíveis e respetivas métricas |
| POST | `/models/upload` | Publica um modelo ZIP descarregado da interface |
| POST | `/train` | Treino em background a partir do Excel (multipart); devolve `job_id` |
| GET | `/train/{job_id}` | Progresso, logs e métricas do treino |
| POST | `/decide` | Decisão para a data indicada |

O pedido `POST /decide` reflete o estado atual da plataforma:

```json
{
  "perfil": "Produtor_Demo", "cultura": "Pera Rocha", "modo": "lot", "data": "2024-11-20",
  "armazens": [{"armazem_id": "A01", "tipo_controlo": "Controlled"}],
  "lotes": [{"lote_id": "L00231", "armazem_id": "A01", "data_colheita": "2024-09-02", "data_entrada": "2024-09-03",
             "quantidade_kg": 12400, "firmeza": 55.1, "brix": 12.3, "acidez": 0.31, "estado": "armazenado"}],
  "leituras": [{"armazem_id": "A01", "data": "2024-09-03", "temperatura": 0.8, "humidade": 94.1, "etileno": 0.02}],
  "mercado": [{"data": "2024-10-20", "preco_kg": 1.05, "procura_prevista_kg": 1350}]
}
```

O serviço reconstrói o estado biológico de cada lote com o kernel LC, a partir das leituras desde a entrada. A resposta traz:

- **Modo `lot`:** `lotes_a_colocar_a_venda`, com a probabilidade, o RSL, a qualidade e a explicação de cada lote.
- **Modo `warehouse`:** `armazem_a_abrir` (ou `null` para esperar), mais a avaliação de cada armazém.

A tab de inferência tem um expander que gera este pedido a partir do Excel e mostra a resposta.

---

## 📝 Notas e próximos passos

- O `lc_kernel.py` e o `lifecycle/lc_v0_2_0.py` são cópias dos ficheiros do `buyer_agent_service`. Se o Lifecycle mudar, é preciso copiá-los de novo.
- **Preço:** o agente decide *o quê* e *quando*; o preço é o do mercado ajustado à qualidade. Propor também o €/kg é uma extensão possível (uma ação contínua adicional).
- **Agente de Armazéns com culturas misturadas:** o modelo é por cultura, por isso só considera os lotes dessa cultura no armazém. Na realidade, abrir o armazém expõe também as outras culturas.
- **Dados reais:** para treinar com dados reais, basta exportar da plataforma as 4 entidades no formato acima. Os treinos de 400 a 600 episódios demoram 1 a 3 minutos em CPU.
