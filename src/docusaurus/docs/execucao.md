---
title : "Execução"
sidebar_position : 3
---

# Execução

A plataforma GEIST é projetada para ser intuitiva e poderosa, integrando tecnologias avançadas como redes neurais e Deep Learning para fornecer previsões futuras precisas. A execução da plataforma começa com a coleta e processamento de grandes volumes de dados financeiros, operacionais e macroeconômicos. As redes neurais são utilizadas para identificar padrões complexos e não-lineares nos dados históricos, enquanto algoritmos de Deep Learning aprimoram continuamente as previsões com base em novas informações. Essas tecnologias se integram perfeitamente para gerar modelos preditivos robustos que simulam cenários futuros variados, ajudando CFOs, CEOs e Diretores de Estratégia a tomar decisões informadas e estratégicas com base em insights detalhados e acionáveis.

# Stack

A plataforma GEIST é construída utilizando uma stack tecnológica robusta e moderna para garantir eficiência, acessibilidade e poder de processamento. Para o front-end e a interface do usuário, GEIST utiliza o Streamlit, uma poderosa plataforma web que facilita a criação de aplicativos de dados interativos e visualizações intuitivas, permitindo uma experiência de usuário fluida e amigável. No backend, a aplicação é alimentada por sofisticados modelos de Inteligência Artificial, incluindo Redes Neurais (NNs) e técnicas de Deep Learning, que são responsáveis por processar grandes volumes de dados e gerar previsões precisas.

# Modelos

## Por que NNs e Deep Learning?

Optamos por utilizar Redes Neurais (NNs) e Deep Learning em nossa plataforma devido à sua capacidade superior de identificar e entender padrões complexos em grandes volumes de dados, independentemente do setor ou do tipo de empresa. As NNs são especialmente eficazes em aprender e generalizar a partir de dados diversos, permitindo que a nossa solução seja altamente adaptável e aplicável a diferentes empresas com necessidades únicas. Além disso, técnicas de Deep Learning ampliam essa capacidade ao lidar com camadas mais profundas de aprendizado, capturando relações não-lineares e sutis nos dados, o que resulta em previsões mais precisas e insights detalhados.

# Como executar?

1 . Acesse o diretório "front".

```
cd .\front\
```

2 . Crie um ambiente virtual em seu terminal.

```
python -m venv venv
```

3 . Ative o ambiente virtual.

```
.\venv\Scripts\activate
```

4 . Instale as depedências necessárias, após ativar o ambiente virtual.

```
pip install -r requirements.txt
```

5 . Com as depedências, inicie a aplicação:

```
streamlit run .\app.py
```

6 . Acesse o endereço: ```http://localhost:8501``` em seu navegador.