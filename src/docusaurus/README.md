# Website

Este website é construído usando o [Docusaurus](https://docusaurus.io/), um gerador moderno de sites estáticos.

### Instalação

```
$ yarn
```

### Desenvolvimento Local

```
$ yarn start
```

Este comando inicia um servidor de desenvolvimento local e abre uma janela do navegador. A maioria das alterações é refletida ao vivo, sem a necessidade de reiniciar o servidor.

### Build

```
$ yarn build
```

Este comando gera conteúdo estático no diretório build e pode ser servido usando qualquer serviço de hospedagem de conteúdo estático.

### Deployment

Usando SSH:

```
$ USE_SSH=true yarn deploy
```

Sem SSH:

```
$ GIT_USER=<Your GitHub username> yarn deploy
```

Se você estiver usando o GitHub Pages para hospedagem, este comando é uma maneira conveniente de construir o website e enviar para o branch gh-pages.
