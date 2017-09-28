## Projetos

Repositório para as atividades da disciplina Aprendizado de Máquina ministrada na Faculdade do Gama, Universidade de Brasília, para os alunos de graduação em Engenharia de Software.

Cada pasta do repositório é referente a um dos problemas propostos pelos profressores, as quais serão criadas e evoluídas ao longo do segundo semestre de 2017. A descrição do contexto de cada problema e das discussões sobre se encontram na Wiki do repositório.

A Wiki dispõe de um pequeno resumo sobre as tecnologias utilizadas na resolução dos problemas, bem como links úteis para quem se interessar pelo conteúdo.

## Ambiente de Desenvolvimento
Abaixo, apresenta-se instruções para criação e uso do ambiente por meio da plataforma Docker.

Clonando o repositório:
```
git clone https://github.com/jarvis-fga/Projetos.git
```

Construindo a imagem:
```
docker build -t machine-learn .
```
O comando acima deve ser realizado dentro da pasta Projetos, onde se encontra o `Dockerfile`.

Executando o container:
```
docker run --name machine -p 8888:8888 -v your_path/Projetos:/code machine-learn:latest /bin/sh ./boot.sh
```

Atenção, a expressão `your_path` deve ser substituída pelo caminho da sua pasta `Projetos`

Acessando o bash do container:
```
docker exec -it machine bash
```
Parando a execução do container:
```
docker stop machine
```
Iniciando novamente:
```
docker start machine
```
Ao executar o container, o servidor do `jupiter-notebook` é iniciado. Acesse-o na porta `8888` do seu `localhost`.
Às vezes o `jupiter-notebook` exige um token de acesso, que pode ser visualizado nos logs do container. Caso tenha executado o container com o comando `start` precisará do comando `docker logs machine` para visualizá-los.

