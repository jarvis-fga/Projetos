## Projetos

Repositório para as atividades da disciplina Aprendizado de Máquina ministrada na Faculdade do Gama, Universidade de Brasília, para os alunos de graduação em Engenharia de Software.

Cada pasta do repositório é referente a um dos problemas propostos pelos profressores, as quais serão criadas e evoluídas ao longo do segundo semestre de 2017. A descrição do contexto de cada problema e das discussões sobre se encontram na Wiki do repositório.

A Wiki dispõe de um pequeno resumo sobre as tecnologias utilizadas na resolução dos problemas, bem como links úteis para quem se interessar pelo conteúdo.

## Ambiente de Desenvolvimento
Construindo a imagem:
```
docker build -t machine-learn .
```

Executando o container:
```
docker run --name machine -p 8888:8888 -v path/Projetos:/code machine-learn:latest /bin/sh ./boot.sh
```

Atenção, a expressão `path` deve ser substituída pelo caminho da sua pasta `Projetos`

Acessando o bash do container:
```
docker exec -it machine-learn bash
```

Ao executar o container, o servidor do `jupiter-notebook` é iniciado. Acesse-o na porta `8888` do seu `localhost`.
