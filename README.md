## Projetos

Repositório para as atividades da disciplina Aprendizado de Máquina

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

