# ia-tarefas

Fazer o download do [Anaconda](https://www.anaconda.com/download/#linux)

Criar um ambiente no Anaconda

```sh
$ conda create -n meuAmbiente
```

Uma vez criado o ambiente, use o comando

```sh
$ source activate meuAmbiente
```

Ao acessar o ambiente, instale os seguintes pacotes nele

```sh
$ conda install numpy jupyter notebook
```

O pacote notebook Jupyter é uma aplicação web que permite que você combine texto explicativo, equações matemáticas, código e visualizações em um único documento facilmente compartilhável.

Agora é so iniciar o Jupyter

```sh
$ jupyter notebook
```

Por padrão o jupyter roda na porta 8888 do localhost.
