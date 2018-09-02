# Tarefas da disciplina de IA

## Configurando projeto
Primeiramente é necessário fazer o download do [Anaconda](https://www.anaconda.com/download/#linux). O Anaconda é um gerenciador de pacotes e ambientes, nele é possível isolar pojetos com versões e pacotes diferentes do Python.

Criar um ambiente no Anaconda
```sh
$ conda create -n ambienteIA
```

Acessar o ambiente criado
```sh
$ conda activate ambienteIA
```

Ao acessar o ambiente, instale os seguintes pacotes nele
```sh
$ conda install numpy pandas
```

##Compilação e execução
Pode-se executar os arquivos de duas maneiras

### Terminal
Basta entrar no diretório de alguma tarefa e rodar algum arquivo **.py**
```sh
$ python3 nome_do_arquivo.py
```

### Usando o Notebook Jupyter
O Notebook Jupyter é uma aplicação web que permite emular os códigos do diretório, fazendo a execução dos arquivos com extensão **.ipynb**

Deve-se instalar o Jupyter em seu ambiente
```sh
$ conda install notebook jupyter
```

Agora é só iniciar o Jupyter
```sh
$ jupyter notebook
```

Por padrão o jupyter deve rodar na porta 8888 do localhost.

##Descrição das Tarefas

### Tarefa 1:
Implementação de um neurônio, sendo um o mesmo com uma carga de 4 entradas.

### Tarefa 2:
Evolução da tarefa 1, fazendo com que o peso das entradas do neurônio seja atualizado conforme a execução do código

### Tarefa 3:
Implementação de uma rede 4x3x2, sendo uma rede com duas camadas, onde a 1º camada é composta por um neurônio de com 4 entradas e a 2ª camada um neurônio com 3 entradas. A carga da 1ª camada é dada de forma aleatória e a carga da 2ª camada é a saida da 1ª camada.

### Tarefa 4:
