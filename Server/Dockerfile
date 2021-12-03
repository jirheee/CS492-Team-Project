FROM node as base

WORKDIR /home/node/app

COPY package*.json ./

RUN apt-get update || : && apt-get install python3-pip -y

RUN apt-get install tzdata && cp /usr/share/zoneinfo/ROK /etc/localtime && echo "ROK" > /etc/timezone

RUN pip3 install torch==1.10.0

RUN pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu102.html

RUN pip3 install torch-sparse torch-cluster -f https://pytorch-geometric.com/whl/torch-1.10.0+cu102.html

RUN pip3 install torch-geometric tensorboard=0.6.0

RUN yarn

COPY . .

RUN ["chmod", "+x", "./entrypoint.sh"]

ADD https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh /
