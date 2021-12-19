# 🤖 Interactive RL Playground

<div align="center">

![TypeScipt](https://img.shields.io/badge/typescript-4.1.0-719af4?logo=typescript)
![React](https://img.shields.io/badge/react-17.0.2-9cf?logo=react)
![Chakra](https://img.shields.io/badge/chakra-1.6.12-%234ED1C5?logo=chakraui)

![Express](https://img.shields.io/badge/express-v4.17.13-010101?logo=express)
![MySQL](https://img.shields.io/badge/mysql-2.18.1-%2300f?logo=mysql)
![Socket.io](https://img.shields.io/badge/Socket.io-4.3.1-black?logo=socket.io&badgeColor=010101)
![Docker](https://img.shields.io/badge/docker-20.10.10-%230db7ed?logo=docker)

![OpenIssues](https://img.shields.io/github/issues-raw/jirheee/CS492-Team-Project)
![ClosedIssues](https://img.shields.io/github/issues-closed-raw/jirheee/CS492-Team-Project)
![License](https://img.shields.io/github/license/jirheee/CS492-Team-Project)
</div>

<br>

## ✏️ What is this?
Artificial Intelligence(AI) is a term that is being used more and more everyday. However, for most people it is hard to get familiar with AI since it contains relatively new technologies and jargon such as artificial neural networks or anything related to machine learning, which are difficult to understand without deep prior knowledge of mathematics and computer science. To bridge the gap between people and neural networks, google tensorflow provides Neural Network Playground, where people can easily construct neural networks to classify linearly inseparable data. Inspired by this playground, we create an **Interactive Reinforcement Learning(RL) playground**, where even users without any background in AI can train and evaluate their own RL agent for well known game `Gomoku`.

Main features are:
1. Build own RL agent that plays `Gomoku`
2. Train the agent to learn how to play Gomoku
3. Monitor the training process of the agent
4. Battle with Gomoku agents that are created by others

<br>

## 📋 How to use Interactive RL Playground

If you want to learn how to use RL playground - [How to use](https://github.com/jirheee/CS492-Team-Project/blob/main/How_To_Use.md)

<br>

## 🖥 How to launch the project
**Dependencies**
- [**NodeJS**](https://nodejs.org/en/) 16.13.1 LTS
- **npm** 8.1.0
- **yarn** 1.22.17
- [**Docker**](https://www.docker.com/products/docker-desktop) 20.10.10

<br>

1️⃣ Clone Repository
```
$ git clone https://github.com/jirheee/CS492-Team-Project.git
```
2️⃣ Install Packages
```
$ cd client && yarn
$ cd ../server && yarn
$ cp .env.example .env
$ cd ..
```
3️⃣ Run Server

Below command will run the dockerized db server and main api server in your localhost.
```
$ cd Server
$ docker-compose up
```
4️⃣ Run Client

Below command will host the frontend at http://localhost:3000
```
$ cd Client
$ yarn start
```

5️⃣ Go to http://localhost:3000, you will be able to build, train, test your own agent!



--------
**2021 KAIST Fall Semester** &middot; CS492(I) - Intro to DL
