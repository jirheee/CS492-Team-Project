# CS492-Team-Project Client

If you are not familiar with Docker, [this video](https://www.youtube.com/watch?v=3c-iBn73dDE&t=7081s) was quite helpful for me understanding the files in this repository.  

## How to launch the server (and the database) on Docker  
0. If using VSCode, install Docker extension.  
    - opening either `Dockerfile` or `docker-compose.yml` should prompt you to install a **[Docker extension](https://marketplace.visualstudio.com/items?itemName=ms-azuretools.vscode-docker)**.  
1. Next, you should install a Docker CLI or **[Docker Desktop](https://www.docker.com/products/docker-desktop)**.  
    - On VSCode, if you see something like "*cannot connect to Docker*" inside the Docker tab on the left, you can install it by clinking on it.  
    This will require you to restart the machine once.  
2. After restart, create `.env` . (You can do it before too)
    - Copy the `.env.example` file to `.env` so we can pass the environmental variables to the container.  
    - For Windows users, I recommend using **PowerShell** instead of regular DOS CMD.  
        ```
        cp .env.example .env
        ```
3. Finally, we can run the server on Docker.  
***IMPORTANT*** [Check the End-Of-Line (EOL) format of your `entrypoint.sh` first](https://github.com/jirheee/CS492-Team-Project/issues/8#issuecomment-960514476).  It should be in "**LF**" format.  
**^^This should be checked before running the command.**  
    - To run the server, open your PowerShell (Windows) or shell (others) and run  
        ```
        docker-compose up --build
        ```
        in the `Server/` directory.
    - If something goes wrong in the middle, you should run
        ```
        docker-compose down -v --rmi local
        ```
        to cleanup the mess. You can do this after `Ctrl+C` or in a separate instance of terminal.  
    
If you see 
```
cs492i-api-server  | Server running on port 5000
```
on the last line, it has been executed successfully.  
If you see 
```
cs492i-api-server exited with code 1
cs492i-api-server exited with code 1
cs492i-api-server exited with code 1
...
```
at the end, or 
```
cs492i-api-server  | standard_init_linux.go:228: exec user process caused: no such file or directory cs492i-api-server exited with code 0
```
somewhere in the middle, your `entrypoint.sh` is likely the culprit.

## After understanding more about Docker  

### Consistent Database (Don't delete the volume!)  
To retain database after closing and cleaning up the server, run
```
docker-compose down
```
without `-v` and `--rmi local`.  
You may also want to run the server with
```
docker-compose up
```
without the `--build`, since `--build` option creates a new image everytime.  
  
## To interact with the server container
If you want to use the alpine linux of the server,  
Start up the server and the database with
```
docker-compose up -d
```
This will run the Docker network (cluster of containers) in the background after initiating it.  
Then run
```
docker-compose exec cs492i-api-server sh
```
to interact with the server container.  
Or at the same time,
```
docker-compose up -d
docker-compose exec cs492i-api-server sh
.
```
You can check if you are in the container and the volum by executing `ls`. You should see
```
# ls
Dockerfile  docker-compose.yml  node_modules  package.json  tsconfig.json
README.md   entrypoint.sh       nodemon.json  src           yarn.lock
```