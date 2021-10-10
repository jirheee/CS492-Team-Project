# CS492-Team-Project Client
## Dependencies
python >= 3.8.8 <br>
pip >= 21.0.1 <br>
fastapi >= 0.70.0 <br>
uvicorn >= 0.15.0 <br>
python-socketio >= 5.4.0 
## How to run server
Run below commands to install dependencies.
```
pip install fastapi
pip install 'uvicorn[standard]'
pip install python-socketio
```
After installing, run `uvicorn main:socketio_app --reload` in src directory to run server.(`--reload` option will let you auto-reload server after any modification happened in src.)

You can check your server by sending `GET /ping` on localhost:8000. If it is running, it will send you back "pong".

## Documentation
You can see auto-generated api docs by going to http://localhost:8000/docs.