from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel
import socketio



class Item(BaseModel):
    name: str
    price: float
    is_offer: Optional[bool] = None

sio = socketio.AsyncServer(cors_allowed_origins='*', async_mode='asgi')
app = FastAPI()
socketio_app = socketio.ASGIApp(sio, app)

@app.get("/ping")
def read_root():
    return {"message": "pong"}

@sio.event
def connect(sid, environ):
    print("connect ", sid)


@sio.on('message')
async def chat_message(sid, data):
    print("message ", data)
    await sio.emit('response', 'hi ' + data)

@sio.event
def disconnect(sid):
    print('disconnect ', sid)