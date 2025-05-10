import os
import json
import asyncio
import websockets
import requests
from dotenv import load_dotenv

load_dotenv()

class GladiaSTTClient:
    def __init__(self):
        self.api_key = os.getenv("GLADIA_API_KEY")
        self.base_url = "https://api.gladia.io/v2/live"
        self.websocket_url = None
        self.session_id = None
    
    def initialize_session(self, encoding="wav/pcm", sample_rate=16000, bit_depth=16, channels=1):
        """Initialize a new Gladia STT session and get the websocket URL"""
        headers = {
            "Content-Type": "application/json",
            "x-gladia-key": self.api_key
        }
        
        payload = {
            "encoding": encoding,
            "sample_rate": sample_rate,
            "bit_depth": bit_depth,
            "channels": channels
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for 4XX/5XX status codes
            
            data = response.json()
            self.session_id = data.get("id")
            self.websocket_url = data.get("url")
            return data
        except requests.exceptions.RequestException as e:
            if hasattr(e, 'response') and e.response is not None:
                error_msg = f"Failed to initialize Gladia session: {e.response.text}"
            else:
                error_msg = f"Failed to initialize Gladia session: {str(e)}"
            raise Exception(error_msg)
    
    async def connect_websocket(self, on_message=None, on_error=None, on_close=None):
        """
        Connect to the Gladia websocket with callback functions
        
        Parameters:
        - on_message: Callback function that receives message data
        - on_error: Callback function that receives error information
        - on_close: Callback function called when connection closes
        
        Returns: The websocket connection object
        """
        if not self.websocket_url:
            raise Exception("Session not initialized. Call initialize_session first.")
            
        try:
            websocket = await websockets.connect(self.websocket_url)
            
            # Start listening for messages in a background task
            async def message_handler():
                try:
                    while True:
                        try:
                            message = await websocket.recv()
                            if on_message:
                                data = json.loads(message)
                                await on_message(data)
                        except websockets.exceptions.ConnectionClosed as e:
                            if on_close:
                                await on_close(e.code, e.reason)
                            break
                        except Exception as e:
                            if on_error:
                                await on_error(e)
                except Exception as e:
                    if on_error:
                        await on_error(e)
            
            # Start the message handler
            asyncio.create_task(message_handler())
            
            return websocket
            
        except Exception as e:
            if on_error:
                await on_error(e)
            raise e
    
    async def send_audio_chunk(self, websocket, chunk):
        """Send a single audio chunk to the websocket"""
        await websocket.send(chunk)
    
    async def end_stream(self, websocket):
        """Send end of stream signal to the websocket"""
        await websocket.send(json.dumps({"event": "end_of_stream"}))
    
    async def stream_audio(self, audio_chunk_generator):
        """Stream audio chunks to the Gladia websocket and receive transcriptions"""
        if not self.websocket_url:
            raise Exception("Session not initialized. Call initialize_session first.")
        
        async with websockets.connect(self.websocket_url) as websocket:
            # Start sending audio chunks
            async for chunk in audio_chunk_generator:
                await websocket.send(chunk)
                
                # Check for responses
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    yield json.loads(response)
                except asyncio.TimeoutError:
                    # No response yet, continue sending
                    pass
            
            # Send end of stream signal
            await websocket.send(json.dumps({"event": "end_of_stream"}))
            
            # Get final responses
            try:
                while True:
                    response = await websocket.recv()
                    yield json.loads(response)
            except websockets.exceptions.ConnectionClosed:
                pass 