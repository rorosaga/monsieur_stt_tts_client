import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import asyncio
import os
from dotenv import load_dotenv
from uuid import uuid4
import time
from audio_handler import AudioHandler, create_phone_call_handler
from text_to_speech import ElevenLabsTTSClient

from speech_to_text import GladiaSTTClient

# Load environment variables
load_dotenv()

app = FastAPI(title="Monsieur STT/TTS API")
stt_client = GladiaSTTClient()
tts_client = ElevenLabsTTSClient()

# Add these variables at an appropriate place after app initialization
active_calls = {}
audio_handlers = {}

@app.get("/")
async def root():
    return {"message": "Welcome to Monsieur STT/TTS API"}

# Initialize Gladia session endpoint
@app.post("/init-stt-session")
async def init_stt_session(
    encoding: str = "wav/pcm", 
    sample_rate: int = 16000, 
    bit_depth: int = 16, 
    channels: int = 1
):
    try:
        session_data = stt_client.initialize_session(
            encoding=encoding,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            channels=channels
        )
        return session_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time STT
@app.websocket("/ws/stt")
async def websocket_stt(websocket: WebSocket):
    await websocket.accept()
    
    # First initialize a session
    try:
        session_data = stt_client.initialize_session()
        await websocket.send_json(session_data)
        
        # Create async generator for audio chunks
        async def audio_chunk_generator():
            while True:
                data = await websocket.receive_bytes()
                if not data:
                    break
                yield data
        
        # Stream audio and return transcriptions
        async for transcription in stt_client.stream_audio(audio_chunk_generator()):
            await websocket.send_json(transcription)
            
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

# Example of how to use the new connect_websocket method
@app.websocket("/ws/stt-simple")
async def websocket_stt_simple(websocket: WebSocket):
    await websocket.accept()
    
    # Initialize a session
    try:
        session_data = stt_client.initialize_session()
        await websocket.send_json(session_data)
        
        # Define callback functions for the Gladia websocket
        async def on_message(data):
            await websocket.send_json(data)
            
        async def on_error(error):
            await websocket.send_json({"error": str(error)})
            
        async def on_close(code, reason):
            await websocket.send_json({"closed": True, "code": code, "reason": reason})
        
        # Connect to the Gladia websocket
        gladia_ws = await stt_client.connect_websocket(
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        # Listen for audio chunks from client and forward to Gladia
        while True:
            try:
                data = await websocket.receive_bytes()
                await stt_client.send_audio_chunk(gladia_ws, data)
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                break
        
        # End the stream when done
        await stt_client.end_stream(gladia_ws)
        
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

# Example endpoint for file-based speech-to-text
class STTRequest(BaseModel):
    audio_file_path: str
    language: Optional[str] = "en"

@app.post("/stt")
async def speech_to_text(request: STTRequest):
    try:
        # Placeholder for actual STT processing
        return {"text": f"Transcription from {request.audio_file_path}", "language": request.language}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Example endpoint for text-to-speech
class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = "default"
    language: Optional[str] = "en"

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """Generate audio from text (non-streaming)"""
    try:
        audio_bytes = await tts_client.synthesize_text(
            text=request.text,
            voice_id=request.voice or "21m00Tcm4TlvDq8ikWAM",
            use_fast_model="flash" in request.model.lower() if request.model else False
        )
        
        # Save audio to file with timestamp
        timestamp = int(time.time())
        filename = f"tts/{timestamp}.mp3"
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        with open(filename, "wb") as f:
            f.write(audio_bytes)
        
        return {
            "audio_url": filename,
            "text": request.text,
            "voice": request.voice,
            "language": request.language
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add these endpoints to main.py
@app.post("/calls/start")
async def start_call(background_tasks: BackgroundTasks):
    """Start a new phone call with STT integration"""
    call_id = str(uuid4())
    
    # Create handler in the background
    audio_handler, stt_ws, transcriptions = await create_phone_call_handler(call_id, stt_client)
    
    # Store references
    audio_handlers[call_id] = audio_handler
    active_calls[call_id] = {
        "id": call_id,
        "start_time": time.time(),
        "websocket": stt_ws,
        "transcriptions": transcriptions,
        "status": "starting"
    }
    
    # Start recording in the background
    background_tasks.add_task(audio_handler.start_recording, stt_ws)
    
    # Update status
    active_calls[call_id]["status"] = "active"
    
    return {
        "call_id": call_id,
        "status": "active",
        "start_time": active_calls[call_id]["start_time"]
    }

@app.post("/calls/{call_id}/stop")
async def stop_call(call_id: str):
    """Stop an active call and save the recording"""
    if call_id not in active_calls:
        raise HTTPException(status_code=404, detail="Call not found")
    
    if call_id not in audio_handlers:
        raise HTTPException(status_code=400, detail="No audio handler for this call")
    
    # Stop recording
    audio_handler = audio_handlers[call_id]
    frames = await audio_handler.stop_recording()
    
    # Save the audio file
    timestamp = int(time.time())
    filename = f"calls/{call_id}_{timestamp}.wav"
    audio_handler.save_to_file(filename)
    
    # Update call info
    active_calls[call_id]["status"] = "completed"
    active_calls[call_id]["end_time"] = timestamp
    active_calls[call_id]["duration"] = timestamp - active_calls[call_id]["start_time"]
    active_calls[call_id]["audio_file"] = filename
    
    return {
        "call_id": call_id,
        "status": "completed",
        "duration": active_calls[call_id]["duration"],
        "audio_file": filename,
        "transcription_count": len(active_calls[call_id]["transcriptions"])
    }

@app.get("/calls/{call_id}")
async def get_call_info(call_id: str):
    """Get information about a call, including transcriptions"""
    if call_id not in active_calls:
        raise HTTPException(status_code=404, detail="Call not found")
        
    call_info = dict(active_calls[call_id])
    
    # Remove non-serializable items
    if "websocket" in call_info:
        del call_info["websocket"]
        
    return call_info

@app.get("/calls")
async def list_calls():
    """List all calls and their statuses"""
    calls_list = []
    for call_id, call_info in active_calls.items():
        calls_list.append({
            "call_id": call_id,
            "status": call_info["status"],
            "start_time": call_info["start_time"],
            "duration": call_info.get("duration", time.time() - call_info["start_time"])
        })
    
    return calls_list

@app.websocket("/ws/tts")
async def websocket_tts(websocket: WebSocket):
    """WebSocket endpoint for streaming text-to-speech"""
    await websocket.accept()
    
    try:
        # Process the first message to get configuration
        config = await websocket.receive_json()
        voice_id = config.get("voice_id", "21m00Tcm4TlvDq8ikWAM")
        use_fast_model = config.get("use_fast_model", False)
        
        await websocket.send_json({
            "status": "ready",
            "voice_id": voice_id,
            "model": "eleven_flash_v2_5" if use_fast_model else "eleven_multilingual_v2"
        })
        
        # Process streaming text
        await tts_client.process_streaming_text(
            websocket, 
            voice_id, 
            use_fast_model
        )
        
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
