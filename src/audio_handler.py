import asyncio
import pyaudio
import wave
import os
import time
from typing import Optional, Callable, Dict, Any, List
import logging
import websockets
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioHandler:
    def __init__(self, channels=1, rate=16000, chunk=1024, format=pyaudio.paInt16):
        """
        Initialize the audio handler for recording and processing audio.
        
        Args:
            channels: Number of audio channels (1 for mono, 2 for stereo)
            rate: Sample rate in Hz
            chunk: Audio chunk size
            format: PyAudio format (default: 16-bit int)
        """
        self.channels = channels
        self.rate = rate
        self.chunk = chunk
        self.format = format
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.frames = []
        self.recording_task = None
        self.websocket = None
        
    async def start_recording(self, stream_to_websocket: Optional[websockets.WebSocketClientProtocol] = None) -> None:
        """
        Start recording audio and optionally stream it to a websocket.
        
        Args:
            stream_to_websocket: Optional websocket to stream audio chunks to
        """
        if self.is_recording:
            logger.warning("Already recording, ignoring start_recording call")
            return
            
        self.websocket = stream_to_websocket
        self.is_recording = True
        self.frames = []
        
        # Start audio stream
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        
        logger.info("Recording started")
        
        # Start recording task
        self.recording_task = asyncio.create_task(self._record_audio())
        
    async def _record_audio(self) -> None:
        """Background task to record audio and stream it if needed"""
        try:
            while self.is_recording:
                if self.stream.is_stopped():
                    self.stream.start_stream()
                    
                # Read audio data
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                self.frames.append(data)
                
                # If we're streaming to a websocket, send the chunk
                if self.websocket:
                    try:
                        await self.websocket.send(data)
                    except Exception as e:
                        logger.error(f"Error sending audio to websocket: {e}")
                        self.websocket = None  # Stop trying to stream
                
                # Small pause to prevent CPU hogging
                await asyncio.sleep(0.01)
                
        except Exception as e:
            logger.error(f"Error in _record_audio: {e}")
            self.is_recording = False
            
    async def stop_recording(self) -> List[bytes]:
        """
        Stop the recording process and return all captured frames.
        
        Returns:
            List of audio frames (bytes)
        """
        if not self.is_recording:
            logger.warning("Not recording, ignoring stop_recording call")
            return self.frames
            
        self.is_recording = False
        
        # If we have an active recording task, wait for it to complete
        if self.recording_task:
            try:
                # Give the task a moment to clean up
                await asyncio.wait_for(self.recording_task, timeout=0.5)
            except asyncio.TimeoutError:
                # If it takes too long, cancel it
                self.recording_task.cancel()
                
        # Stop and close the audio stream
        if self.stream and not self.stream.is_stopped():
            self.stream.stop_stream()
        if self.stream:
            self.stream.close()
            self.stream = None
            
        logger.info(f"Recording stopped, captured {len(self.frames)} frames")
        
        # If we were streaming to a websocket, send end of stream
        if self.websocket:
            try:
                await self.websocket.send(json.dumps({"event": "end_of_stream"}))
            except Exception as e:
                logger.error(f"Error sending end_of_stream: {e}")
            self.websocket = None
            
        return self.frames
        
    def save_to_file(self, filename: str) -> str:
        """
        Save recorded audio to a WAV file.
        
        Args:
            filename: Path to save the WAV file
            
        Returns:
            The path to the saved file
        """
        if not self.frames:
            logger.warning("No audio frames to save")
            return ""
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(self.frames))
            
        logger.info(f"Audio saved to {filename}")
        return filename
        
    async def handle_phone_call(self, call_id: str, stt_websocket: websockets.WebSocketClientProtocol) -> Dict[str, Any]:
        """
        Handle a complete phone call: record audio, stream to STT, and save the call.
        
        Args:
            call_id: Unique identifier for the call
            stt_websocket: WebSocket connected to the STT service
            
        Returns:
            Dict with call information and file path
        """
        timestamp = int(time.time())
        filename = f"calls/{call_id}_{timestamp}.wav"
        
        # Start recording and streaming to the STT websocket
        await self.start_recording(stream_to_websocket=stt_websocket)
        
        # This would normally wait for the call to end
        # For demo purposes, we'll just wait a short time
        # In a real app, you would return this function and let external code
        # call stop_recording when the call actually ends
        
        call_info = {
            "call_id": call_id,
            "timestamp": timestamp,
            "audio_file": filename,
            "duration": 0
        }
        
        return call_info
        
    def __del__(self):
        """Cleanup resources when object is destroyed"""
        if self.stream:
            if not self.stream.is_stopped():
                self.stream.stop_stream()
            self.stream.close()
            
        if self.audio:
            self.audio.terminate()


# Factory function to easily create an audio handler for phone calls
async def create_phone_call_handler(call_id: str, stt_client):
    """
    Create an audio handler and set up a call with STT integration.
    
    Args:
        call_id: Unique ID for the call
        stt_client: Instance of GladiaSTTClient
        
    Returns:
        Tuple of (AudioHandler, websocket, call_info)
    """
    # Initialize the session with Gladia
    session_data = stt_client.initialize_session()
    
    # Set up transcription handlers
    transcriptions = []
    
    async def on_message(data):
        transcriptions.append(data)
        logger.info(f"Transcription: {data}")
    
    async def on_error(error):
        logger.error(f"STT error: {error}")
    
    async def on_close(code, reason):
        logger.info(f"STT connection closed: {code} - {reason}")
    
    # Connect to the Gladia websocket
    stt_ws = await stt_client.connect_websocket(
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    
    # Create and configure the audio handler
    audio_handler = AudioHandler()
    
    return audio_handler, stt_ws, transcriptions 