import os
import asyncio
import json
import logging
from typing import Optional, List, Dict, Any
from fastapi import WebSocket
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class ElevenLabsTTSClient:
    def __init__(self):
        self.api_key = os.getenv("ELEVENLABS_API_KEY")
        self.client = ElevenLabs(api_key=self.api_key)
        self.default_model = "eleven_multilingual_v2"  # for higher quality
        self.fast_model = "eleven_flash_v2_5"          # for faster responses
        
    async def synthesize_text(self, 
                             text: str, 
                             voice_id: str = "21m00Tcm4TlvDq8ikWAM", 
                             use_fast_model: bool = False,
                             play_audio: bool = False) -> bytes:
        """
        Synthesize text to speech
        
        Args:
            text: The text to convert to speech
            voice_id: ElevenLabs voice ID
            use_fast_model: Whether to use the faster model
            play_audio: Whether to play the audio locally
            
        Returns:
            Complete audio as bytes
        """
        model_id = self.fast_model if use_fast_model else self.default_model
        logger.info(f"Synthesizing text using model: {model_id}")
        
        # Use a thread for the non-async ElevenLabs API call
        audio_bytes = b''
        
        # Define a function to run in an executor
        def generate_audio():
            nonlocal audio_bytes
            audio_stream = self.client.text_to_speech.convert_as_stream(
                text=text,
                voice_id=voice_id,
                model_id=model_id
            )
            
            # If we want to play audio locally
            if play_audio:
                stream(audio_stream)
                return b''  # Already played, so return empty
            
            # Otherwise collect audio bytes
            chunks = []
            for chunk in audio_stream:
                if isinstance(chunk, bytes):
                    chunks.append(chunk)
            
            return b''.join(chunks)
        
        # Run in a thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(None, generate_audio)
        
        return audio_bytes
    
    async def stream_text_to_websocket(self, 
                                      websocket: WebSocket, 
                                      text: str, 
                                      voice_id: str = "21m00Tcm4TlvDq8ikWAM",
                                      use_fast_model: bool = False) -> None:
        """
        Stream TTS audio to a WebSocket as it's generated
        
        Args:
            websocket: WebSocket connection to send audio chunks to
            text: Text to synthesize
            voice_id: ElevenLabs voice ID
            use_fast_model: Whether to use the faster model
        """
        model_id = self.fast_model if use_fast_model else self.default_model
        logger.info(f"Streaming text to websocket using model: {model_id}")
        
        # Define a function to run in an executor
        async def stream_audio():
            try:
                audio_stream = self.client.text_to_speech.convert_as_stream(
                    text=text,
                    voice_id=voice_id,
                    model_id=model_id
                )
                
                chunk_count = 0
                for chunk in audio_stream:
                    if isinstance(chunk, bytes):
                        # Send audio chunk to websocket
                        await websocket.send_bytes(chunk)
                        chunk_count += 1
                        
                        # Small delay to avoid overwhelming the connection
                        await asyncio.sleep(0.01)
                
                # Send end-of-stream message
                await websocket.send_json({
                    "status": "complete",
                    "chunks_sent": chunk_count
                })
                
                logger.info(f"Sent {chunk_count} audio chunks to websocket")
                
            except Exception as e:
                logger.error(f"Error in stream_audio: {e}")
                # Send error message
                await websocket.send_json({
                    "error": str(e)
                })
        
        # Start the streaming task
        asyncio.create_task(stream_audio())
    
    async def process_streaming_text(self, 
                                   websocket: WebSocket, 
                                   voice_id: str = "21m00Tcm4TlvDq8ikWAM",
                                   use_fast_model: bool = False) -> None:
        """
        Process text chunks coming from a WebSocket and return audio
        
        Args:
            websocket: WebSocket connection for bidirectional communication
            voice_id: ElevenLabs voice ID
            use_fast_model: Whether to use the faster model
        """
        try:
            active_synthesis = False
            buffer = ""
            
            while True:
                message = await websocket.receive()
                
                # Handle text messages (JSON or raw text)
                if "text" in message:
                    try:
                        data = json.loads(message["text"])
                        # Check if it's a control message
                        if "command" in data:
                            if data["command"] == "flush":
                                # Process any remaining text in buffer
                                if buffer:
                                    await self.stream_text_to_websocket(
                                        websocket, 
                                        buffer, 
                                        voice_id, 
                                        use_fast_model
                                    )
                                    buffer = ""
                                    
                            elif data["command"] == "synthesize":
                                text = data.get("text", "")
                                if text:
                                    await self.stream_text_to_websocket(
                                        websocket, 
                                        text, 
                                        voice_id, 
                                        use_fast_model
                                    )
                                    
                            continue
                            
                        # If it has a text field, extract it
                        if "text" in data:
                            text_chunk = data["text"]
                        else:
                            text_chunk = message["text"]
                            
                    except json.JSONDecodeError:
                        # Not JSON, treat as raw text
                        text_chunk = message["text"]
                    
                    # If the message ends with a sentence-ending character,
                    # process the buffer plus this chunk
                    if text_chunk.strip().endswith((".", "!", "?", ":", ";")) and buffer:
                        buffer += text_chunk
                        await self.stream_text_to_websocket(
                            websocket, 
                            buffer, 
                            voice_id, 
                            use_fast_model
                        )
                        buffer = ""
                    else:
                        # Add to buffer
                        buffer += text_chunk
                        
                        # If buffer gets too long, process it anyway
                        if len(buffer) > 200:  # Arbitrary length threshold
                            await self.stream_text_to_websocket(
                                websocket, 
                                buffer, 
                                voice_id, 
                                use_fast_model
                            )
                            buffer = ""
                
                # Handle close message
                elif "type" in message and message["type"] == "websocket.disconnect":
                    # Process any remaining text
                    if buffer:
                        await self.stream_text_to_websocket(
                            websocket, 
                            buffer, 
                            voice_id, 
                            use_fast_model
                        )
                    break
        
        except Exception as e:
            logger.error(f"Error in process_streaming_text: {e}")
            await websocket.send_json({
                "error": str(e)
            })

# Example standalone usage
if __name__ == "__main__":
    client = ElevenLabsTTSClient()
    
    # Test the API by synthesizing text and playing it
    asyncio.run(client.synthesize_text(
        "This is a test of the ElevenLabs text to speech system using the streaming API.",
        play_audio=True
    ))