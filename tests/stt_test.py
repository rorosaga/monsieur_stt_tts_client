import asyncio
import os
import sys
import signal
import time
import json
import base64
from dotenv import load_dotenv

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pyaudio
import websockets

# Handle Ctrl+C gracefully
signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

# Audio constants
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = 3200
SAMPLE_RATE = 16000

async def main():
    # Load environment variables
    load_dotenv()
    
    print("==== Monsieur STT Test ====")
    print("This test will record audio from your microphone and transcribe it in real-time.")
    
    # Get API key from .env
    api_key = os.getenv("GLADIA_API_KEY")
    if not api_key:
        print("Error: GLADIA_API_KEY not found in .env file")
        return
    
    # Initialize the session directly using the Gladia API
    config = {
        "encoding": "wav/pcm",
        "sample_rate": SAMPLE_RATE,
        "bit_depth": 16,
        "channels": CHANNELS,
        "language_config": {
            "languages": [],  # Empty list means auto-detect
            "code_switching": True,
        }
    }
    
    print("Initializing Gladia STT session...")
    
    try:
        # Initialize session
        import requests
        response = requests.post(
            "https://api.gladia.io/v2/live",
            headers={"X-Gladia-Key": api_key},
            json=config,
            timeout=3
        )
        
        if not response.ok:
            print(f"Error: {response.status_code}: {response.text}")
            return
            
        session_data = response.json()
        websocket_url = session_data["url"]
        session_id = session_data["id"]
        
        print(f"Session initialized: {session_id}")
        print(f"WebSocket URL: {websocket_url}")
        
    except Exception as e:
        print(f"Error initializing session: {e}")
        return
    
    # Set up audio recording
    p = pyaudio.PyAudio()
    
    # Connect to the websocket
    async with websockets.connect(websocket_url) as websocket:
        print("Connected to Gladia WebSocket")
        print("\n################ Begin session ################\n")
        
        # Function to handle incoming messages
        async def receive_messages():
            async for message in websocket:
                content = json.loads(message)
                if content.get("type") == "transcript" and content.get("data", {}).get("is_final"):
                    # Format final transcriptions
                    start = content["data"]["utterance"]["start"]
                    end = content["data"]["utterance"]["end"]
                    text = content["data"]["utterance"]["text"].strip()
                    print(f"\n[{start:.2f}s --> {end:.2f}s] {text}")
                elif content.get("type") == "transcript":
                    # Interim transcriptions
                    text = content.get("data", {}).get("utterance", {}).get("text", "").strip()
                    print(f"\r\033[K[interim] {text}", end="", flush=True)
                elif content.get("type") == "post_final_transcript":
                    print("\n################ End of session ################\n")
                    print(json.dumps(content, indent=2, ensure_ascii=False))
                else:
                    # Other message types
                    pass  # Ignore acknowledgment messages
        
        # Start message receiver task
        receive_task = asyncio.create_task(receive_messages())
        
        # Wait for user to press Enter to start recording
        input("Press Enter to start recording (Ctrl+C to stop)...")
        print("Recording... Press Ctrl+C to stop.")
        
        # Open audio stream
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=FRAMES_PER_BUFFER
        )
        
        # Record and send audio
        recording = True
        frames = []
        
        try:
            while recording:
                # Read audio data
                data = stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                frames.append(data)
                
                # Encode and send to Gladia
                data_b64 = base64.b64encode(data).decode("utf-8")
                json_data = json.dumps({
                    "type": "audio_chunk", 
                    "data": {"chunk": data_b64}
                })
                
                await websocket.send(json_data)
                await asyncio.sleep(0.1)  # Send chunks every 100ms
                
        except (KeyboardInterrupt, SystemExit):
            print("\nStopping recording...")
        finally:
            # Stop recording
            recording = False
            stream.stop_stream()
            stream.close()
            
            # Send stop_recording message
            try:
                await websocket.send(json.dumps({"type": "stop_recording"}))
                print("Sent stop_recording signal")
                
                # Wait a moment for final transcriptions
                await asyncio.sleep(2)
            except:
                pass
            
            # Save the recording
            filename = f"test_recording_{int(time.time())}.wav"
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
            
            import wave
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(b''.join(frames))
            
            print(f"Recording saved to {filename}")
            print("Test completed.")
            
            # Clean up
            p.terminate()
            
            # Cancel the receive task
            receive_task.cancel()
            try:
                await receive_task
            except asyncio.CancelledError:
                pass

if __name__ == "__main__":
    asyncio.run(main())
