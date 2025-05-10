import asyncio
import os
import sys
import json
import signal
import time
import random
import tempfile
import threading
import queue
import io
import websockets
from dotenv import load_dotenv

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from text_to_speech import ElevenLabsTTSClient
from elevenlabs import stream

# Handle Ctrl+C gracefully
signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

# Sample AI responses
AI_RESPONSES = [
    "Hello! I'd be happy to help you find information about that ticket. Let me look up the details for you.",
    "I see that ticket #TK-4872 was created on May 3rd regarding an issue with the checkout process on your website.",
    "The engineering team has been working on it and left a note yesterday that they've identified the root cause. It appears to be a conflict with one of the third-party payment processors.",
    "They've implemented a fix that's currently in testing. According to the latest update, they expect to deploy it to production by tomorrow morning.",
    "Would you like me to notify the customer about this timeline? I can prepare a message explaining the situation and when they can expect the issue to be resolved."
]

# Test record of generated audio
class AudioRecord:
    def __init__(self):
        self.total_bytes = 0
        self.chunks_received = 0
        self.full_audio = bytearray()
        
    def add_chunk(self, chunk):
        if isinstance(chunk, bytes):
            self.chunks_received += 1
            self.total_bytes += len(chunk)
            self.full_audio.extend(chunk)
            
    def write_to_file(self, filename="test_output.mp3"):
        if self.total_bytes > 0:
            with open(filename, "wb") as f:
                f.write(self.full_audio)
            print(f"Wrote {self.total_bytes/1024:.2f} KB to {filename}")
            return filename
        return None

# Simulate LLM token-by-token streaming patterns
async def simulate_llm_streaming(text, min_chunk=1, max_chunk=5, min_delay=0.05, max_delay=0.2):
    """Simulate LLM streaming by breaking text into small chunks with variable timing"""
    words = text.split()
    i = 0
    
    while i < len(words):
        # Determine how many words to send in this chunk
        chunk_size = min(random.randint(min_chunk, max_chunk), len(words) - i)
        chunk = " ".join(words[i:i+chunk_size])
        
        # Add punctuation to the chunk if it contains it
        i += chunk_size
        
        # Return the chunk
        yield chunk
        
        # Simulate thinking/processing delay
        await asyncio.sleep(random.uniform(min_delay, max_delay))

class MockWebSocket:
    """Mock WebSocket class to capture and display audio chunks"""
    def __init__(self, audio_record=None):
        self.received_bytes = 0
        self.chunks_received = 0
        self.messages = []
        self.audio_record = audio_record
        
    async def send_json(self, data):
        print(f"Sending config: {json.dumps(data)}")
        return None
        
    async def send_bytes(self, data):
        self.chunks_received += 1
        self.received_bytes += len(data)
        print(f"\rReceived audio chunk: {self.chunks_received} (Total: {self.received_bytes/1024:.2f} KB)", end="")
        
        # Record audio if we have a recorder
        if self.audio_record:
            self.audio_record.add_chunk(data)
            
        return None
        
    async def receive(self):
        if not self.messages:
            # Create a close message to end the loop
            return {"type": "websocket.disconnect"}
            
        msg = self.messages.pop(0)
        return {"text": msg}
        
    def queue_message(self, message):
        self.messages.append(message)

async def test_direct_streaming(play_audio=False):
    """Test streaming TTS directly with pre-prepared text"""
    print("\n==== Testing Direct TTS Streaming ====")
    client = ElevenLabsTTSClient()
    
    text = "This is a direct streaming test of the text-to-speech system. It should convert this entire paragraph in one go."
    
    print(f"Synthesizing: {text}")
    
    if play_audio:
        # Stream directly using the ElevenLabs stream function (the best option for seamless playback)
        print("Playing audio with ElevenLabs stream function...")
        await client.synthesize_text(
            text=text,
            use_fast_model=True,
            play_audio=True
        )
    else:
        # Just collect stats if not playing
        audio_record = AudioRecord()
        mock_ws = MockWebSocket(audio_record=audio_record)
        
        await client.stream_text_to_websocket(
            websocket=mock_ws,
            text=text,
            use_fast_model=True
        )
        
        # Wait for streaming to complete
        await asyncio.sleep(5)
        print(f"\nCompleted! Received {mock_ws.chunks_received} audio chunks ({mock_ws.received_bytes/1024:.2f} KB)")

async def test_llm_streaming(play_audio=False):
    """Test TTS with simulated LLM streaming"""
    print("\n==== Testing LLM-style TTS Streaming ====")
    client = ElevenLabsTTSClient()
    
    # Choose a random response from our samples
    response = random.choice(AI_RESPONSES)
    print(f"Full response that will be streamed: \n\"{response}\"\n")
    
    if play_audio:
        # For audio playback, we'll collect the full text first, then play it
        # (this demonstrates the current best approach for seamless playback)
        print("Collecting streamed text chunks...")
        
        full_text = ""
        async for chunk in simulate_llm_streaming(response):
            print(f"\nText chunk: \"{chunk}\"")
            full_text += chunk
            await asyncio.sleep(0.1)
            
        print(f"\nFull collected text: \"{full_text}\"")
        print("Playing complete audio...")
        
        # Play the full text seamlessly
        await client.synthesize_text(
            text=full_text,
            use_fast_model=True,
            play_audio=True
        )
    else:
        # Create a mock websocket and audio recorder
        audio_record = AudioRecord()
        mock_ws = MockWebSocket(audio_record=audio_record)
        
        # Configure client with initial message
        mock_ws.queue_message(json.dumps({
            "voice_id": "21m00Tcm4TlvDq8ikWAM",
            "use_fast_model": True
        }))
        
        # Start processing in the background
        process_task = asyncio.create_task(
            client.process_streaming_text(
                websocket=mock_ws,
                use_fast_model=True
            )
        )
        
        print("Streaming response word by word...")
        
        # Queue up the streaming chunks with delays to simulate LLM behavior
        async for chunk in simulate_llm_streaming(response):
            print(f"\nSending text chunk: \"{chunk}\"")
            mock_ws.queue_message(chunk)
            
            # Small delay to let the TTS client process
            await asyncio.sleep(0.1)
        
        # Force process remaining text
        print("\nSending flush command...")
        mock_ws.queue_message(json.dumps({"command": "flush"}))
        
        # Wait for processing to complete
        await asyncio.sleep(5)
        
        # Cancel the task
        process_task.cancel()
        try:
            await process_task
        except asyncio.CancelledError:
            pass
        
        print(f"Completed! Received {mock_ws.chunks_received} audio chunks ({mock_ws.received_bytes/1024:.2f} KB)")
        
        # Write collected audio to file
        if mock_ws.received_bytes > 0:
            audio_record.write_to_file(f"llm_streaming_test_{int(time.time())}.mp3")

async def test_real_websocket_connection(play_audio=False):
    """Test connecting to the actual TTS websocket endpoint"""
    print("\n==== Testing Real WebSocket Connection ====")
    
    # Connect to the actual server
    try:
        uri = "ws://localhost:8000/ws/tts"
        print(f"Connecting to {uri}...")
        
        async with websockets.connect(uri) as websocket:
            # Send configuration
            await websocket.send(json.dumps({
                "voice_id": "21m00Tcm4TlvDq8ikWAM",
                "use_fast_model": True
            }))
            
            # Wait for ready message
            response = json.loads(await websocket.recv())
            print(f"Received: {response}")
            
            # Choose a random response from our samples
            response = random.choice(AI_RESPONSES)
            print(f"Full response that will be streamed: \n\"{response}\"\n")
            
            if play_audio:
                # For playback, collect all text first then play afterward
                full_text = ""
                print("Collecting streamed text chunks...")
                
                async for chunk in simulate_llm_streaming(response):
                    print(f"\nText chunk: \"{chunk}\"")
                    await websocket.send(chunk)
                    full_text += chunk
                
                # Flush remaining text
                await websocket.send(json.dumps({"command": "flush"}))
                
                print(f"\nCollected all text. Saving to file for playback.")
                
                # Wait for all audio data
                audio_record = AudioRecord()
                bytes_received = 0
                
                try:
                    while True:
                        audio_data = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        
                        # If it's binary data, it's audio
                        if isinstance(audio_data, bytes):
                            bytes_received += len(audio_data)
                            audio_record.add_chunk(audio_data)
                            print(f"\rReceived audio: {bytes_received/1024:.2f} KB", end="")
                        else:
                            # Likely a status message
                            try:
                                msg = json.loads(audio_data)
                                print(f"\nReceived message: {msg}")
                                if msg.get("status") == "complete":
                                    break
                            except:
                                print(f"\nReceived non-JSON message: {audio_data}")
                except asyncio.TimeoutError:
                    print("\nNo more data received.")
                
                # Save audio to file and play it
                if bytes_received > 0:
                    filename = audio_record.write_to_file(f"websocket_test_{int(time.time())}.mp3")
                    print(f"Playing audio from file {filename}")
                    
                    # Play using ElevenLabs client
                    client = ElevenLabsTTSClient()
                    with open(filename, "rb") as f:
                        audio_data = f.read()
                    
                    # Use direct playback with MPV
                    from elevenlabs import play
                    play(audio_data)
            else:
                # Just send the data and print statistics
                chunks_sent = 0
                bytes_received = 0
                
                async for chunk in simulate_llm_streaming(response):
                    print(f"\nSending text chunk: \"{chunk}\"")
                    await websocket.send(chunk)
                    chunks_sent += 1
                    
                    # Try to receive any audio data that might be ready
                    try:
                        # Set a short timeout to avoid blocking
                        audio_data = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        
                        # If it's binary data, it's audio
                        if isinstance(audio_data, bytes):
                            bytes_received += len(audio_data)
                            print(f"\rReceived audio: {bytes_received/1024:.2f} KB", end="")
                    except asyncio.TimeoutError:
                        # No data available yet, continue
                        pass
                
                # Flush any remaining text
                await websocket.send(json.dumps({"command": "flush"}))
                
                # Wait for any remaining audio chunks
                print("\nWaiting for remaining audio chunks...")
                try:
                    while True:
                        audio_data = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        
                        # If it's binary data, it's audio
                        if isinstance(audio_data, bytes):
                            bytes_received += len(audio_data)
                            print(f"\rReceived audio: {bytes_received/1024:.2f} KB", end="")
                        else:
                            # Likely a status message
                            try:
                                msg = json.loads(audio_data)
                                print(f"\nReceived message: {msg}")
                            except:
                                print(f"\nReceived non-JSON message: {audio_data}")
                except asyncio.TimeoutError:
                    # No more data, we're done
                    print("\nNo more data received, test complete.")
    except Exception as e:
        print(f"Error in WebSocket test: {e}")
        print("Make sure the server is running at localhost:8000")

async def main():
    """Run all tests"""
    load_dotenv()
    
    print("==== Monsieur TTS Streaming Test ====")
    print("This test will simulate an LLM streaming text to the TTS client.")
    
    # Check if we should play audio
    play_audio = input("Do you want to listen to the generated audio? (y/n): ").lower().strip() == 'y'
    
    try:
        # Test 1: Direct streaming
        await test_direct_streaming(play_audio=play_audio)
        
        # Test 2: LLM-style streaming with mock websocket
        await test_llm_streaming(play_audio=play_audio)
        
        # Test 3: Real websocket connection (if server is running)
        server_test = input("\nDo you want to test with the actual server? (y/n): ").lower().strip() == 'y'
        if server_test:
            await test_real_websocket_connection(play_audio=play_audio)
        
        print("\nAll tests completed!")
    finally:
        pass

if __name__ == "__main__":
    asyncio.run(main())
