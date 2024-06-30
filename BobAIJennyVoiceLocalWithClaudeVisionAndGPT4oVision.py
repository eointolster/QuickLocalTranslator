from TTS.api import TTS
import pygame
import time
import sounddevice as sd
import numpy as np
import whisper
import scipy.signal
import keyboard
import os
import aiohttp
import asyncio
import json
import re
import logging

from dotenv import load_dotenv
import anthropic
from openai import OpenAI
import base64

# Load environment variables
load_dotenv()
anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define global frames for audio data
frames = []

# Initialize conversation history
conversation_history = []

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to use Claude Vision
async def claude_vision(image_path, prompt):
    base64_image = encode_image(image_path)
    
    message = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )
    
    return message.content[0].text

# Function to use GPT-4 Vision
async def gpt4_vision(image_path, prompt):
    base64_image = encode_image(image_path)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    
    return response.choices[0].message.content

# Function to play audio
def play_audio(file_path):
    print(f"Playing audio file: {file_path}")
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.quit()
        print(f"Audio file played successfully: {file_path}")
    except FileNotFoundError:
        print(f"Audio file not found: {file_path}")
    except Exception as e:
        print(f"Error playing audio file: {str(e)}")

# Function to record audio with * key
def record_audio_with_star_key(sample_rate=44100, channels=1):
    global frames
    frames.clear()

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    with sd.InputStream(device=1, samplerate=sample_rate, channels=channels, callback=callback):
        print("Press and hold the '*' key to start recording...")
        keyboard.wait('*')
        print("Recording... Release the '*' key to stop.")
        while keyboard.is_pressed('*'):
            pass
        print("Recording stopped.")

    return np.concatenate(frames, axis=0)

# Function to transcribe audio using Whisper
def transcribe_with_whisper(audio, sample_rate=16000):
    if sample_rate != 16000:
        audio = scipy.signal.resample_poly(audio, 16000, sample_rate)
    audio = audio.flatten()
    audio = audio.astype(np.float32)
    model = whisper.load_model("base")
    result = model.transcribe(audio)
    return result['text']

# Function to update conversation history
def update_conversation_history(role, content):
    global conversation_history
    conversation_history.append({"role": role, "content": content})
    if len(conversation_history) > 20:  # Keep only the last 20 messages
        conversation_history = conversation_history[-20:]

# Function to reset the conversation history periodically
def reset_conversation_history():
    global conversation_history
    conversation_history = []

# Async function to send prompt to Ollama API
async def send_to_ollama_api(prompt):
    async with aiohttp.ClientSession() as session:
        if not prompt:
            logging.error("Empty prompt received for Ollama API.")
            return None

        history_prompt = ""
        for entry in conversation_history:
            role = entry['role']
            content = entry['content']
            history_prompt += f"{role}: {content}\n"

        full_prompt = f"{history_prompt}user: {prompt}\nassistant:"

        payload = {"prompt": full_prompt, "model": "bob:latest"}
        headers = {"Content-Type": "application/json"}
        async with session.post("http://localhost:11434/api/generate", headers=headers, json=payload) as response:
            response_text = await response.text()
            logging.debug(f"Raw response from Ollama: {response_text}")

            try:
                fixed_response = "[" + re.sub(r'}\s*{', '},{', response_text) + "]"
                responses = json.loads(fixed_response)
                full_response = ''.join(item['response'] for item in responses if not item['done'])
                return full_response
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON response from Ollama API: {str(e)}")
                return None

# Main loop
async def main():
    sd.default.device = 1  # Set the desired input device here
    sample_rate = 44100
    conversation_count = 0
    reset_interval = 10  # Reset after every 10 conversations
    image_path = "myImages/Mandlebulb.png"  # Update this with the actual image path

    while True:
        if keyboard.is_pressed('1'):  # Numpad 1 for Claude Vision
            print("Analyzing image with Claude Vision...")
            claude_result = await claude_vision(image_path, "Describe this image in detail.")
            answer = await send_to_ollama_api(f"Analyze this image description: {claude_result}")
            print(f"Claude Vision Analysis: {answer}")
            play_audio_response(answer)
            while keyboard.is_pressed('1'):  # Wait for key release
                pass
        elif keyboard.is_pressed('2'):  # Numpad 2 for GPT-4 Vision
            print("Analyzing image with GPT-4 Vision...")
            gpt4_result = await gpt4_vision(image_path, "Describe this image in detail.")
            answer = await send_to_ollama_api(f"Analyze this image description: {gpt4_result}")
            print(f"GPT-4 Vision Analysis: {answer}")
            play_audio_response(answer)
            while keyboard.is_pressed('2'):  # Wait for key release
                pass
        elif keyboard.is_pressed('*'):
            audio = record_audio_with_star_key(sample_rate=sample_rate)
            if audio.size > 0:
                transcribed_text = transcribe_with_whisper(audio, sample_rate=sample_rate)
                print(f"Transcribed Text: {transcribed_text}")

                if not transcribed_text.strip():
                    continue

                update_conversation_history("user", transcribed_text)
                answer = await send_to_ollama_api(transcribed_text)
                print(f"Response: {answer}")
                play_audio_response(answer)

        if keyboard.is_pressed('esc'):  # Use 'esc' key to exit the program
            break

        await asyncio.sleep(0.1)  # Small delay to prevent high CPU usage

def play_audio_response(text):
    device = "cuda"
    tts = TTS(model_name="tts_models/en/jenny/jenny", progress_bar=False).to(device)
    tts.tts_to_file(text=text, file_path='output.wav')
    play_audio('output.wav')

if __name__ == "__main__":
    asyncio.run(main())