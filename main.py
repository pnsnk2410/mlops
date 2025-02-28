import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastrtc import (ReplyOnPause, Stream, get_stt_model, get_tts_model)

# Load the Qwen model and tokenizer from Hugging Face
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize STT and TTS models
stt_model = get_stt_model()
tts_model = get_tts_model()

def echo(audio):
    # Convert audio to text using STT
    prompt = stt_model.stt(audio)
    
    # Generate a response using the Qwen model
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Convert the response to audio using TTS
    for audio_chunk in tts_model.stream_tts_sync(response):
        yield audio_chunk

# Create and start the stream
stream = Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")