from transformers import MusicgenForConditionalGeneration
import torch
from transformers import AutoProcessor
from IPython.display import Audio
import scipy

MAX_TOKENS = 1024

# Load the model
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

# Set the device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.to(device)

# Generate unconditional audio
unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
audio_values = model.generate(**unconditional_inputs, do_sample=True, max_new_tokens=MAX_TOKENS)

# Get the sampling rate
sampling_rate = model.config.audio_encoder.sampling_rate

scipy.io.wavfile.write("musicgen_unconditional.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())
audio_length_in_s = MAX_TOKENS / model.config.audio_encoder.frame_rate
audio_length_in_s

# Load the processor
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

# Process the inputs
inputs = processor(
    text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
    padding=True,
    return_tensors="pt",
)
audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=MAX_TOKENS)


scipy.io.wavfile.write("musicgen_sample.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())
audio_length_in_s = MAX_TOKENS / model.config.audio_encoder.frame_rate
audio_length_in_s
