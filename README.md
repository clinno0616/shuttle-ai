# Shuttle Jaguar Image Generator

A Streamlit web application that generates images from text descriptions using the Shuttle Jaguar model from ShuttleAI.

## Features

- Text-to-image generation using state-of-the-art Shuttle Jaguar model
- Adjustable image dimensions (256-1024px)
- Two optimization modes: Standard and Memory Efficient
- Customizable generation parameters:
  - Guidance Scale (1.0-20.0)
  - Number of Inference Steps (1-100)
  - Max Sequence Length (64-512)
- Optional seed setting for reproducible results
- Real-time VRAM usage monitoring
- One-click image download
- Memory optimization features for stable performance

## Requirements

```
streamlit
torch
diffusers == 0.24.0
Pillow
```

## Installation

Clone this repository:
```bash
git clone https://github.com/clinno0616/shuttle-ai.git
cd shuttle-ai
```
## Usage

1. Start the Streamlit application:
```bash
streamlit run app2.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Enter your text prompt in the input field

4. Adjust generation settings as needed:
   - Image dimensions
   - Optimization mode
   - Guidance scale
   - Number of inference steps
   - Max sequence length
   - Random seed (optional)

5. Click "Generate Image" to create your image

6. Download the generated image using the "Download Image" button

## Optimization Modes

### Standard Mode
- Maximum dimension: 1024px
- Default dimension: 768px
- Higher quality output
- Requires more VRAM

### Memory Efficient Mode
- Maximum dimension: 768px
- Default dimension: 512px
- Reduced VRAM usage
- Slightly lower quality output
- Faster generation times

## Memory Optimization Features

The application includes several memory optimization techniques:
- Sequential CPU offload
- Attention slicing
- Inference mode
- Automatic memory cleanup
- VRAM monitoring

## Credits

- Model: [shuttleai/shuttle-jaguar](https://huggingface.co/shuttleai/shuttle-jaguar)
- Framework: [Streamlit](https://streamlit.io/)
- Diffusion Pipeline: [Hugging Face Diffusers](https://huggingface.co/docs/diffusers/)
