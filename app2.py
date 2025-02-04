import streamlit as st
import torch
from diffusers import DiffusionPipeline
from PIL import Image
import os
import gc

# Set page config
st.set_page_config(page_title="shuttleai/shuttle-jaguar", layout="wide")

# Add title and description
st.title("shuttleai/shuttle-jaguar")
st.markdown("Generate images from text descriptions using the Shuttle Jaguar model.")

# Initialize session state for the model
if 'pipe' not in st.session_state:
    @st.cache_resource
    def load_model():
        # Load the model with optimizations
        pipe = DiffusionPipeline.from_pretrained(
            "shuttleai/shuttle-jaguar",
            torch_dtype=torch.bfloat16,
            use_safetensors=True
        )
        
        # 啟用記憶體優化
        pipe.enable_sequential_cpu_offload()  # 更穩定的 CPU 卸載方式
        pipe.enable_attention_slicing()  # 注意力切片，不指定切片大小
        
        return pipe
    
    with st.spinner("Loading model... This might take a while."):
        st.session_state.pipe = load_model()

# Create sidebar for model settings
st.sidebar.header("Generation Settings")

# Add optimization mode option
optimization_mode = st.sidebar.radio(
    "Optimization Mode",
    ["Standard", "Memory Efficient"],
    help="Memory Efficient mode uses less VRAM but might be slower"
)

# Input for prompt
prompt = st.text_area("Enter your prompt:", 
                     value="A cat holding a sign that says hello world",
                     height=100)

# Create two columns for dimension inputs
col1, col2 = st.columns(2)

# 根據優化模式調整尺寸
max_dim = 768 if optimization_mode == "Memory Efficient" else 1024
default_dim = 512 if optimization_mode == "Memory Efficient" else 768

with col1:
    width = st.number_input("Image Width", 
                           min_value=256, 
                           max_value=max_dim, 
                           value=default_dim, 
                           step=128)

with col2:
    height = st.number_input("Image Height", 
                            min_value=256, 
                            max_value=max_dim, 
                            value=default_dim, 
                            step=128)

# Advanced settings in sidebar
guidance_scale = st.sidebar.slider("Guidance Scale", 
                                 min_value=1.0, 
                                 max_value=20.0, 
                                 value=3.5, 
                                 step=0.5)

# 根據優化模式調整步數
default_steps = 3 if optimization_mode == "Memory Efficient" else 4
num_inference_steps = st.sidebar.slider("Number of Inference Steps", 
                                      min_value=1, 
                                      max_value=100, 
                                      value=default_steps)

max_sequence_length = st.sidebar.number_input("Max Sequence Length",
                                            min_value=64,
                                            max_value=512,
                                            value=256)

# Add seed option
use_seed = st.sidebar.checkbox("Use Random Seed")
if use_seed:
    seed = st.sidebar.number_input("Seed", value=42, step=1)

# Display current VRAM usage if available
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    current_vram = torch.cuda.memory_allocated() / 1024**3
    st.sidebar.markdown(f"Current VRAM Usage: {current_vram:.2f} GB")

# Generate button
if st.button("Generate Image"):
    try:
        with st.spinner("Generating image..."):
            # 清理記憶體
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Set up generator if using seed
            generator = None
            if use_seed:
                generator = torch.Generator("cpu").manual_seed(seed)
            
            # Generate image
            with torch.inference_mode():
                image = st.session_state.pipe(
                    prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=max_sequence_length,
                    generator=generator
                ).images[0]
            
            # Display the generated image
            #st.image(image, caption="Generated Image", use_container_width=True)
            st.image(image, caption="Generated Image")
            # Add download button
            col1, col2 = st.columns(2)
            with col1:
                # Save image temporarily
                temp_path = "generated_image.png"
                image.save(temp_path)
                
                # Create download button
                with open(temp_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Image",
                        data=file,
                        file_name="generated_image.png",
                        mime="image/png"
                    )
                
                # Clean up temporary file
                os.remove(temp_path)
            
            # 顯示生成後的 VRAM 使用量
            if torch.cuda.is_available():
                current_vram = torch.cuda.memory_allocated() / 1024**3
                st.sidebar.markdown(f"VRAM Usage After Generation: {current_vram:.2f} GB")
                
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Try using Memory Efficient mode or reducing image dimensions if you're experiencing memory issues.")

# Add information about the model and optimizations
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
### Model Information
- Model: Shuttle Jaguar
- Provider: ShuttleAI
- Current Mode: {optimization_mode}
- Optimizations Enabled:
  - Sequential CPU Offload
  - Attention Slicing
  - Inference Mode
  - Memory Cleanup
""")

# Add usage instructions
st.sidebar.markdown("""
### Usage Tips
1. Use Memory Efficient mode if experiencing VRAM issues
2. Reduce dimensions for faster generation
3. Lower inference steps for speed
4. Clear VRAM between generations if needed
""")