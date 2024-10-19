import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoImageProcessor, ResNetForImageClassification
from PIL import Image

def primer_modelo():
    if "modelo1" not in st.session_state:
        #Carpeta del modelo local
        model_id = "D:/Downloads/INTELIGENCIA ARTIFICIAL/PARCIAL 2/Punto 5/stable_diffusion_model"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        st.session_state["modelo1"] = pipe

def segundo_modelo():
    if "modelo2" not in st.session_state:
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        st.session_state["modelo2"] = (processor, model)

import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoImageProcessor, ResNetForImageClassification

def primer_modelo():
    """Carga el modelo de generación de imágenes (Stable Diffusion) y lo almacena en session_state."""
    if "modelo1" not in st.session_state:
        model_id = "D:/Downloads/INTELIGENCIA ARTIFICIAL/PARCIAL 2/Punto 5/stable_diffusion_model"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Carga el modelo de generación de imágenes (Stable Diffusion)
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        pipe = pipe.to(device)
        
        # Guarda el modelo en la sesión
        st.session_state["modelo1"] = pipe

def segundo_modelo():
    """Carga el modelo de clasificación de imágenes (ResNet-50) y lo almacena en session_state."""
    if "modelo2" not in st.session_state:
        # Carga el procesador y el modelo de clasificación de imágenes
        processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        
        # Guarda el procesador y el modelo en la sesión
        st.session_state["modelo2"] = (processor, model)

def proc_im_clas(image):
    try:
        processor, model = st.session_state["modelo2"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_label = logits.argmax(-1).item()
        return model.config.id2label[predicted_label]
    except Exception as e:
        raise RuntimeError(f"Error al clasificar la imagen: {str(e)}")
