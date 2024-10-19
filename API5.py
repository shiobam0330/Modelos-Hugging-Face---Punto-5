import streamlit as st
from utils import primer_modelo, segundo_modelo, proc_im_clas
from PIL import Image

# Cargar los modelos
primer_modelo()
segundo_modelo()

col1, col2 = st.columns(2)

with col1:
    st.title("Generación de imágenes con IA")
    peticion = st.text_area("Solicitud de imagen", "Funciona mejor si es en inglés")
    
    generar_imagen = None
    if st.button("Generar"):
        try:
            pipe = st.session_state["modelo1"]
            generar_imagen = pipe(peticion).images[0]
            st.image(generar_imagen, caption="Imagen generada", use_column_width=True)
            st.session_state["generar_imagen"] = generar_imagen
        except Exception as e:
            st.error(f"Ocurrió un error: {str(e)}")
    
    st.subheader("Clasificación de la Imagen Generada")
    if "generar_imagen" in st.session_state:
        if st.button("Clasificar"):
            try:
                image = st.session_state["generar_imagen"]
                resultado = proc_im_clas(image)
                st.success(f"La imagen ha sido clasificada como: {resultado}")
            except Exception as e:
                st.error(f"Ocurrió un error durante la clasificación: {str(e)}")
    else:
        st.warning("Genera una imagen primero para poder clasificarla.")

with col2:
    st.title("Clasificación de una imagen")
    archivo = st.file_uploader("Cargar imagen", type=["png", "jpg", "jpeg"])
    
    if archivo is not None:
        if st.button("Clasificar Imagen"):
            try:
                image = Image.open(archivo)
                st.image(image, caption="Imagen cargada", use_column_width=True)
                resultado = proc_im_clas(image)
                st.success(f"La imagen ha sido clasificada como: {resultado}")
            except Exception as e:
                st.error(f"Ocurrió un error durante la clasificación: {str(e)}")
    else:
        st.warning("Por favor, carga una imagen.")
