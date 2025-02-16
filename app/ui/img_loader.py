import streamlit as st
import base64


def render_image(filepath: str, width=200):
   if filepath in st.session_state:
        st.image(st.session_state[filepath],width=width)
   else:     
      mime_type = filepath.split('.')[-1:][0].lower()
      with open('img/'+filepath, "rb") as f:
         content_bytes = f.read()
         content_b64encoded = base64.b64encode(content_bytes).decode()
         image_string = f'data:image/{mime_type};base64,{content_b64encoded}'
         st.image(image_string,width=width)
         st.session_state[filepath]=image_string