import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn as nn
from torchvision import models
import streamlit as st
from transformers import pipeline

# -------------------------------
# Leaf Model Loader
# -------------------------------
@st.cache_resource
def load_leaf_model():
    device = torch.device("cpu")  # Use CPU; change to "cuda" if GPU is available

    # 1. Define architecture
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)  # 10 classes

    # 2. Load the checkpoint safely
    try:
        checkpoint = torch.load("best_model.pth", map_location=device)
        
        # 3. Load weights
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            st.error("Checkpoint format not recognized.")
            st.stop()
        
        model.eval()
        model.to(device)
        return model

    except FileNotFoundError:
        st.error("Model file 'best_model.pth' not found.")
        st.stop()
    except Exception as e:
        st.error(f"Critical Error loading 'best_model.pth': {e}")
        st.stop()


# -------------------------------
# Expert Model Loader
# -------------------------------
@st.cache_resource
def load_expert_model():
    try:
        return pipeline("text-generation", model="gpt2")
    except Exception as e:
        st.error(f"Error loading GPT-2 model: {e}")
        st.stop()

st.header("Leaf Diseases Classification")

class_names = {0 : 'Tomato_Bacterial_spot', 1: 'Tomato_Early_blight', 2: 'Tomato_Late_blight' , 3: 'Tomato_Leaf_Mold',
               4: 'Tomato_Septoria_leaf_spot', 5: 'Tomato_Spider_mites_Two_spotted_spider_mite', 6 : 'Tomato__Target_Spot',
               7 : 'Tomato__Tomato_YellowLeaf__Curl_Virus', 8 : 'Tomato__Tomato_mosaic_virus', 9 : 'Tomato_healthy'}

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    if st.button('Identify Image'):
        img_tensor = preprocess(image).unsqueeze(0) 
        with torch.no_grad():
            output = load_leaf_model(img_tensor)
            # Softmax on the class dimension
            probs = torch.nn.functional.softmax(output, dim=1)[0]
            confidence, index = torch.max(probs, dim=0)
            
            label = class_names[index.item()]
            
        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence.item()*100:.2f}%")
        st.progress(float(confidence.item()))