import streamlit as st
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms

device = torch.device("cpu")

leaf_model = models.resnet50(pretrained=False)
leaf_model.fc = nn.Linear(leaf_model.fc.in_features, 10)

checkpoint = torch.load("best_model.pth", map_location=device)
st.write(print(checkpoint['model_state_dict'].keys()))
leaf_model.load_state_dict(checkpoint["model_state_dict"])

leaf_model.eval()


# leaf_model = torch.load('best_model.py', map_location=torch.device('cpu'), weights_only=False)
# leaf_model.eval()

@st.cache_resource 
def load_expert_model():
    return pipeline(
        "text-generation",
        model="gpt2",
        torch_dtype='auto'
    )

class_names = {0 : 'Tomato_Bacterial_spot', 1: 'Tomato_Early_blight', 2: 'Tomato_Late_blight' , 3: 'Tomato_Leaf_Mold',
               4: 'Tomato_Septoria_leaf_spot', 5: 'Tomato_Spider_mites_Two_spotted_spider_mite', 6 : 'Tomato__Target_Spot',
               7 : 'Tomato__Tomato_YellowLeaf__Curl_Virus', 8 : 'Tomato__Tomato_mosaic_virus', 9 : 'Tomato_healthy'}

# uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
        
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        
        if st.button('Identify Image'):
            
            # with st.spinner('Analyzing...'):
            img_tensor = preprocess(image).unsqueeze(0) 
            with torch.no_grad():
                output = leaf_model(img_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                confidence, index = torch.max(probabilities, dim=0)
                label = class_names[index.item()]
                    
                percent = confidence.item() * 100
                
                st.success(f"**Prediction:** {label}")
                st.info(f"**Confidence:** {percent:.2f}%")
                
                
                st.progress(confidence.item())

# model_name = "gpt2"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# generator = pipeline (
#     "text-generation",
#     model = model_name,
#     tokenizer = model_name,
#     torch_dtype = 'auto'
# )

# def leaf_disease(leaf):
#     prompt = f"""
#     ### Instruction:
#     You are a professional plant pathologist. A deep learning model has identified a leaf disease in a crop. 
#     Provide a concise, expert report for a farmer.

#     ### Disease Detected: 
#     {label}

#     ### Report Requirements:
#     1. Symptoms: Briefly describe how to confirm this.
#     2. Immediate Action: What should the farmer do today?
#     3. Organic Treatment: List two natural remedies.
#     4. Prevention: How to avoid this in the next season.

#     ### Expert Advice:
#     """

#     max_token = 300
#     outputs = generator(prompt, max_new_tokens=max_token)

#     return(outputs[0]["generated_text"])

  