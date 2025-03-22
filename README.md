![dzfgdf.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/gFcXjzt_OA-46WpFfz-9L.png)

# **Alphabet-Sign-Language-Detection**
> **Alphabet-Sign-Language-Detection** is an image classification vision-language encoder model fine-tuned from **google/siglip2-base-patch16-224** for a single-label classification task. It is designed to classify images into **sign language alphabet** categories using the **SiglipForImageClassification** architecture.  

```py
Classification Report:
              precision    recall  f1-score   support

           A     0.9995    1.0000    0.9998      4384
           B     1.0000    1.0000    1.0000      4441
           C     1.0000    1.0000    1.0000      3993
           D     1.0000    0.9998    0.9999      4940
           E     1.0000    1.0000    1.0000      4658
           F     1.0000    1.0000    1.0000      5750
           G     0.9992    0.9996    0.9994      4978
           H     1.0000    0.9979    0.9990      4807
           I     0.9992    1.0000    0.9996      4856
           J     1.0000    0.9996    0.9998      5227
           K     0.9972    1.0000    0.9986      5426
           L     1.0000    0.9998    0.9999      5089
           M     1.0000    0.9964    0.9982      3328
           N     0.9955    1.0000    0.9977      2635
           O     0.9998    1.0000    0.9999      4564
           P     1.0000    0.9993    0.9996      4100
           Q     1.0000    1.0000    1.0000      4187
           R     0.9998    0.9984    0.9991      5122
           S     0.9998    0.9998    0.9998      5147
           T     1.0000    1.0000    1.0000      4722
           U     0.9984    0.9998    0.9991      5041
           V     1.0000    0.9984    0.9992      5116
           W     0.9998    1.0000    0.9999      4926
           X     1.0000    0.9995    0.9998      4387
           Y     1.0000    1.0000    1.0000      5185
           Z     0.9996    1.0000    0.9998      4760

    accuracy                         0.9996    121769
   macro avg     0.9995    0.9996    0.9995    121769
weighted avg     0.9996    0.9996    0.9996    121769
```
![demo.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/AVpi4xPsVq6PV9NzonHoi.png)

The model categorizes images into the following 26 classes:  
- **Class 0:** "A"  
- **Class 1:** "B"  
- **Class 2:** "C"  
- **Class 3:** "D"  
- **Class 4:** "E"  
- **Class 5:** "F"  
- **Class 6:** "G"  
- **Class 7:** "H"  
- **Class 8:** "I"  
- **Class 9:** "J"  
- **Class 10:** "K"  
- **Class 11:** "L"  
- **Class 12:** "M"  
- **Class 13:** "N"  
- **Class 14:** "O"  
- **Class 15:** "P"  
- **Class 16:** "Q"  
- **Class 17:** "R"  
- **Class 18:** "S"  
- **Class 19:** "T"  
- **Class 20:** "U"  
- **Class 21:** "V"  
- **Class 22:** "W"  
- **Class 23:** "X"  
- **Class 24:** "Y"  
- **Class 25:** "Z"  

# **Run with Transformers🤗**

```python
!pip install -q transformers torch pillow gradio
```

```python
import gradio as gr
from transformers import AutoImageProcessor
from transformers import SiglipForImageClassification
from transformers.image_utils import load_image
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/Alphabet-Sign-Language-Detection"
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

def sign_language_classification(image):
    """Predicts sign language alphabet category for an image."""
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    labels = {
        "0": "A", "1": "B", "2": "C", "3": "D", "4": "E", "5": "F", "6": "G", "7": "H", "8": "I", "9": "J",
        "10": "K", "11": "L", "12": "M", "13": "N", "14": "O", "15": "P", "16": "Q", "17": "R", "18": "S", "19": "T",
        "20": "U", "21": "V", "22": "W", "23": "X", "24": "Y", "25": "Z"
    }
    predictions = {labels[str(i)]: round(probs[i], 3) for i in range(len(probs))}
    
    return predictions

# Create Gradio interface
iface = gr.Interface(
    fn=sign_language_classification,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(label="Prediction Scores"),
    title="Alphabet Sign Language Detection",
    description="Upload an image to classify it into one of the 26 sign language alphabet categories."
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
```

# **Intended Use:**  

The **Alphabet-Sign-Language-Detection** model is designed for sign language image classification. It helps categorize images of hand signs into predefined alphabet categories. Potential use cases include:  

- **Sign Language Education:** Assisting learners in recognizing and practicing sign language alphabets.  
- **Accessibility Enhancement:** Supporting applications that improve communication for the hearing impaired.  
- **AI Research:** Advancing computer vision models in sign language recognition.  
- **Gesture Recognition Systems:** Enabling interactive applications with real-time sign language detection.
