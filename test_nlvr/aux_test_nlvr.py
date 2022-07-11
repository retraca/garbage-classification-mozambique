from transformers import ViltProcessor, ViltForImagesAndTextClassification, AutoTokenizer, AutoConfig
import torch
from PIL import Image
from numpy import asarray

model_path = "dandelin/vilt-b32-finetuned-nlvr2"
#config = AutoConfig.from_pretrained(model_path,  output_hidden_states=True, output_attentions=True)
processor = ViltProcessor.from_pretrained(model_path)
model = ViltForImagesAndTextClassification.from_pretrained(model_path)

""" image1= torch.hub.download_url_to_file('https://lil.nlp.cornell.edu/nlvr/exs/ex0_0.jpg', 'image1.jpg')
image2= torch.hub.download_url_to_file('https://lil.nlp.cornell.edu/nlvr/exs/ex0_1.jpg', 'image2.jpg') """

img = Image.open('image1.jpg')
image1 = asarray(img)
img = Image.open('image2.jpg')
image2 = asarray(img)

text = "The left image contains more garbage than the right image"

# prepare inputs
encoding = processor([image1, image2], text, return_tensors="pt")


# forward pass
with torch.no_grad():
    outputs = model(input_ids=encoding.input_ids, pixel_values=encoding.pixel_values.unsqueeze(0))
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=1)

    output = dict()
    
    
for label, id in model.config.label2id.items():
    output[label] = probs[:,id].item()
    
print(output)
