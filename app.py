import gradio as gr
import os
from core.predict import ImageClassifier
from PIL import Image

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, 'model', 'cnn_128_model-100.pth')
class_name = {0: 'Cat', 1: 'Dog', 2: 'Person'} 
classifier = ImageClassifier(model_path=model_path, class_name=class_name)

def classify_image(image):
    if image is None:
        return None, None
    image_path = os.path.join(base_dir, 'uploaded_image.jpg')
    image.save(image_path)

    label, output_path = classifier.predict(image_path)
    
    return label, Image.open(output_path)

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type='pil'),
    outputs=[gr.Textbox(label='Prediction'), gr.Image(label='Labeled Image')],
    title='Image Classification Gradio App',
    description='Upload an image to classify image of Cat, Dog, Person'
)

if __name__ == '__main__':
    demo.launch()

