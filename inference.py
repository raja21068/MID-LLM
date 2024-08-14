import torch
from PIL import Image
from torchvision import transforms
import openai
from ipfs_blockchain import get_from_ipfs
from models import BrainTumorClassifier

def predict_and_generate_report(model, image_path, global_model_hash):
    # Load the global model parameters
    serialized_params = get_from_ipfs(global_model_hash)
    global_model_params = json.loads(serialized_params)
    model.load_state_dict({k: torch.tensor(v) for k, v in global_model_params.items()})

    # Preprocessing the image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Model inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        classification_result = 'Tumor' if predicted.item() == 1 else 'No Tumor'

    # OpenAI API call to generate a report
    prompt = f"A brain MRI image has been classified as: {classification_result}. Please provide a detailed report on the implications of this classification."
    
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=250
    )
    
    report = response.choices[0].text.strip()

    print(f"Classification Result: {classification_result}")
    print(f"Generated Report: {report}")

    # Optionally, save the report to a file
    with open("classification_report.txt", "w") as report_file:
        report_file.write(f"Classification Result: {classification_result}\n\n")
        report_file.write("Generated Report:\n")
        report_file.write(report)
