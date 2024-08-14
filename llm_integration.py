def generate_brain_tumor_prompt(description, patient_info, model_output):
    """Generate a refined prompt for brain tumor diagnosis and treatment."""
    
    prompt = f"""
    Patient Information:
    {patient_info}
    
    Diagnostic Imaging Analysis:
    {description}

    Based on the imaging analysis of the brain, the model detected the following tumor characteristics:
    - Tumor size: {model_output['size']}
    - Tumor location: {model_output['location']}
    - Tumor type (if classified): {model_output.get('type', 'Unknown')}
    - Confidence score: {model_output['confidence']}

    Given these details, please provide a detailed diagnosis including:
    1. The likely type of tumor and its implications.
    2. Recommended treatment options considering the tumor's characteristics, the patient's medical history, and potential side effects.
    3. Potential outcomes of each treatment option, including any risk factors or complications.
    4. Suggested follow-up actions, including additional diagnostic tests, monitoring strategies, and long-term care recommendations.
    """
    
    return prompt

def predict_diagnosis_with_prompt_engineering(model, image_path, global_model_hash, patient_info):
    """Uses the LLM to generate a diagnosis or prediction based on a refined prompt."""
    try:
        # Preprocess the image
        image = preprocess_image(image_path)
        
        # Run the inference
        model_output = predict_and_generate_report(model, image, global_model_hash)
        
        # Process the model's output into a textual description
        description = process_model_output(model_output)
        
        # Generate a refined prompt using prompt engineering
        refined_prompt = generate_brain_tumor_prompt(description, patient_info, model_output)
        
        # Generate the diagnosis using the LLM with the refined prompt
        logger.info("Generating diagnosis using the LLM with refined prompt.")
        diagnosis = llm_pipeline(refined_prompt)
        
        return diagnosis[0]['generated_text']
    
    except Exception as e:
        logger.error(f"Error during diagnosis prediction: {e}")
        return "An error occurred during diagnosis prediction."

if __name__ == "__main__":
    # Example usage with patient information and refined prompt engineering
    brain_tumor_model = BrainTumorClassifier()
    image_path = "/path/to/your/image.nii.gz"  # Replace with your actual image path
    global_model_hash = "QmHash"  # Replace with your actual global model hash
    patient_info = "Age: 45, Gender: Male, Previous medical history: Hypertension, Symptoms: Headaches, dizziness"
    
    diagnosis = predict_diagnosis_with_prompt_engineering(brain_tumor_model, image_path, global_model_hash, patient_info)
    print(f"Generated Diagnosis: {diagnosis}")
