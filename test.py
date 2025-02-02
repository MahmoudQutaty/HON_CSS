


from flask import Blueprint, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
import json
import re

# Initialize Flask Blueprint
routes = Blueprint('routes', __name__)

# Load tokenizer and model
print("Starting model loading sequence...")
print("Loading tokenizer for a smaller model...")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
print("Tokenizer loaded successfully for distilgpt2.")

print("Loading model for a smaller model...")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
print("Model loaded successfully for distilgpt2.")

@routes.route('/')
def chatbot():
    return render_template('chatbot.html')

def preprocess_doctors(doctor_data):
    """
    Preprocess doctor data to create a combined description for each doctor.
    """
    processed_doctors = []
    for doctor in doctor_data:
        processed_doctors.append({
            "name": doctor.get("name", "").strip(),
            "information": doctor.get("info", "").strip().lower(),
            "web_page": doctor.get("address", "").strip(),
            "department": {
                "name": doctor.get("department", {}).get("name", "").strip().lower(),
                "conditions": [cond.lower() for cond in doctor.get("department", {}).get("conditions", [])],
                "services": [service.lower() for service in doctor.get("department", {}).get("services", [])],
                "definition": doctor.get("department", {}).get("info", "").strip().lower()
            },
            "areas_of_expertise": [skill.get("name", "").strip().lower() for skill in doctor.get("skills", [])],
            "licenses": [license.get("name", "").strip().lower() for license in doctor.get("licences", [])]
        })
        
    return processed_doctors

def generate_recommendations(user_input, processed_doctors, max_recommendations=4):
    """
    Generate smart recommendations using DistilGPT-2.
    """
    # Limit doctor descriptions to avoid exceeding model token limit
    doctors_limited = processed_doctors[:5]  # Reduce doctor data size to fit model limits

    # Simplify the prompt to generate plain text
    prompt = (
            "You are a helpful medical assistant. Based on the user's input and detailed doctors' data "
            "(including areas of expertise, licenses, specialized conditions, department services, and department name), "
            "recommend up to four doctors who are the best fit to address the user's problem. "
            "Ensure the recommendations are ranked by their ability to solve the problem based on: "
            "1. Area of expertise (highest priority), 2. Department services, conditions and definition. "
            "Also, prioritize doctors who can solve more problems. "
            "Response format: JSON with the following structure: "
            "{ "
            "  'recommendations': [ "
            "    { "
            "      'name': <Doctor's name>, "
            "      'web_page': <Doctor's profile or relevant webpage>, "
            "      'department': <Department name>, "
            "      'reason': <Specific reason for the recommendation, matching expertise and services to the user's problem>, "
            "      'definition': <Brief explanation of the department and how it addresses the problem> "
            "    }, "
            "    ... (recommendations between 1 to 4 maximum. ) "
            "  ] "
            "} "
        )
    # Tokenize input (ensure truncation)
    inputs = tokenizer(prompt, return_tensors="pt", max_length=400, truncation=True)

    # Generate response with controlled new tokens
    outputs = model.generate(
        **inputs, 
        max_new_tokens=200, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id  # Explicitly set pad_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

    # Parse the plain text response into JSON
    recommendations = {"recommendations": []}
    try:
        # Use regex to extract doctor recommendations
        doctor_entries = re.findall(
            r"- Name: (.*?)\n- Web Page: (.*?)\n- Department: (.*?)\n- Reason: (.*?)\n- Definition: (.*?)(?:\n|$)",
            response,
            re.DOTALL
        )

        for entry in doctor_entries:
            name, web_page, department, reason, definition = entry
            recommendations["recommendations"].append({
                "name": name.strip(),
                "web_page": web_page.strip(),
                "department": department.strip(),
                "reason": reason.strip(),
                "definition": definition.strip()
            })

        # Limit recommendations
        recommendations["recommendations"] = recommendations["recommendations"][:max_recommendations]

    except Exception as e:
        print(f"Error parsing model response: {e}")
        return {"recommendations": []}

    return recommendations

@routes.route('/recommendDoctor', methods=['POST'])
def recommend_doctor():
    # Get user input
    data = request.json
    user_input = data.get('problem', '').strip().lower()

    # Fetch doctor data from Spring Boot API
    try:
        spring_boot_url = "http://localhost:8080/doctor"
        response = requests.get(spring_boot_url)
        response.raise_for_status()
        doctor_data = response.json()
    except requests.exceptions.RequestException as e:
        return jsonify({'message': 'Error fetching data from Spring Boot API.', 'error': str(e)}), 500

    # Preprocess doctor data
    processed_doctors = preprocess_doctors(doctor_data)

    # Generate recommendations
    recommendations = generate_recommendations(user_input, processed_doctors)

    # If no recommendations are found, return a message
    if not recommendations["recommendations"]:
        return jsonify({'message': 'No suitable doctors found for the input problem. Please refine your input.'})

    return jsonify(recommendations)