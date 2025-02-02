from flask import Blueprint, request, jsonify, render_template
import openai
import requests
import json

import os
from dotenv import load_dotenv



# Load environment variables from .env file
load_dotenv()

# Use the API key securely
openai.api_key = os.getenv("OPENAI_API_KEY")

routes = Blueprint('routes', __name__)

@routes.route('/')
def chatbot():
    return render_template('chatbot.html')

@routes.route('/recommendDoctor', methods=['POST'])
def recommend_doctor():
    # Get user input
    data = request.json
    user_input = data.get('problem', '').strip().lower()  # Normalize input

    print("User Input:", user_input)
    print("____________________________________")

    # Step 1: Read doctor data from JSON file instead of API
    json_file_path = os.path.join(os.getcwd(), 'app\doctors.json')  # Ensure correct path
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            doctor_data = json.load(file)
    except FileNotFoundError:
        print("Error: doctors.json file not found")
        return jsonify({'message': 'Error: doctors.json file not found'}), 500
    except json.JSONDecodeError:
        return jsonify({'message': 'Error: Invalid JSON format in doctors.json'}), 500

    # Preprocess and normalize doctor data
    processed_doctors = []
    for doctor in doctor_data:
        processed_doctors.append({
            "name": doctor.get("name", "").strip(),
            "information": doctor.get("info", "").strip().lower(),
            "web_page": doctor.get("web_page", "").strip(),
            "department": {
                "name": doctor.get("department", {}).get("name", "").strip().lower(),
                "conditions": [cond.lower() for cond in doctor.get("department", {}).get("conditions", [])],
                "services": [service.lower() for service in doctor.get("department", {}).get("services", [])],
                "definition": doctor.get("department", {}).get("info", "").strip().lower()
            },
            "areas_of_expertise": [skill.get("name", "").strip().lower() for skill in doctor.get("skills", [])],
            "licenses": [license.get("name", "").strip().lower() for license in doctor.get("licences", [])]
        })

    print("Processed Doctor Data:", processed_doctors)
    print("___________________________________________________________________")
    
    # Step 2: Attempt to match using AI model
    try:
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


        ai_response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"User's problem: {user_input}\nDoctors data: {processed_doctors}"}
            ]
        )
        ai_response_text = ai_response['choices'][0]['message']['content'].strip()

        print(f"AI Response (Raw): {ai_response_text}")
        print("_____________________________________________________________________________")

        # Validate and clean the AI response
        if ai_response_text.startswith("```") and ai_response_text.endswith("```"):
            # Remove code block formatting if present
            ai_response_text = ai_response_text.strip("```").strip("json")

        print(f"AI Response (Sanitized): {ai_response_text}")

        # Parse the AI response
        recommendations = json.loads(ai_response_text)

        # Step 3: Sort the recommendations
        user_problem_tokens = set(user_input.split())  # Tokenize user problem for comparison

        # Calculate relevance based on expertise, department services, and conditions
        def calculate_relevance(doctor):
            expertise_score = sum(1 for token in user_problem_tokens if token in doctor.get("areas_of_expertise", "").lower())
            department_name = doctor.get("department", "").lower()
            condition_score = sum(1 for token in user_problem_tokens if token in department_name)
            service_score = 0  # Assuming no services are embedded in department as strings (adjust if necessary)
            if doctor.get("department", ""):
                department_services = doctor.get("department", {}).get("services", [])
                service_score = sum(1 for token in user_problem_tokens if any(token in service for service in department_services))

            # Higher weight for expertise and matching conditions/services
            return (2 * expertise_score) + condition_score + service_score

        # Custom sorting to prioritize General Medicine and doctors solving more problems
        def custom_sort(doctor):
            is_general_medicine = "general medicine" in doctor.get("department", "").lower()
            relevance_score = calculate_relevance(doctor)
            num_issues_solved = len(doctor.get("department", {}).get("conditions", [])) + len(doctor.get("department", {}).get("services", []))

            # Prioritize General Medicine, then relevance, then number of issues solved
            return (not is_general_medicine, -relevance_score, -num_issues_solved)
            
        # Apply the sorting
        '''recommendations['recommendations'] = sorted(
            recommendations['recommendations'],
            key=custom_sort
        )'''
        # Return sorted recommendations
        return jsonify(recommendations)

    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return jsonify({'message': 'Error processing AI response. Invalid JSON.', 'error': str(e)}), 500
    except Exception as e:
        print(f"Error during AI processing:", e)
        return jsonify({'message': 'Error during AI processing.', 'error': str(e)}), 500

    # If no recommendations are found after retrying
    return jsonify({'message': 'No suitable doctors found for the input problem. Please refine your input.'})


def filter_relevant_doctors(user_input, doctors):
    user_tokens = set(user_input.lower().split())  # Convert user input to lowercase tokens
    
    scored_doctors = []

    for doctor in doctors:
        # Extract relevant fields
        expertise = set(doctor.get("areas_of_expertise", []))
        department_name = doctor.get("department", {}).get("name", "").lower()
        conditions = set(doctor.get("department", {}).get("conditions", []))
        services = set(doctor.get("department", {}).get("services", []))
        
        # Compute matches
        expertise_matches = len(user_tokens & expertise)  # Common words in expertise
        department_match = int(any(token in department_name for token in user_tokens))
        condition_match = len(user_tokens & conditions)  # Common conditions
        service_match = len(user_tokens & services)  # Common services

        # Calculate a match score
        match_score = expertise_matches + department_match + condition_match + service_match
        
        if match_score > 0:
            doctor["match_score"] = match_score
            scored_doctors.append(doctor)

    # Sort doctors by match score (highest score first)
    scored_doctors = sorted(scored_doctors, key=lambda x: x["match_score"], reverse=True)

    return scored_doctors[:4]  # Send top 4 doctors 