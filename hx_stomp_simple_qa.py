import json
from typing import Dict, List, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests

class HXStompSimpleQA:
    def __init__(self):
        # Load the pedals and units data from JSON
        with open('data/hx_pedals.json', 'r') as f:
            self.pedals_data = json.load(f)
        with open('data/hx_pedals_units.json', 'r') as f:
            self.units_data = json.load(f)
            
        # Create parameter info lookup
        self.parameter_info = {}
        for param_data in self.units_data:
            for param_name, details in param_data.items():
                self.parameter_info[param_name] = details
            
        # Load and process all knowledge sources
        self.knowledge_base = self.load_knowledge_base()
        self.recipes = []

    def format_pedal_params(self, params: List[Dict]) -> List[Dict]:
        """Format pedal parameters with their units and ranges"""
        formatted_params = []
        for param in params:
            if isinstance(param, dict):
                param_name = list(param.keys())[0] if param else None
                if param_name and param_name in self.parameter_info:                    
                    formatted_param = {
                        param_name: "Default"
                    }
                    formatted_params.append(formatted_param)
        return formatted_params

    def load_knowledge_base(self) -> List[str]:
        """Load all knowledge sources"""
        knowledge = []
        
        # Process pedals data into a flat list of pedals with parameters
        self.pedals_info = []
        for category in self.pedals_data:
            if 'subcategories' in category:
                for subcat in category['subcategories']:
                    if 'models' in subcat:
                        for model in subcat['models']:
                            if isinstance(model, dict) and not model.get('use_subcategory'):
                                if model.get('name') and model.get('params') is not None:
                                    pedal_info = {
                                        "name": model['name'],
                                        "category": category['name'],
                                        "subcategory": subcat['name'],
                                        "params": self.format_pedal_params(model['params'])
                                    }
                                    self.pedals_info.append(pedal_info)
                                    
                                    # Add pedal info to knowledge base
                                    desc = f"Q: What are the settings for {model['name']}?\n"
                                    desc += f"A: {model['name']} from {category['name']} > {subcat['name']}\n"
                                    if model.get('params'):
                                        desc += "Parameters: " + ", ".join(f"{list(p.keys())[0]}" for p in model['params'] if isinstance(p, dict))
                                    knowledge.append(desc)
        
        # Load additional knowledge sources
        knowledge.extend(self._load_qa_data('data/hx_manual_qa_data.csv'))
        knowledge.extend(self._load_qa_data('data/hx_pedal_order_qa_data.csv'))
        
        return knowledge

    def _load_qa_data(self, filepath: str) -> List[str]:
        """Helper function to load Q&A data from CSV files"""
        try:
            # Read file content first to handle different line endings and encodings
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().replace('\r\n', '\n')
            
            # Split into lines and process
            lines = content.split('\n')
            qa_pairs = []
            
            # Skip header
            for i in range(1, len(lines)):
                line = lines[i].strip()
                if not line:  # Skip empty lines
                    continue
                    
                # Split on semicolon, but keep semicolons within quotes
                parts = []
                current = ''
                in_quotes = False
                for char in line:
                    if char == '"':
                        in_quotes = not in_quotes
                    elif char == ';' and not in_quotes:
                        parts.append(current.strip('"'))
                        current = ''
                    else:
                        current += char
                if current:
                    parts.append(current.strip('"'))
                
                if len(parts) >= 2:
                    question = parts[0].strip()
                    answer = parts[1].strip()
                    if question and answer:  # Only add if both question and answer exist
                        qa_pairs.append(f"Q: {question}\nA: {answer}")
            
            return qa_pairs
        except Exception as e:
            print(f"Error loading data from {filepath}: {str(e)}")
            return []
    
    def generate_recipe(self, question: str, reset_context: bool) -> Dict:
        try:
            # Enhanced prompt with examples and style context
            prompt = f"""You are a Line 6 HX Stomp expert. Generate a unique block chain recipe based on the specific question.

            Rules:
            - Response must follow this format:
            Suggestion: [brief description of the tone]
            Chain: [block1] (param1: value1, param2: value2) > [block2] (param1: value1, param2: value2) > [block3]...

            - Maximum 8 blocks in the chain
            - Each block must be a real HX Stomp effect/amp/cab
            - Parameters must use proper units (dB, ms, Hz, %)
            - Effects must be separated by ">" symbol

            Examples of different styles:
            1. Blues: "Suggestion: Classic blues tone with warm overdrive
            Chain: Teemah! (Drive: 65%, Tone: 55%) > US Double Nrm (Drive: 4.5, Bass: 6.0, Mid: 7.0) > Room Verb (Decay: 2.5s)"

            2. Ambient: "Suggestion: Ethereal ambient wash with modulation
            Chain: Poly Sustain > Mod Echo (Time: 750ms, Feedback: 45%) > Particle Verb (Decay: 8s, Mix: 65%)"

            Question: {question}
            Remember to tailor the response specifically to this question and avoid generic responses.
            Answer:"""

            # Modified API call with better context handling
            response = requests.post('http://localhost:11434/api/generate', 
                json={
                    "model": "tinyllama",
                    "prompt": prompt,
                    "stream": False,
                    "raw": True
                },
                timeout=30
            )
            
            if response.status_code == 200:
                recipe_text = response.json()['response'].strip()
                
                # Validate response format
                if not ('Suggestion:' in recipe_text and 'Chain:' in recipe_text):
                    raise ValueError("Invalid response format")

                # Parse the response into the expected format
                recipe_dict = {
                    'question': question,
                    'answer': recipe_text,
                    'similarity_score': 1.0,
                    'structured_data': {
                        'suggestion': recipe_text.split('Chain:')[0].replace('Suggestion:', '').strip(),
                        'chain': [block.strip() for block in recipe_text.split('Chain:')[1].strip().split('>')]
                    }
                }

                return recipe_dict
            else:
                raise Exception(f"Ollama API error: {response.status_code}")

        except Exception as e:
            print(f"Error generating recipe: {str(e)}")
            # Return a fallback recipe from the CSV data
            with open('data/hx_receipts.csv', 'r', encoding='utf-8') as f:
                content = f.read().split('\n')[1:]  # Skip header
                if content:
                    fallback = content[0].split(';')[1]  # Use first recipe as fallback
                    return {
                        'question': question,
                        'answer': fallback,
                        'similarity_score': 0.5,
                        'structured_data': {
                            'suggestion': fallback.split('Chain:')[0].replace('Suggestion:', '').strip(),
                            'chain': [block.strip() for block in fallback.split('Chain:')[1].strip().split('>')]
                        }
                    }
    
    def extract_pedal_params(self, block_text: str) -> tuple[str, dict]:
        """Extract pedal name and parameters from a block of text"""
        params_dict = {}
        pedal_name = block_text
        
        # Extract parameters if present
        params_start = block_text.find('(')
        if params_start != -1:
            pedal_name = block_text[:params_start].strip()
            params_str = block_text[params_start:].strip('()')
            
            for param_pair in params_str.split(','):
                if ':' in param_pair:
                    key, value = map(str.strip, param_pair.split(':', 1))
                    params_dict[key.lower()] = value
                    
        return pedal_name, params_dict

    def match_parameters(self, suggested_params: dict, pedal_info: dict) -> List[Dict]:
        """Match suggested parameters with actual pedal parameters using TF-IDF"""
        if not pedal_info or not isinstance(pedal_info, dict) or 'params' not in pedal_info:
            return []

        # Ensure params is a list and not empty
        params = pedal_info.get('params', [])
        if not params or not isinstance(params, list):
            return []

        # Create vectorizer for parameter matching
        param_vectorizer = TfidfVectorizer(lowercase=True)
        
        # Safely extract parameter names, handling both dict and list formats
        pedal_param_names = []
        for param in params:
            if isinstance(param, dict):
                # Extract the first (and should be only) key from each parameter dict
                param_names = list(param.keys())
                if param_names:
                    pedal_param_names.append(param_names[0].lower())
        
        if not pedal_param_names:
            return []

        # Fit and transform pedal parameter names
        param_matrix = param_vectorizer.fit_transform(pedal_param_names)
        
        matched_params = []
        for suggested_param, value in suggested_params.items():
            # Transform suggested parameter name
            suggested_vector = param_vectorizer.transform([suggested_param])
            
            # Calculate similarities
            similarities = cosine_similarity(suggested_vector, param_matrix)[0]
            
            # Find best match if similarity is above threshold
            best_idx = np.argmax(similarities)
            if similarities[best_idx] > 0.9:  # Adjust threshold as needed
                actual_param_name = list(params[best_idx].keys())[0]
                matched_params.append({actual_param_name: value})
        
        return matched_params

    def find_relevant_pedals(self, question: str, reset_context: bool, top_k: Optional[int] = None) -> List[Dict]:
        """Find the most relevant pedals based on the question and recipe structure"""
        try:    
            # Find relevant recipes first to get context
            question_lower = question.lower()
            recipe_dict = self.generate_recipe(question_lower, reset_context)
            
            # Get the best recipe's answer from the dictionary
            recipe_answer = recipe_dict['answer'].lower()
            self.recipes.append("I have found some recipes that might help you.")
            self.recipes.append(recipe_answer)

            # Split recipe into individual blocks
            blocks = [b.strip() for b in recipe_answer.split('>')]
            matched_pedals = []
            
            for block in blocks:
                # Extract pedal name and parameters from block
                pedal_name, suggested_params = self.extract_pedal_params(block)
                
                # Find matching pedal through multiple stages
                best_match = None
                highest_score = 0
                
                # Create vectorizers for different matching stages
                name_vectorizer = TfidfVectorizer()
                category_vectorizer = TfidfVectorizer()
                
                # Prepare pedal data for matching
                pedal_names = [p['name'].lower() for p in self.pedals_info]
                pedal_categories = [f"{p['category']} {p['subcategory']}".lower() for p in self.pedals_info]
                
                # Create TF-IDF matrices
                names_matrix = name_vectorizer.fit_transform(pedal_names)
                categories_matrix = category_vectorizer.fit_transform(pedal_categories)
                
                # Match name
                name_vector = name_vectorizer.transform([pedal_name])
                name_similarities = cosine_similarity(name_vector, names_matrix)[0]
                
                # Match category
                category_vector = category_vectorizer.transform([pedal_name])
                category_similarities = cosine_similarity(category_vector, categories_matrix)[0]
                
                # Combine similarities with weights
                combined_similarities = name_similarities * 0.3 + category_similarities * 0.7
                
                # Find best matching pedal
                best_idx = np.argmax(combined_similarities)
                if combined_similarities[best_idx] > 0.1:  # Minimum similarity threshold
                    best_match = self.pedals_info[best_idx].copy()
                    
                    # Match and update parameters
                    if suggested_params and best_match.get('params'):
                        try:
                            matched_params = self.match_parameters(suggested_params, best_match)
                            if matched_params:
                                best_match['params'] = matched_params
                        except Exception as param_error:
                            print(f"Error matching parameters: {str(param_error)}")
                            # Keep original params if matching fails
                            pass
                    
                    matched_pedals.append(best_match)
            
            return matched_pedals
            
        except Exception as e:
            # Fallback to empty list instead of returning the error
            print(f"Error parsing recipe structure: {str(e)}")
            return []

    def answer_question(self, question: str, reset_context: bool = True) -> Dict[str, Union[str, List[Dict]]]:
        """Answer a question about the HX Stomp with structured JSON response"""
        try:
            # Add an initial recipe
            self.recipes = []
            self.recipes.append("Generating the pedal chain...")
            
            # Keywords that indicate the user wants to create or modify a pedal chain
            chain_related_keywords = [
                'create', 'setup', 'build', 'make', 'configure', 'chain',
                'pedal', 'effect', 'tone', 'sound', 'patch', 'recipe', 'ambient'
            ]
            
            # Check if the question is about creating/modifying a chain
            is_chain_request = any(keyword in question.lower() for keyword in chain_related_keywords)
            
            if not is_chain_request:
                return {
                    "error": "I am specifically designed to help with creating and configuring pedal chains. For general information about the HX Stomp, please refer to the manual or contact Line 6 support.",
                    "pedals": [],
                    "total_pedals": 0,
                    "remaining_slots": 8,
                    "max_chain_size": 8,
                }
            
            # Find relevant pedals based on the question
            relevant_pedals = self.find_relevant_pedals(question, reset_context)
                
            response = {
                "pedals": relevant_pedals,
                "total_pedals": len(relevant_pedals),
                "remaining_slots": max(0, 8 - len(relevant_pedals)),
                "max_chain_size": 8,
                "recipes": self.recipes
            }
            
            return response
            
        except Exception as e:
            return {
                "error": f"An error occurred: {str(e)}",
                "pedals": [],
                "total_pedals": 0,
                "remaining_slots": 8,
                "max_chain_size": 8
            }