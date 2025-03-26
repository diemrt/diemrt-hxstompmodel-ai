import json
import pandas as pd
from typing import Dict, List, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
        
        # Initialize the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.knowledge_base)

    def format_pedal_params(self, params: List[Dict]) -> List[Dict]:
        """Format pedal parameters with their units and ranges"""
        formatted_params = []
        for param in params:
            if isinstance(param, dict):
                param_name = list(param.keys())[0] if param else None
                if param_name and param_name in self.parameter_info:
                    param_info = self.parameter_info[param_name]
                    unit = param_info.get('unitOfMeasure', '')
                    
                    # Handle different parameter types
                    if unit == 'boolean':
                        suggested_value = "Off"
                        value_with_unit = suggested_value
                    elif unit == 'type':
                        suggested_value = str(param_info['from'])
                        value_with_unit = f"Type {suggested_value}"
                    else:
                        # Handle numeric parameters with proper ranges and units
                        try:
                            from_val = float(param_info.get('from', 0))
                            to_val = float(param_info.get('to', 100))
                            
                            # Use middle value by default
                            suggested_value = (from_val + to_val) / 2
                            
                            # Format based on unit type
                            if unit in ['ms', 'Hz']:
                                # Keep decimals for time and frequency
                                suggested_value = f"{suggested_value:.1f}"
                            else:
                                # Round to integer for other numeric values
                                suggested_value = f"{int(suggested_value)}"
                            
                            # Special handling for specific units
                            if unit == 'dB':
                                if suggested_value == '-inf':
                                    value_with_unit = "-∞ dB"
                                else:
                                    value_with_unit = f"{suggested_value} dB"
                            elif unit == '%':
                                value_with_unit = f"{suggested_value}%"
                            elif unit == 'Hz':
                                value_with_unit = f"{suggested_value} Hz"
                            elif unit == 'ms':
                                value_with_unit = f"{suggested_value} ms"
                            elif unit == 'semitones':
                                value_with_unit = f"{suggested_value} st"
                            elif unit == 'cents':
                                value_with_unit = f"{suggested_value} cents"
                            else:
                                value_with_unit = str(suggested_value)
                        except:
                            suggested_value = str(param_info.get('from', 'Default'))
                            value_with_unit = suggested_value
                    
                    formatted_param = {
                        param_name: value_with_unit
                    }
                    formatted_params.append(formatted_param)
        return formatted_params

    def create_pedal_json(self, model: Dict) -> Dict:
        """Create a structured JSON representation of a pedal"""
        formatted_pedal = {
            "name": model.get("name", "Unknown"),
            "params": self.format_pedal_params(model.get("params", [])),
            "position": model.get("position", 0)  # Add position in chain (0-7)
        }
        return formatted_pedal

    def validate_pedal_chain(self, pedals: List[Dict]) -> List[Dict]:
        """Validate and format a chain of pedals"""
        # Ensure no more than 8 pedals
        if len(pedals) > 8:
            pedals = pedals[:8]
        
        # Sort pedals by position if specified
        pedals.sort(key=lambda x: x.get("position", 999))
        
        # Assign positions to pedals that don't have one
        current_position = 0
        formatted_chain = []
        for pedal in pedals:
            if current_position < 8:  # Only process up to 8 pedals
                if "position" not in pedal:
                    pedal["position"] = current_position
                formatted_chain.append(pedal)
                current_position += 1
        
        return formatted_chain

    def load_pedals_json(self) -> List[Dict]:
        """Process pedals data into structured JSON format with chain validation"""
        pedals_list = []
        current_chain = []
        
        for category in self.pedals_data:
            if 'subcategories' in category:
                for subcat in category['subcategories']:
                    if 'models' in subcat:
                        for model in subcat['models']:
                            if not isinstance(model, dict) or 'use_subcategory' in model:
                                continue
                            
                            # Add category and subcategory to model
                            model['category'] = category['name']
                            model['subcategory'] = subcat['name']
                            
                            pedal_json = self.create_pedal_json(model)
                            current_chain.append(pedal_json)
                        
                        # Validate chain when we finish processing a subcategory
                        if current_chain:
                            validated_chain = self.validate_pedal_chain(current_chain)
                            pedals_list.extend(validated_chain)
                            current_chain = []  # Reset for next chain
        
        return pedals_list

    def load_knowledge_base(self) -> List[str]:
        """Load all knowledge sources"""
        knowledge = []
        
        # Load recipes first for better matching
        recipes = self._load_qa_data('data/hx_receipts.csv')
        if recipes:
            knowledge.extend(recipes)
        
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

    def find_relevant_pedals(self, question: str, top_k: Optional[int] = None) -> List[Dict]:
        """Find the most relevant pedals based on the question"""
        # Keywords for different effect types
        effect_keywords = {
            'ambient': ['reverb', 'delay', 'modulation', 'chorus'],
            'blues': ['overdrive', 'tube', 'compression', 'boost'],
            'rock': ['distortion', 'overdrive', 'delay'],
            'clean': ['compression', 'eq', 'reverb'],
            'worship': ['delay', 'reverb', 'modulation'],
        }
        
        # Check for style keywords and add relevant effects to search
        question_lower = question.lower()
        expanded_question = question_lower
        for style, effects in effect_keywords.items():
            if style in question_lower:
                expanded_question += ' ' + ' '.join(effects)
        
        # Convert pedals to searchable text for matching
        pedal_texts = []
        for pedal in self.pedals_info:
            text = f"{pedal['name']} {pedal['category']} {pedal['subcategory']}"
            # Add parameter names to improve matching
            if pedal.get('params'):
                param_names = ' '.join(list(p.keys())[0] for p in pedal['params'] if isinstance(p, dict))
                text += f" {param_names}"
            pedal_texts.append(text.lower())
        
        # Create TF-IDF matrix for pedal texts
        pedal_vectorizer = TfidfVectorizer()
        pedal_matrix = pedal_vectorizer.fit_transform(pedal_texts)
        question_vector = pedal_vectorizer.transform([expanded_question])
        
        # Calculate similarities
        similarities = cosine_similarity(question_vector, pedal_matrix)[0]
        
        # Filter out irrelevant results (similarity score too low)
        min_similarity = 0.05 if 'ambient' in question_lower else 0.1  # Lower threshold for ambient
        relevant_indices = [i for i, score in enumerate(similarities) if score > min_similarity]
        
        # Sort by similarity score
        relevant_indices.sort(key=lambda i: similarities[i], reverse=True)
        
        # If top_k is specified, limit the results
        if top_k is not None:
            relevant_indices = relevant_indices[:top_k]
        
        # Deduplicate by category (e.g., don't return too many compressors)
        seen_categories = {}
        filtered_pedals = []
        for idx in relevant_indices:
            pedal = self.pedals_info[idx]
            category = pedal['category']
            if category not in seen_categories:
                seen_categories[category] = 1
                filtered_pedals.append(pedal)
            elif seen_categories[category] < 2:  # Allow up to 2 pedals per category
                seen_categories[category] += 1
                filtered_pedals.append(pedal)
        
        return filtered_pedals

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts"""
        # Clean and normalize the texts
        text1 = text1.lower().strip()
        text2 = text2.lower().strip()
        
        # Create TF-IDF vectors for the texts
        vectorizer = TfidfVectorizer(stop_words='english')
        try:
            tfidf = vectorizer.fit_transform([text1, text2])
            return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        except:
            return 0.0

    def _apply_pedal_order_rules(self, pedals: List[Dict]) -> List[Dict]:
        """Apply pedal ordering rules based on best practices"""
        # Define category order based on hx_pedal_order_qa_data.csv guidelines
        category_order = {
            'Dynamics': 0,      # Compression first
            'Distortion': 1,    # Gain effects after compression
            'EQ': 2,           # EQ to shape the distorted signal
            'Modulation': 3,    # Modulation effects after gain
            'Delay': 4,        # Time-based effects near the end
            'Reverb': 5        # Reverb typically last
        }
        
        # Sort pedals based on their category
        ordered_pedals = sorted(pedals, key=lambda p: category_order.get(p['category'], 999))
        
        # Assign positions based on the sorted order
        for i, pedal in enumerate(ordered_pedals):
            pedal['position'] = i
        
        return ordered_pedals

    def answer_question(self, question: str) -> Dict[str, Union[str, List[Dict]]]:
        """Answer a question about the HX Stomp with structured JSON response"""
        try:
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
                    "max_chain_size": 8
                }
            
            # First, search for matching recipes in the knowledge base
            recipes = []
            best_similarity = 0
            
            for qa_pair in self.knowledge_base:
                if qa_pair.startswith("Q:"):
                    qa_lines = qa_pair.split('\n')
                    if len(qa_lines) >= 2:
                        recipe_q = qa_lines[0][3:].strip()  # Remove "Q: "
                        recipe_a = qa_lines[1][3:].strip()  # Remove "A: "
                        
                        # Calculate similarity between question and recipe question
                        similarity = self._calculate_similarity(question, recipe_q)
                        
                        # Store recipe if similarity is above threshold
                        # Lower threshold for ambient sounds to catch more matches
                        threshold = 0.15 if 'ambient' in question.lower() else 0.2
                        if similarity > threshold:
                            if similarity > best_similarity:
                                # Insert best match at the beginning
                                recipes.insert(0, recipe_a)
                                best_similarity = similarity
                            else:
                                recipes.append(recipe_a)
            
            # Find relevant pedals based on the question
            relevant_pedals = self.find_relevant_pedals(question)
            
            # If we found recipes, enhance pedal selection with recipe pedals
            if recipes and relevant_pedals:
                for recipe in recipes:
                    recipe_pedals = self.find_relevant_pedals(recipe)
                    for recipe_pedal in recipe_pedals:
                        if not any(p['name'] == recipe_pedal['name'] for p in relevant_pedals):
                            relevant_pedals.append(recipe_pedal)
            
            # Apply pedal ordering based on hx_pedal_order_qa_data.csv guidelines
            if relevant_pedals:
                ordered_pedals = self._apply_pedal_order_rules(relevant_pedals)
                validated_pedals = self.validate_pedal_chain(ordered_pedals)
                
                response = {
                    "pedals": validated_pedals,
                    "total_pedals": len(validated_pedals),
                    "remaining_slots": max(0, 8 - len(validated_pedals)),
                    "max_chain_size": 8,
                    "recipes": recipes if recipes else None
                }
                
                return response
            elif recipes:  # If we have recipes but no pedals, still return the recipes
                return {
                    "pedals": [],
                    "total_pedals": 0,
                    "remaining_slots": 8,
                    "max_chain_size": 8,
                    "recipes": recipes
                }
            
        except Exception as e:
            return {
                "error": f"An error occurred: {str(e)}",
                "pedals": [],
                "total_pedals": 0,
                "remaining_slots": 8,
                "max_chain_size": 8
            }

if __name__ == "__main__":
    # Example usage
    qa = HXStompSimpleQA()
    
    # Test questions
    test_questions = [
        "Tell me about the Compulsive Drive",
        "What are the parameters for delay pedals?",
        "Show me reverb options",
        "What modulation effects are available?",
        "How do I set up a distortion pedal?",
        "Create a pedal chain with a compressor, overdrive, and reverb",
        "Configure a patch with delay and modulation",
        "What is the best order for distortion and reverb?",
        "How to build a pedalboard for ambient sounds?",
        "Give me a recipe for a blues tone",
        "What is the capital of France?"
    ]
    
    print("HX Stomp Simple QA System - Test Results:")
    for question in test_questions:
        print(f"\nQ: {question}")
        response = qa.answer_question(question)
        print(f"A: {json.dumps(response, indent=2)}")
        print("-" * 80)