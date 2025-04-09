import json
import pandas as pd
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
        
    def find_relevant_recipes(self, question: str, top_k: int = 3) -> List[Dict]:
        """
        Find relevant recipes from knowledge base based on question similarity.
        Returns recipes sorted by similarity score, with scores included.
        """
        # Load and prepare recipes data
        recipes_data = []
        with open('data/hx_receipts.csv', 'r', encoding='utf-8') as f:
            content = f.read().replace('\r\n', '\n')
            lines = content.split('\n')[1:]  # Skip header
            for line in lines:
                if line.strip():
                    q, a = line.split(';')
                    recipes_data.append({
                        'question': q.strip(),
                        'answer': a.strip()
                    })

        # Return empty list if no recipes found
        if not recipes_data:
            return []

        # Vectorize recipes questions for comparison
        vectorizer = TfidfVectorizer()
        recipe_questions = [r['question'] for r in recipes_data]
        questions_matrix = vectorizer.fit_transform(recipe_questions)
        query_vector = vectorizer.transform([question])

        # Calculate similarities
        similarities = cosine_similarity(query_vector, questions_matrix)[0]
        
        # Create list of recipes with their similarity scores
        scored_recipes = [
            {**recipes_data[i], 'similarity_score': similarities[i]}
            for i in range(len(recipes_data))
            if similarities[i] > 0.1  # Keep threshold for relevance
        ]
        
        # Sort by similarity score in descending order
        scored_recipes.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        # Return top_k recipes or all if less available
        return scored_recipes[:top_k]
    
    def generate_recipe_suggestions(self, question: str, max_suggestions: int = 10) -> List[Dict]:
        """
        Generate recipe suggestions using Ollama and existing knowledge.
        Returns a list of suggested recipes with pedal chains.
        """
        try:
            # First, get relevant existing recipes
            existing_recipes = self.find_relevant_recipes(question, top_k=3)
            
            # Add existing recipes to context
            context += "\nExample recipes:\n"
            for recipe in existing_recipes:
                context += f"Q: {recipe['question']}\nA: {recipe['answer']}\n\n"
            
            # Prepare prompt for Ollama
            prompt = f"""Based on the following context and question, suggest {max_suggestions} new pedal chain recipes.
            Keep suggestions concise and focused on pedal combinations.
            
            Context:
            {context}
            
            Question: {question}
            
            Generate {max_suggestions} different recipe suggestions."""

            # Call Ollama API
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": "tinyllama",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                generated_text = response.json()['response']
                
                # Process generated suggestions and combine with existing recipes
                suggestions = []
                
                # Add existing recipes first
                for recipe in existing_recipes:
                    suggestions.append({
                        'source': 'existing',
                        'question': recipe['question'],
                        'answer': recipe['answer'],
                        'similarity_score': recipe.get('similarity_score', 1.0)
                    })
                
                # Add generated suggestions
                # Split the generated text into separate suggestions
                generated_parts = generated_text.split('\n\n')
                for part in generated_parts:
                    if part.strip():
                        suggestions.append({
                            'source': 'generated',
                            'question': question,
                            'answer': part.strip(),
                            'similarity_score': 0.8  # Default score for generated content
                        })
                
                return suggestions[:max_suggestions]
            
            return existing_recipes  # Fallback to existing recipes if API call fails
            
        except Exception as e:
            print(f"Error generating recipes: {str(e)}")
            return existing_recipes  # Fallback to existing recipes

    def find_relevant_pedals(self, question: str, top_k: Optional[int] = None) -> List[Dict]:
        """Find the most relevant pedals based on the question"""
        # Find relevant recipes first to get context
        question_lower = question.lower()
        best_recips = self.generate_recipe_suggestions(question_lower, top_k)
        
        # Use the best recipe's answer as context for pedal search
        answer_lower = best_recips[0]['answer'].lower() if best_recips else ''
        self.recipes.append("I have found some recipes that might help you.")
        self.recipes.append(answer_lower)
        
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
        question_vector = pedal_vectorizer.transform([answer_lower])
        
        # Calculate similarities
        similarities = cosine_similarity(question_vector, pedal_matrix)[0]
        
        # Filter out irrelevant results (similarity score too low)
        min_similarity = 0.05 if 'ambient' in answer_lower else 0.1  # Lower threshold for ambient
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

    def answer_question(self, question: str) -> Dict[str, Union[str, List[Dict]]]:
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
            relevant_pedals = self.find_relevant_pedals(question)
            
            # Apply pedal ordering based on hx_pedal_order_qa_data.csv guidelines
            if relevant_pedals:
                validated_pedals = self.validate_pedal_chain(relevant_pedals)
                
                response = {
                    "pedals": validated_pedals,
                    "total_pedals": len(validated_pedals),
                    "remaining_slots": max(0, 8 - len(validated_pedals)),
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