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
        return {
            "name": model.get("name", "Unknown"),
            "params": self.format_pedal_params(model.get("params", []))
        }
            
    def load_pedals_json(self) -> List[Dict]:
        """Process pedals data into structured JSON format"""
        pedals_list = []
        
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
                            pedals_list.append(pedal_json)
        
        return pedals_list

    def load_knowledge_base(self) -> List[str]:
        """Load all knowledge sources"""
        self.pedals_info = []
        
        # Process pedals data into a flat list of pedals with parameters
        for category in self.pedals_data:
            if 'subcategories' in category:
                for subcat in category['subcategories']:
                    if 'models' in subcat:
                        for model in subcat['models']:
                            if isinstance(model, dict) and not model.get('use_subcategory'):
                                # Only process models with actual data
                                if model.get('name') and model.get('params') is not None:
                                    pedal_info = {
                                        "name": model['name'],
                                        "category": category['name'],
                                        "subcategory": subcat['name'],
                                        "params": self.format_pedal_params(model['params'])
                                    }
                                    self.pedals_info.append(pedal_info)
        
        # Create searchable text from pedal info
        knowledge = []
        for pedal in self.pedals_info:
            desc = f"Pedal: {pedal['name']}\n"
            desc += f"Category: {pedal['category']} > {pedal['subcategory']}\n"
            if pedal['params']:
                desc += "Parameters:\n"
                for param in pedal['params']:
                    param_name = list(param.keys())[0]
                    param_value = param[param_name]
                    desc += f"• {param_name}: {param_value}\n"
            knowledge.append(desc)
        
        # Load additional knowledge sources
        knowledge.extend(self._load_qa_data('data/hx_manual_qa_data.csv'))
        knowledge.extend(self._load_qa_data('data/hx_pedal_order_qa_data.csv'))
        knowledge.extend(self._load_qa_data('data/hx_receipts.csv'))
        
        return knowledge

    def _load_qa_data(self, filepath: str) -> List[str]:
        """Helper function to load Q&A data from CSV files"""
        try:
            df = pd.read_csv(filepath, sep=';', on_bad_lines='skip')
            if len(df.columns) > 2:
                df = df.iloc[:, :2]
                df.columns = ['Question', 'Answer']
            return [f"Q: {row['Question']}\nA: {row['Answer']}" for _, row in df.iterrows()]
        except Exception as e:
            print(f"Error loading data from {filepath}: {str(e)}")
            return []

    def find_relevant_pedals(self, question: str, top_k: int = 3) -> List[Dict]:
        """Find the most relevant pedals based on the question"""
        # Convert pedals to searchable text for matching
        pedal_texts = []
        for pedal in self.pedals_info:
            text = f"{pedal['name']} {pedal['category']} {pedal['subcategory']}"
            pedal_texts.append(text.lower())
        
        # Create TF-IDF matrix for pedal texts
        pedal_vectorizer = TfidfVectorizer()
        pedal_matrix = pedal_vectorizer.fit_transform(pedal_texts)
        question_vector = pedal_vectorizer.transform([question.lower()])
        
        # Calculate similarities
        similarities = cosine_similarity(question_vector, pedal_matrix)
        top_indices = similarities[0].argsort()[-top_k:][::-1]
        
        return [self.pedals_info[i] for i in top_indices]

    def answer_question(self, question: str) -> Dict[str, Union[str, List[Dict]]]:
        """Answer a question about the HX Stomp with structured JSON response"""
        try:
            # Find relevant pedals
            relevant_pedals = self.find_relevant_pedals(question)
            
            # Simplify the response format
            response = {
                "pedals": [
                    {
                        "name": pedal["name"],
                        "params": pedal["params"]
                    }
                    for pedal in relevant_pedals
                ]
            }
            
            return response
            
        except Exception as e:
            return {
                "error": f"An error occurred: {str(e)}",
                "pedals": []
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
        "How do I set up a distortion pedal?"
    ]
    
    print("HX Stomp Simple QA System - Test Results:")
    for question in test_questions:
        print(f"\nQ: {question}")
        response = qa.answer_question(question)
        print(f"A: {json.dumps(response, indent=2)}")
        print("-" * 80)