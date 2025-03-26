import json
from typing import Dict, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class HXStompSimpleQA:
    def __init__(self):
        # Load the pedals data from JSON
        with open('data/hx_pedals.json', 'r') as f:
            self.pedals_data = json.load(f)
            
        # Create a knowledge base for different tones/styles
        self.tone_knowledge = {
            "lead guitar": {
                "description": "Lead guitar tones typically need compression, overdrive/distortion, and some ambience",
                "recommended_effects": ["Compressor", "Distortion", "Delay", "Reverb"]
            },
            "clean tone": {
                "description": "Clean tones benefit from compression and subtle modulation",
                "recommended_effects": ["Compressor", "EQ", "Modulation"]
            },
            "ambient": {
                "description": "Ambient tones use multiple delays and reverbs with modulation",
                "recommended_effects": ["Modulation", "Delay", "Reverb"]
            },
            "rock rhythm": {
                "description": "Rock rhythm needs tight compression and medium gain distortion",
                "recommended_effects": ["Compressor", "Distortion", "EQ"]
            },
            "metal": {
                "description": "Metal tones use high gain distortion with noise gate and tight EQ",
                "recommended_effects": ["Noise Gate", "Distortion", "EQ"]
            }
        }
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.descriptions = [v["description"] for v in self.tone_knowledge.values()]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.descriptions)

    def find_matching_tone(self, question: str) -> str:
        """Find the most relevant tone based on the question using TF-IDF similarity"""
        question_vector = self.vectorizer.transform([question])
        similarities = cosine_similarity(question_vector, self.tfidf_matrix)[0]
        best_match_idx = np.argmax(similarities)
        return list(self.tone_knowledge.keys())[best_match_idx]

    def find_pedal_by_category(self, category: str) -> Optional[Dict]:
        """Find a suitable pedal from a given category"""
        for category_data in self.pedals_data:
            if category.lower() in category_data["name"].lower():
                # Get the first available model from the first subcategory
                if "subcategories" in category_data:
                    for subcat in category_data["subcategories"]:
                        if "models" in subcat and subcat["models"]:
                            return next((model for model in subcat["models"] 
                                      if isinstance(model, dict) and "use_subcategory" not in model), None)
        return None

    def generate_param_values(self, param_name: str) -> str:
        """Generate reasonable default values for parameters"""
        param_defaults = {
            "Gain": "8",
            "Drive": "7",
            "Tone": "6",
            "Level": "Unity",
            "Mix": "50%",
            "Time": "400ms",
            "Feedback": "40%",
            "Bass": "0dB",
            "Treble": "0dB",
            "Decay": "4.0s",
            "Depth": "50%",
            "Rate": "3.0Hz"
        }
        
        # Check for common parameter names
        for key in param_defaults:
            if key.lower() in param_name.lower():
                return param_defaults[key]
        
        return "Default"

    def generate_pedal_chain(self, question: str) -> Dict:
        """Generate a chain of pedals based on the question"""
        # Find the matching tone type
        tone_type = self.find_matching_tone(question)
        recommended_effects = self.tone_knowledge[tone_type]["recommended_effects"]
        
        pedal_chain = []
        
        # Generate pedals for each recommended effect type
        for effect_type in recommended_effects:
            pedal = self.find_pedal_by_category(effect_type)
            if pedal:
                pedal_config = {
                    "name": pedal["name"],
                    "params": []
                }
                
                # Generate parameters if available
                if "params" in pedal:
                    for param in pedal["params"]:
                        if isinstance(param, dict):
                            param_name = list(param.keys())[0]
                            param_value = self.generate_param_values(param_name)
                            pedal_config["params"].append({param_name: param_value})
                
                pedal_chain.append(pedal_config)
        
        return {"pedals": pedal_chain}

    def answer_question(self, question: str) -> str:
        """Generate JSON response for the question"""
        try:
            pedal_chain = self.generate_pedal_chain(question)
            return json.dumps(pedal_chain, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

if __name__ == "__main__":
    # Example usage
    qa = HXStompSimpleQA()
    
    # Test questions
    test_questions = [
        "How can I create a tone for a lead guitar?",
        "I need a clean tone for jazz",
        "Set up an ambient soundscape",
        "Create a heavy metal distortion tone",
        "What's a good rock rhythm tone?"
    ]
    
    print("HX Stomp Simple QA System - Test Results:")
    for question in test_questions:
        print(f"\nQ: {question}")
        print(f"A: {qa.answer_question(question)}")
        print("-" * 80)  # Separator