import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import Dataset
import json
import os
from sentence_transformers import SentenceTransformer
import tempfile
import re
import requests
from typing import Dict, Optional
from pathlib import Path

class HXStompQA:
    def __init__(self):
        # Get the absolute path to the project root directory
        self.project_root = Path(__file__).parent.absolute()
        
        # Use temporary directory for cache
        self.cache_dir = tempfile.mkdtemp()
        self.model_name = "deepset/minilm-uncased-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=self.cache_dir)
        self.load_knowledge_base()
        self.ollama_endpoint = "http://localhost:11434/api/generate"

    def load_pedals_json(self):
        """Load and process the HX Stomp pedals/effects data from JSON"""
        # First load the JSON structure documentation using absolute path
        structure_df = pd.read_csv(self.project_root / 'data' / 'json_structure.csv', sep=';')
        json_structure = {}
        for _, row in structure_df.iterrows():
            concept = row['Concept']
            if concept not in json_structure:
                json_structure[concept] = []
            json_structure[concept].append({
                'property': row['JSON Property'],
                'description': row['Description']
            })
        
        # Now load and process the pedals data using absolute path
        with open(self.project_root / 'data' / 'hx_pedals.json', 'r') as f:
            pedals_data = json.load(f)
            
        knowledge = []
        # Add JSON structure documentation
        for concept, properties in json_structure.items():
            desc = f"JSON Structure - {concept}:\n"
            for prop in properties:
                desc += f"- {prop['property']}: {prop['description']}\n"
            knowledge.append(desc)
            
        # Process pedals data
        for category in pedals_data:
            if 'subcategories' in category:
                for subcat in category['subcategories']:
                    if 'models' in subcat:
                        for model in subcat['models']:
                            if not isinstance(model, dict) or 'use_subcategory' in model:
                                continue
                            # Create a description for each effect model
                            desc = f"Effect: {model.get('name', 'Unknown')} (ID: {model.get('id', 'Unknown')})\n"
                            desc += f"Category: {category['name']} > {subcat['name']}\n"
                            
                            # Add parameters if available
                            if 'params' in model and model['params']:  # Check if params exists and is not empty
                                params = []
                                for param in model['params']:
                                    if isinstance(param, dict):  # Check if param is a dictionary
                                        param_name = list(param.keys())[0] if param else None
                                        if param_name:
                                            display_name = param[param_name] if param[param_name] else param_name
                                            params.append(display_name)
                                if params:
                                    desc += f"Parameters: {' | '.join(params)}"
                            
                            knowledge.append(desc)
        return knowledge

    def load_manual_qa(self):
        """Load and process the HX Stomp manual Q&A data from CSV"""
        try:
            # Read CSV with error handling for inconsistent delimiters using absolute path
            qa_df = pd.read_csv(self.project_root / 'data' / 'hx_manual_qa_data.csv', sep=';', on_bad_lines='skip')
            
            # Ensure we only have two columns
            if len(qa_df.columns) > 2:
                # Keep only the first two columns
                qa_df = qa_df.iloc[:, :2]
                qa_df.columns = ['Question', 'Answer']
            
            # Convert to list of strings
            knowledge = []
            for _, row in qa_df.iterrows():
                qa_text = f"Q: {row['Question']}\nA: {row['Answer']}"
                knowledge.append(qa_text)
            
            return knowledge
        except Exception as e:
            print(f"Error loading manual QA data: {str(e)}")
            return []

    def load_pedal_order_qa(self):
        """Load Q&A pairs about pedal ordering from the dedicated dataset"""
        try:
            order_df = pd.read_csv(self.project_root / 'data' / 'hx_pedal_order_qa_data.csv', sep=';', on_bad_lines='skip')
            if len(order_df.columns) > 2:
                order_df = order_df.iloc[:, :2]
                order_df.columns = ['Question', 'Answer']
            knowledge = [f"Q: {row['Question']}\nA: {row['Answer']}" for _, row in order_df.iterrows()]
            return knowledge
        except Exception as e:
            print(f"Error loading pedal order QA data: {str(e)}")
            return []
        
    def load_receipts_qa(self):
        """Load tone recipes Q&A pairs"""
        try:
            recipes_df = pd.read_csv(self.project_root / 'data' / 'hx_receipts.csv', sep=';', on_bad_lines='skip')
            if len(recipes_df.columns) > 2:
                recipes_df = recipes_df.iloc[:, :2]
                recipes_df.columns = ['Question', 'Answer']
            knowledge = [f"Q: {row['Question']}\nA: {row['Answer']}" for _, row in recipes_df.iterrows()]
            return knowledge
        except Exception as e:
            print(f"Error loading recipes QA data: {str(e)}")
            return []

    def load_knowledge_base(self):
        """Load all knowledge sources and create embeddings"""
        self.knowledge_chunks = []
        
        # Load AI context settings
        try:
            context_df = pd.read_csv(self.project_root / 'data' / 'ai_context_settings.csv', sep=';')
            self.ai_context = {row['Setting']: {
                'description': row['Description'],
                'example': row['Example']
            } for _, row in context_df.iterrows()}
        except Exception as e:
            print(f"Warning: Could not load AI context settings: {str(e)}")
            self.ai_context = {}
        
        # Load all knowledge sources
        self.knowledge_chunks.extend(self.load_pedals_json())
        self.knowledge_chunks.extend(self.load_manual_qa())
        self.knowledge_chunks.extend(self.load_pedal_order_qa())
        self.knowledge_chunks.extend(self.load_receipts_qa())
            
        # Create embeddings for similarity search
        self.embeddings = self.semantic_model.encode(self.knowledge_chunks, convert_to_tensor=True)

    def clean_answer(self, text):
        # Remove more prefixes and clean up formatting
        text = re.sub(r'\[CLS\]|\[SEP\]', '', text)
        text = re.sub(r'^(?:question:|response:|answer:|a:|q:)\s*', '', text, flags=re.IGNORECASE)
        
        # Clean up numeric values
        text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)  # Fix decimal formatting
        text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1%', text)
        text = re.sub(r'(\d+(?:\.\d+)?)\s*(ms|hz|db)', r'\1 \2', text, flags=re.IGNORECASE)
        
        # Improve list formatting
        text = re.sub(r'(?m)^[-•]\s*', '• ', text)
        text = re.sub(r'(?m)^\d+\.\s+', '• ', text)
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        text = text.replace(' . ', '. ')
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('•'):
                lines.append(line)
            else:
                lines.append(line.capitalize())
        
        text = '\n'.join(lines)
        
        # Ensure proper sentence endings
        if text and not text.endswith('.'):
            text += '.'
            
        return text

    def format_parameters(self, text):
        """Format parameters with consistent spacing and organization"""
        if 'parameters:' in text.lower():
            main_content, params = text.split('parameters:', 1)
            main_content = main_content.strip()
            
            # Process and group parameters
            param_list = []
            current_group = []
            
            for param in params.split('|'):
                param = param.strip()
                if not param:
                    continue
                    
                if ':' in param:
                    name, value = param.split(':', 1)
                    param_formatted = f"• {name.strip()}: {value.strip()}"
                else:
                    param_formatted = f"• {param}"
                
                if len(current_group) >= 3:  # Group parameters in sets of 3
                    param_list.extend(current_group)
                    param_list.append('')  # Add spacing between groups
                    current_group = []
                current_group.append(param_formatted)
            
            if current_group:
                param_list.extend(current_group)
            
            if param_list:
                return f"{main_content}\n\nParameters:\n" + '\n'.join(param_list)
        
        return text

    def find_relevant_context(self, question, top_k=2):
        # Encode the question
        question_embedding = self.semantic_model.encode(question, convert_to_tensor=True)
        
        # Calculate cosine similarity using PyTorch
        similarities = F.cosine_similarity(question_embedding.unsqueeze(0), self.embeddings, dim=1)
        
        # Get top k indices
        top_indices = torch.argsort(similarities, descending=True)[:top_k]
        
        # Combine relevant contexts
        context = " ".join([self.knowledge_chunks[i] for i in top_indices])
        return context

    def enhance_with_tinyllama(self, base_answer: str, question: str, context: str) -> str:
        """Generate clearer, more focused responses"""
        prompt = f"""Based on the following context about the Line 6 HX Stomp, 
        provide a clear and focused response. Include specific parameter values where relevant, 
        but keep the response concise and practical.

        Guidelines:
        - Remove unnecessary prefixes (Question:, Response:, etc.)
        - Format parameter values consistently (e.g., 0.5 not 0. 5)
        - Group related parameters together
        - Use bullet points for lists
        - Focus on practical information
        - Keep responses concise

        Question: {question}
        Context: {context}
        Initial Answer: {base_answer}

        Enhanced Response:"""

        try:
            response = requests.post(
                self.ollama_endpoint,
                json={
                    "model": "tinyllama",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                enhanced = response.json().get("response", "").strip()
                if enhanced and len(enhanced) > 20:
                    enhanced = self.clean_answer(enhanced)
                    return self.format_parameters(enhanced)
            
            return base_answer
            
        except Exception as e:
            print(f"Error enhancing response: {str(e)}")
            return base_answer

    def answer_question(self, question: str) -> Dict[str, Optional[str]]:
        try:
            # Find relevant context
            context = self.find_relevant_context(question)
            
            # Get base answer using the QA model
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )

            with torch.no_grad():
                outputs = self.qa_model(**inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                
            start_idx = torch.argmax(start_logits)
            end_idx = torch.argmax(end_logits)
            
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            base_answer = self.tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx+1])
            
            # Clean up the base answer
            base_answer = self.clean_answer(base_answer)
            
            # Handle case when no good answer is found
            if not base_answer or base_answer.isspace() or len(base_answer) < 10:
                context_lines = [line.strip() for line in context.split('.') if line.strip()]
                relevant_info = next((line for line in context_lines 
                                   if any(keyword in line.lower() 
                                        for keyword in question.lower().split())), None)
                if relevant_info:
                    base_answer = self.clean_answer(relevant_info)
                else:
                    best_line = max(context_lines, key=len) if context_lines else context[:200]
                    base_answer = self.clean_answer(best_line)
            
            # Format parameters if present
            base_answer = self.format_parameters(base_answer)
            
            # Enhance the answer using TinyLlama
            enhanced_answer = self.enhance_with_tinyllama(base_answer, question, context)
                
            return {
                "answer": enhanced_answer,
                "context": context,
                "base_answer": base_answer  # Include original answer for reference
            }
        except Exception as e:
            return {
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "context": None,
                "base_answer": None
            }

if __name__ == "__main__":
    # Example usage
    print("Initializing HX Stomp QA System...")
    qa_system = HXStompQA()
    
    print("\nHX Stomp QA System Ready!")
    print("Ask questions about HX Stomp presets and effects (type 'quit' to exit)\n")
    
    # Test questions
    test_questions = [
        "Give me a full list of the pedals to use, with the names and the parameters of the HX Stomp, in the correct order",
        "What's the recommended signal chain order for my effects?",
        "How can I create a preset for an ambience sound?",
        "How can I tweak my Simple Delay pedal to get a warm tone?",
        "What are all the parameters for the Hall Reverb?",
        "Show me all available distortion pedals and their parameters",
        "How do I create an ethereal soundscape?",
        "What's a good setting for rhythmic delays?",
        "List all modulation effects available in the HX Stomp",
        "For a Simple Delay pedal, how can I set the feedback to and mix to create an ambient sound?",
    ]
    
    print("Running test questions:")
    for question in test_questions:
        print(f"\nQ: {question}")
        result = qa_system.answer_question(question)
        print(f"A: {result['answer']}\n")
        print("-" * 80)  # Add separator between questions for better readability