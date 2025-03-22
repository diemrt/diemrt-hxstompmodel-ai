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

class HXStompQA:
    def __init__(self):
        # Use temporary directory for cache
        self.cache_dir = tempfile.mkdtemp()
        self.model_name = "deepset/minilm-uncased-squad2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=self.cache_dir)
        self.load_knowledge_base()

    def load_pedals_json(self):
        """Load and process the HX Stomp pedals/effects data from JSON"""
        # First load the JSON structure documentation
        structure_df = pd.read_csv('data/json_structure.csv', sep=';')
        json_structure = {}
        for _, row in structure_df.iterrows():
            concept = row['Concept']
            if concept not in json_structure:
                json_structure[concept] = []
            json_structure[concept].append({
                'property': row['JSON Property'],
                'description': row['Description']
            })
        
        # Now load and process the pedals data
        with open('data/hx_pedals.json', 'r') as f:
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
            # Read CSV with error handling for inconsistent delimiters
            qa_df = pd.read_csv('data/hx_manual_qa_data.csv', sep=';', on_bad_lines='skip')
            
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
            order_df = pd.read_csv('data/hx_pedal_order_qa_data.csv', sep=';', on_bad_lines='skip')
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
            recipes_df = pd.read_csv('data/hx_receipts.csv', sep=';', on_bad_lines='skip')
            if len(recipes_df.columns) > 2:
                recipes_df = recipes_df.iloc[:, :2]
                recipes_df.columns = ['Question', 'Answer']
            knowledge = [f"Q: {row['Question']}\nA: {row['Answer']}" for _, row in recipes_df.iterrows()]
            return knowledge
        except Exception as e:
            print(f"Error loading recipes QA data: {str(e)}")
            return []

    def load_manual_text(self):
        """Load the full manual text"""
        with open('data/hx_stomp_manual.txt', 'r') as f:
            manual_text = f.read()
            
        # Split manual into manageable chunks
        # Remove page headers and clean up formatting
        cleaned_text = re.sub(r'\n\d+\n', '\n', manual_text)
        chunks = re.split(r'\n\s*\n', cleaned_text)
        # Filter out very short chunks and clean them
        knowledge = [chunk.strip() for chunk in chunks if len(chunk.strip()) > 50]
        return knowledge

    def load_knowledge_base(self):
        """Load all knowledge sources and create embeddings"""
        self.knowledge_chunks = []
        
        # Load all knowledge sources
        self.knowledge_chunks.extend(self.load_pedals_json())
        self.knowledge_chunks.extend(self.load_manual_qa())
        self.knowledge_chunks.extend(self.load_pedal_order_qa())  # Added pedal order knowledge
        self.knowledge_chunks.extend(self.load_receipts_qa())
        self.knowledge_chunks.extend(self.load_manual_text())
            
        # Create embeddings for similarity search
        self.embeddings = self.semantic_model.encode(self.knowledge_chunks, convert_to_tensor=True)

    def clean_answer(self, text):
        # Remove special tokens and question repetition
        text = re.sub(r'\[CLS\]|\[SEP\]', '', text)
        text = re.sub(r'^.*\?', '', text)  # Remove question repetition
        
        # Clean up formatting artifacts
        text = re.sub(r'\s*\(\s*', ' (', text)
        text = re.sub(r'\s*\)\s*', ') ', text)
        text = re.sub(r'\s*:\s*', ': ', text)
        text = re.sub(r'\s*\|\s*', ' | ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Format lists with bullet points
        text = re.sub(r'(?m)^(\d+\.\s+)', '• ', text)
        
        # Split into sentences and capitalize each one
        sentences = []
        for s in text.split('.'):
            s = s.strip()
            if s:
                # Format percentages consistently
                s = re.sub(r'(\d+)\s*%', r'\1%', s)
                # Format ranges consistently
                s = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1-\2', s)
                # Remove excessive spacing around dashes
                s = re.sub(r'\s+\-\s+', '-', s)
                sentences.append(s.capitalize())
        
        text = '. '.join(sentences)
        
        # Ensure proper sentence ending
        if not text.endswith('.'):
            text = text + '.'
            
        return text

    def format_parameters(self, text):
        if 'parameters:' in text.lower():
            # Split parameters into a cleaner format
            parts = text.split('Parameters:')
            if len(parts) > 1:
                # Extract main content
                main_content = parts[0].strip()
                
                # Format parameters with better spacing and bullet points
                params = [p.strip() for p in parts[1].split('|')]
                formatted_params = '\n\nParameters:\n' + '\n'.join('  • ' + p.strip().replace(':', ': ').replace(' - ', '-') for p in params)
                
                # Add separator lines for better visual distinction
                return f"{main_content}.\n\n{'─' * 40}{formatted_params}\n"
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

    def answer_question(self, question):
        try:
            # Find relevant context
            context = self.find_relevant_context(question)
            
            # Prepare input for the model
            inputs = self.tokenizer(
                question,
                context,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )

            # Get model outputs
            with torch.no_grad():
                outputs = self.qa_model(**inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                
            # Get the most likely answer span
            start_idx = torch.argmax(start_logits)
            end_idx = torch.argmax(end_logits)
            
            # Convert token positions to text
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            answer = self.tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx+1])
            
            # Clean up the answer
            answer = self.clean_answer(answer)
            
            # If no good answer is found, extract key information from context
            if not answer or answer.isspace() or len(answer) < 10:
                context_lines = [line.strip() for line in context.split('.') if line.strip()]
                relevant_info = next((line for line in context_lines 
                                   if any(keyword in line.lower() 
                                        for keyword in question.lower().split())), None)
                if relevant_info:
                    answer = self.clean_answer(relevant_info)
                else:
                    best_line = max(context_lines, key=len) if context_lines else context[:200]
                    answer = self.clean_answer(best_line)
            
            # Format parameters if present
            answer = self.format_parameters(answer)
                
            return {
                "answer": answer,
                "context": context
            }
        except Exception as e:
            return {
                "answer": f"I apologize, but I encountered an error: {str(e)}",
                "context": None
            }

if __name__ == "__main__":
    # Example usage
    print("Initializing HX Stomp QA System...")
    qa_system = HXStompQA()
    
    print("\nHX Stomp QA System Ready!")
    print("Ask questions about HX Stomp presets and effects (type 'quit' to exit)\n")
    
    # Test questions
    test_questions = [
        "How can I create a preset for an ambience sound?",
        "How can I tweak my Simple Delay pedal to get a warm tone?",
        "What are the parameters for the Hall Reverb?",
        "How do I create an ethereal soundscape?",
        "What's a good setting for rhythmic delays?"
    ]
    
    print("Running test questions:")
    for question in test_questions:
        print(f"\nQ: {question}")
        result = qa_system.answer_question(question)
        print(f"A: {result['answer']}")