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
        """Clean and format the answer in Markdown"""
        # Remove special tokens and prefixes
        text = re.sub(r'\[CLS\]|\[SEP\]', '', text)
        text = re.sub(r'^(?:question:|response:|answer:|a:|q:)\s*', '', text, flags=re.IGNORECASE)
        
        # Extract parameters section if it exists
        params_match = re.search(r'(?:parameters?|settings?):(.+?)(?=\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        params_section = ""
        if params_match:
            params_text = params_match.group(1)
            params = []
            for line in params_text.split('\n'):
                line = line.strip()
                if ':' in line:
                    param, value = line.split(':', 1)
                    params.append(f"* **{param.strip()}**: {value.strip()}")
                elif line and not line.startswith('*'):
                    params.append(f"* {line}")
            if params:
                params_section = "\n### Parameters\n\n" + "\n".join(params) + "\n"
            text = re.sub(r'(?:parameters?|settings?):(.+?)(?=\n\n|\Z)', '', text, re.IGNORECASE | re.DOTALL)

        # Split into sections and process each
        sections = re.split(r'\n(?=###)', text)
        formatted_sections = []
        
        for section in sections:
            if section.strip():
                # Process section title
                if section.startswith('###'):
                    title, content = section.split('\n', 1)
                    formatted_sections.append(f"\n{title.strip()}\n")
                    section = content
                
                # Format numbered steps
                if re.search(r'^\d+[\.)]\s*', section, re.MULTILINE):
                    steps = []
                    for line in section.split('\n'):
                        if re.match(r'^\d+[\.)]\s*', line):
                            line = re.sub(r'^(\d+)[\.)]\s*', r'\1. ', line)
                            steps.append(line)
                        elif line.strip():
                            steps.append(line)
                    formatted_sections.append("\n".join(steps))
                else:
                    formatted_sections.append(section)

        # Combine all sections
        text = "\n\n".join(s.strip() for s in formatted_sections if s.strip())
        
        # Add parameters section if it exists
        if params_section:
            text += "\n" + params_section

        # Final cleanup
        text = re.sub(r'\n\n+', '\n\n', text)  # Remove excessive newlines
        text = text.strip()
        
        return text

    def format_parameters(self, text):
        """Format parameters section in Markdown"""
        if not text:
            return text
            
        # Try to find parameters section
        if 'parameters:' in text.lower():
            try:
                parts = text.split('parameters:', 1)
                if len(parts) == 2:
                    main_content, params = parts
                    main_content = main_content.strip()
                    
                    # Process parameters
                    param_lines = []
                    for param in params.split('|'):
                        param = param.strip()
                        if not param:
                            continue
                            
                        if ':' in param:
                            name, value = param.split(':', 1)
                            param_lines.append(f"* **{name.strip()}**: {value.strip()}")
                        else:
                            param_lines.append(f"* {param}")
                    
                    if param_lines:
                        return f"{main_content}\n\n### Parameters\n\n" + '\n'.join(param_lines)
                else:
                    return text
            except Exception as e:
                print(f"Error formatting parameters: {str(e)}")
                return text
                
        return text

    def find_relevant_context(self, question, top_k=5):  # Increased from 2 to 5 for more context
        # Encode the question
        question_embedding = self.semantic_model.encode(question, convert_to_tensor=True)
        
        # Calculate cosine similarity using PyTorch
        similarities = F.cosine_similarity(question_embedding.unsqueeze(0), self.embeddings, dim=1)
        
        # Get top k indices
        top_indices = torch.argsort(similarities, descending=True)[:top_k]
        
        # Get similarity scores for weighting
        top_scores = similarities[top_indices]
        
        # Weight contexts by similarity score and combine
        weighted_contexts = []
        for idx, score in zip(top_indices, top_scores):
            context = self.knowledge_chunks[idx]
            # Only include contexts with similarity above 0.3
            if score > 0.3:
                weighted_contexts.append(context)
        
        return " ".join(weighted_contexts)

    def enhance_with_tinyllama(self, base_answer: str, question: str, context: str) -> str:
        """Generate clearer, more focused responses in Markdown format"""
        prompt = f"""As an expert on the Line 6 HX Stomp, provide a detailed and structured response in Markdown format.
        Focus on practical, actionable information and specific parameter values where relevant.

        When describing effects or settings:
        1. Start with an overview
        2. List specific steps or configurations
        3. Include parameter values with explanations
        4. Add any relevant tips or warnings

        Question: {question}
        Context: {context}
        Base Information: {base_answer}

        Provide a clear, structured response with appropriate sections and formatting:"""

        try:
            response = requests.post(
                self.ollama_endpoint,
                json={
                    "model": "tinyllama",
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7,  # Added temperature for more focused responses
                    "top_p": 0.9  # Added top_p for better quality
                },
                timeout=30
            )
            
            if response.status_code == 200:
                try:
                    response_json = response.json()
                    if isinstance(response_json, dict) and "response" in response_json:
                        enhanced = response_json["response"].strip()
                        if enhanced and len(enhanced) > 20:
                            return self.clean_answer(enhanced)
                    elif isinstance(response_json, str) and len(response_json.strip()) > 20:
                        # Handle case where response is directly a string
                        return self.clean_answer(response_json.strip())
                except (ValueError, AttributeError) as e:
                    print(f"Error parsing model response: {str(e)}")
            
            return self.clean_answer(base_answer)
            
        except Exception as e:
            print(f"Error enhancing response: {str(e)}")
            return self.clean_answer(base_answer)

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
                
                # Get the top 3 most likely answers
                start_logits = outputs.start_logits[0]
                end_logits = outputs.end_logits[0]
                
                # Get top 3 start and end positions
                top_starts = torch.topk(start_logits, k=3)
                top_ends = torch.topk(end_logits, k=3)
                
                answers = []
                for start_idx, end_idx in zip(top_starts.indices, top_ends.indices):
                    if end_idx >= start_idx and end_idx - start_idx < 100:  # Reasonable answer length
                        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx+1])
                        answer = self.tokenizer.convert_tokens_to_string(tokens)
                        if len(answer.strip()) > 10:  # Minimum answer length
                            answers.append(answer)
                
                # Combine answers if we found multiple good ones
                base_answer = " ".join(answers) if answers else ""
            
            # Clean up the base answer
            base_answer = self.clean_answer(base_answer)
            
            # Handle case when no good answer is found
            if not base_answer or base_answer.isspace() or len(base_answer) < 10:
                # Try to find relevant information from context
                keywords = [word.lower() for word in question.split() if len(word) > 3]
                context_lines = [line.strip() for line in context.split('.') if line.strip()]
                
                relevant_lines = []
                for line in context_lines:
                    if any(keyword in line.lower() for keyword in keywords):
                        relevant_lines.append(line)
                
                if relevant_lines:
                    base_answer = self.clean_answer(" ".join(relevant_lines))
                else:
                    return {
                        "answer": "I apologize, but I couldn't find a specific answer to your question. Could you please rephrase it or be more specific?",
                        "context": None,
                        "base_answer": None
                    }
            
            # Format parameters if present
            base_answer = self.format_parameters(base_answer)
            
            # Enhance the answer using TinyLlama
            enhanced_answer = self.enhance_with_tinyllama(base_answer, question, context)
                
            return {
                "answer": enhanced_answer,
                "context": context,
                "base_answer": base_answer
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