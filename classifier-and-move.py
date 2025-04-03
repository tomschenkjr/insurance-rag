import os
import shutil
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import joblib
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2

class InsuranceDocumentOrganizer:
    def __init__(self):
        self.categories = {
            'health': 'Health insurance',
            'cell': 'Cell phone insurance', 
            'condo': 'Condo insurance',
            'umbrella': 'Umbrella insurance',
            'business_travel': 'Business travel insurance',
            'watch': 'Watch insurance',
            'life': 'Life insurance',
            'dental': 'Dental insurance',
            'disability': 'Disability insurance',
            'vision': 'Vision insurance'
        }
        
        self.credit_cards = {
            'chase': {
                'sapphire_reserve': 'Chase Sapphire Reserve',
                'freedom_unlimited': 'Chase Freedom Unlimited'
            },
            'amex': {
                'platinum': 'American Express Platinum'
            }
        }
        
        self.model = None
        self.vectorizer = None
        
    def load_training_data(self, training_file: str) -> List[Dict]:
        """Load training data from JSON file"""
        try:
            with open(training_file, 'r') as f:
                training_data = json.load(f)
            return training_data
        except Exception as e:
            raise Exception(f"Failed to load training data: {str(e)}")
        
    def train_model(self, training_file: str):
        """Train SVM model on document training data"""
        # Load training data
        training_data = self.load_training_data(training_file)
        
        # Initialize and train vectorizer
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform([d['text'] for d in training_data])
        y = [d['category'] for d in training_data]
        
        # Train model
        self.model = SVC(kernel='linear')
        self.model.fit(X, y)
        
        print(f"Model trained successfully with {len(training_data)} examples")
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            raise Exception(f"Failed to extract text from {pdf_path}: {str(e)}")
        
    def classify_document(self, document_text: str) -> str:
        """Classify document into insurance category"""
        if not self.model or not self.vectorizer:
            raise Exception("Model not trained. Please train the model first.")
            
        X = self.vectorizer.transform([document_text])
        category = self.model.predict(X)[0]
        return category
        
    def classify_credit_card(self, document_text: str) -> Dict:
        """Classify credit card documents by issuer and card type"""
        card_info = {'issuer': None, 'card_type': None}
        
        text_lower = document_text.lower()
        
        # Check issuer
        if 'chase' in text_lower:
            card_info['issuer'] = 'chase'
            if 'sapphire reserve' in text_lower:
                card_info['card_type'] = 'sapphire_reserve'
            elif 'freedom unlimited' in text_lower:
                card_info['card_type'] = 'freedom_unlimited'
                
        elif 'american express' in text_lower or 'amex' in text_lower:
            card_info['issuer'] = 'amex'
            if 'platinum' in text_lower:
                card_info['card_type'] = 'platinum'
                
        return card_info
        
    def get_target_folder(self, category: str, year: str) -> str:
        """Get target folder path for document category and year"""
        if year == str(datetime.now().year):
            year = "Current"
            
        return os.path.join(category, year)
        
    def organize_document(self, pdf_path: str):
        """Organize single document into appropriate folder"""
        # Extract text
        doc_text = self.extract_text_from_pdf(pdf_path)
        
        # Get classification
        category = self.classify_document(doc_text)
        
        # Special handling for credit cards
        if category == 'credit_card':
            card_info = self.classify_credit_card(doc_text)
            if card_info['issuer'] and card_info['card_type']:
                category = os.path.join('Credit Cards', 
                                      self.credit_cards[card_info['issuer']][card_info['card_type']])
        
        # Get document year
        year = str(datetime.now().year)
        
        # Create target folder
        target_folder = self.get_target_folder(category, year)
        Path(target_folder).mkdir(parents=True, exist_ok=True)
        
        # Move file
        filename = os.path.basename(pdf_path)
        target_path = os.path.join(target_folder, filename)
        shutil.move(pdf_path, target_path)
        
        print(f"Moved {filename} to {target_path}")
        
    def organize_documents(self, input_folder: str):
        """Organize all PDF documents in input folder"""
        pdf_files = list(Path(input_folder).glob("*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {input_folder}")
            return
            
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                self.organize_document(str(pdf_file))
            except Exception as e:
                print(f"Error processing {pdf_file}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Insurance Document Organizer')
    parser.add_argument('--training-file', required=True, help='Path to training data JSON file')
    parser.add_argument('--input-dir', required=True, help='Directory containing PDF files to classify')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.training_file):
        print(f"Error: Training file {args.training_file} does not exist")
        return
        
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        return
    
    # Initialize and run organizer
    organizer = InsuranceDocumentOrganizer()
    
    try:
        # Train model
        print("Training model...")
        organizer.train_model(args.training_file)
        
        # Organize documents
        print("\nOrganizing documents...")
        organizer.organize_documents(args.input_dir)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
