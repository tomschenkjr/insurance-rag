import os
import json
import shutil
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import fitz  # PyMuPDF
import re

class InsuranceDocumentOrganizer:
    def __init__(self, base_folder="./insurance_docs", training_data_path="training_data.json"):
        self.base_folder = base_folder
        self.training_data_path = training_data_path
        self.model = None
        self.vectorizer = None
        self.categories = [
            "health",
            "cell",
            "condo",
            "umbrella",
            "business_travel",
            "watch",
            "life",
            "dental",
            "disability",
            "vision",
            "credit_card"
        ]
        
        # Credit card subcategories for issue #2
        self.credit_card_issuers = {
            "chase": ["sapphire_reserve", "freedom_unlimited", "freedom_flex"],
            "american_express": ["platinum", "gold", "blue_cash"],
            "citi": ["double_cash", "custom_cash"],
            "discover": ["it", "miles"],
        }
        
        # Ensure base folder exists
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)
        
        # Create folder structure
        self.create_folder_structure()
    
    def create_folder_structure(self):
        """Create the folder structure for organizing documents"""
        current_year = datetime.now().year
        
        # Create main category folders
        for category in self.categories:
            if category == "credit_card":
                # For credit cards, create issuer and card type subfolders
                for issuer, cards in self.credit_card_issuers.items():
                    for card in cards:
                        card_path = os.path.join(self.base_folder, category, issuer, card)
                        
                        # Create year subfolders for each card
                        if not os.path.exists(card_path):
                            os.makedirs(card_path)
                        
                        # Create current year folder
                        current_path = os.path.join(card_path, "Current")
                        if not os.path.exists(current_path):
                            os.makedirs(current_path)
                        
                        # Create previous years folders
                        for year in range(current_year, current_year - 3, -1):
                            year_path = os.path.join(card_path, str(year))
                            if not os.path.exists(year_path):
                                os.makedirs(year_path)
            else:
                # For other insurance types
                category_path = os.path.join(self.base_folder, category)
                
                if not os.path.exists(category_path):
                    os.makedirs(category_path)
                
                # Create current year folder
                current_path = os.path.join(category_path, "Current")
                if not os.path.exists(current_path):
                    os.makedirs(current_path)
                
                # Create previous years folders
                for year in range(current_year, current_year - 3, -1):
                    year_path = os.path.join(category_path, str(year))
                    if not os.path.exists(year_path):
                        os.makedirs(year_path)
    
    def load_training_data(self):
        """Load training data from JSON file"""
        try:
            with open(self.training_data_path, 'r') as f:
                training_data = json.load(f)
            
            texts = [item['text'] for item in training_data]
            categories = [item['category'] for item in training_data]
            
            return texts, categories
        except FileNotFoundError:
            print(f"Error: Training data file '{self.training_data_path}' not found.")
            return [], []
    
    def train_model(self):
        """Train the SVM model for document classification"""
        texts, categories = self.load_training_data()
        
        if not texts:
            print("No training data available. Model training failed.")
            return False
        
        # Create a pipeline with TF-IDF vectorizer and SVM classifier
        self.model = Pipeline([
            ('vectorizer', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ('classifier', SVC(kernel='linear', probability=True))
        ])
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            texts, categories, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        accuracy = self.model.score(X_test, y_test)
        print(f"Model trained with accuracy: {accuracy:.2f}")
        
        return True
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text content from a PDF file"""
        try:
            text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def classify_document(self, pdf_path):
        """Classify a document using the trained model"""
        if not self.model:
            print("Model not trained. Please train the model first.")
            return None, None, None
        
        text = self.extract_text_from_pdf(pdf_path)
        
        if not text:
            print(f"Could not extract text from {pdf_path}")
            return None, None, None
        
        # Get prediction and probabilities
        category = self.model.predict([text])[0]
        probabilities = self.model.predict_proba([text])[0]
        
        # Get the confidence score for the predicted category
        category_idx = list(self.model.classes_).index(category)
        confidence = probabilities[category_idx]
        
        # If credit card, further classify by issuer and card type
        card_issuer = None
        card_type = None
        
        if category == "credit_card":
            # Create a simple keyword-based classifier for credit cards
            card_issuer, card_type = self.classify_credit_card(text)
        
        return category, card_issuer, card_type
    
    def classify_credit_card(self, text):
        """Classify credit card documents by issuer and card type"""
        text = text.lower()
        
        # Check for issuer
        issuer = None
        for potential_issuer in self.credit_card_issuers.keys():
            if potential_issuer.lower().replace("_", " ") in text:
                issuer = potential_issuer
                break
        
        # Check for card type
        card_type = None
        if issuer:
            for potential_card in self.credit_card_issuers.get(issuer, []):
                if potential_card.lower().replace("_", " ") in text:
                    card_type = potential_card
                    break
        
        return issuer, card_type
    
    def extract_document_year(self, pdf_path, text=None):
        """Extract the year from document content or filename"""
        if text is None:
            text = self.extract_text_from_pdf(pdf_path)
        
        # Look for year patterns in the text (2020-2030)
        years = re.findall(r'\b(20[2-3][0-9])\b', text)
        
        if years:
            # Return the most recent year found
            return max(years)
        
        # Check filename for year
        filename = os.path.basename(pdf_path)
        years_in_filename = re.findall(r'\b(20[2-3][0-9])\b', filename)
        
        if years_in_filename:
            return max(years_in_filename)
        
        # If no year found, use current year
        return str(datetime.now().year)
    
    def move_document(self, pdf_path, category, card_issuer=None, card_type=None):
        """Move document to appropriate folder based on classification"""
        year = self.extract_document_year(pdf_path)
        current_year = str(datetime.now().year)
        
        # Determine target folder
        if category == "credit_card" and card_issuer and card_type:
            if year == current_year:
                target_folder = os.path.join(self.base_folder, category, card_issuer, card_type, "Current")
            else:
                target_folder = os.path.join(self.base_folder, category, card_issuer, card_type, year)
        else:
            if year == current_year:
                target_folder = os.path.join(self.base_folder, category, "Current")
            else:
                target_folder = os.path.join(self.base_folder, category, year)
        
        # Ensure target folder exists
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        # Move the file
        filename = os.path.basename(pdf_path)
        target_path = os.path.join(target_folder, filename)
        
        # If file already exists, add a suffix
        if os.path.exists(target_path):
            base, ext = os.path.splitext(filename)
            i = 1
            while os.path.exists(os.path.join(target_folder, f"{base}_{i}{ext}")):
                i += 1
            target_path = os.path.join(target_folder, f"{base}_{i}{ext}")
        
        try:
            shutil.copy2(pdf_path, target_path)
            print(f"Moved: {pdf_path} -> {target_path}")
            return True
        except Exception as e:
            print(f"Error moving file {pdf_path}: {e}")
            return False
    
    def process_documents(self, input_folder):
        """Process all PDF documents in the input folder"""
        if not self.model:
            success = self.train_model()
            if not success:
                print("Failed to train model. Exiting.")
                return
        
        # Check if input folder exists
        if not os.path.exists(input_folder):
            print(f"Input folder '{input_folder}' does not exist.")
            return
        
        # Process each PDF file in the input folder
        processed_count = 0
        for filename in os.listdir(input_folder):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(input_folder, filename)
                
                print(f"Processing {filename}...")
                category, card_issuer, card_type = self.classify_document(pdf_path)
                
                if category:
                    success = self.move_document(pdf_path, category, card_issuer, card_type)
                    if success:
                        processed_count += 1
        
        print(f"Processing complete. Organized {processed_count} documents.")


# Example usage
if __name__ == "__main__":
    # Create sample training data if it doesn't exist
    if not os.path.exists("training_data.json"):
        sample_data = [
            {"text": "This health insurance policy provides coverage for medical expenses...", "category": "health"},
            {"text": "Your condo insurance policy covers damage to your unit...", "category": "condo"},
            {"text": "Your umbrella insurance provides extra liability coverage...", "category": "umbrella"},
            {"text": "Chase Sapphire Reserve credit card benefits include travel insurance...", "category": "credit_card"},
            {"text": "American Express Platinum card offers premium travel benefits...", "category": "credit_card"},
            {"text": "Your dental insurance covers preventive care and basic procedures...", "category": "dental"},
            {"text": "This cell phone insurance policy covers damage, loss, and theft...", "category": "cell"},
            {"text": "Your life insurance policy provides $500,000 in coverage...", "category": "life"},
            {"text": "This watch insurance policy covers your luxury timepieces...", "category": "watch"},
            {"text": "Vision insurance benefits include eye exams and glasses...", "category": "vision"},
            {"text": "Short-term disability insurance provides income if you're unable to work...", "category": "disability"},
            {"text": "Business travel insurance covers you while traveling for work...", "category": "business_travel"}
        ]
        
        with open("training_data.json", "w") as f:
            json.dump(sample_data, f, indent=4)
    
    # Initialize and run the document organizer
    organizer = InsuranceDocumentOrganizer()
    
    # Process documents in a folder
    input_folder = "./input_docs"  # Change this to your input folder
    
    # Create input folder if it doesn't exist
    if not os.path.exists(input_folder):
        os.makedirs(input_folder)
        print(f"Created input folder: {input_folder}")
        print("Please place your PDF documents in this folder and run the script again.")
    else:
        organizer.process_documents(input_folder)