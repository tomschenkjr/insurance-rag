# Insurance Document Classification System

This project implements a machine learning-based system for automatically classifying and organizing insurance-related PDF documents. The system uses a combination of text extraction, TF-IDF vectorization, and SVM classification to categorize documents into specific insurance types and subcategories.

## Prerequisites

- Python 3.8 or higher
- Ollama installed on your system
- Sufficient disk space and RAM (DeepSeek-R1:14B is a large model)
- At least 16GB RAM recommended for optimal performance

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd insurance-rag
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

- `classify-and-move.py`: Main script for document classification and organization
- `training_data.json`: Training data for the classification model
- `input_docs/`: Directory for placing PDF documents to be processed
- `organized_insurance_docs/`: Output directory for classified documents

## Usage

### 1. Prepare Training Data

The system requires training data to learn how to classify documents. The training data should be organized in the following structure:

```
training_data/
├── Health_insurance/
│   ├── document1.pdf
│   └── ...
├── Cell_phone_insurance/
│   └── ...
├── Condo_insurance/
│   └── ...
├── Umbrella_insurance/
│   └── ...
├── Business_travel_insurance/
│   └── ...
├── Watch_insurance/
│   └── ...
├── Life_insurance/
│   └── ...
├── Dental_insurance/
│   └── ...
├── Disability_insurance/
│   └── ...
├── Vision_insurance/
│   └── ...
└── Credit_Cards/
    ├── Chase/
    │   ├── Sapphire_Reserve/
    │   ├── Freedom_Unlimited/
    │   └── Freedom_Flex/
    ├── American_Express/
    │   ├── Platinum/
    │   ├── Gold/
    │   └── Green/
    ├── Citi/
    │   ├── Double_Cash_Card/
    │   └── Custom_Cash/
    ├── Discover/
    │   ├── It/
    │   └── Miles/
    └── Capital_One/
        ├── Venture/
        └── Quicksilver/
```

### 2. Run the Classification Script

To process and organize documents:

```bash
python classify-and-move.py
```

The script will:
1. Check for training data
2. Create necessary directories if they don't exist
3. Process PDF files in the `input_docs` folder
4. Organize documents into appropriate categories in `organized_insurance_docs`

## Document Organization

Documents are organized by:
1. Main insurance category
2. For credit cards: issuer and specific card type
3. Year (Current and previous 3 years)

Example output structure:
```
organized_insurance_docs/
├── health/
│   ├── Current/
│   ├── 2024/
│   ├── 2023/
│   └── 2022/
├── credit_card/
│   ├── chase/
│   │   ├── sapphire_reserve/
│   │   │   ├── Current/
│   │   │   ├── 2024/
│   │   │   ├── 2023/
│   │   │   └── 2022/
│   │   └── ...
│   └── ...
└── ...
```

## Features

- Automatic text extraction from PDF documents
- Machine learning-based classification
- Special handling for credit card documents
- Year-based organization
- Duplicate file handling
- Detailed processing logs

## Customization

You can customize the classification by modifying:
1. Categories in the `categories` list
2. Credit card issuers and card types in `credit_card_issuers`
3. Training data structure
4. Model parameters (TF-IDF and SVM settings)

## License

MIT License

Copyright (c) 2024 Tom Schenk Jr

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. 