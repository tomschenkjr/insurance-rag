# Home

# Purpose

The purpose of this project is to automatically organize and file a variety of insurance and benefit plan documents into structured folders on a computer using AI. The program should analyze documents, categorize them based on type, and file them into an appropriate folder for the document type and year.

```mermaid
graph LR;
    A[Create SVM model] --> B[Read PDF policy docs];
    B --> C[Categorize documents];
    C --> D[Health insurance];
    C --> E[Condo insurance];
    C --> F[Umbrella insurance];
    C --> G[Credit Cards];
    G --> H[Credit Chase Sapphire Reserve];
    G --> I[American Express Platinum];
    D --> J[Current];
    D --> K[2024];
    E --> L[Current];
    E --> M[2024];
```