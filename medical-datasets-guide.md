# Medical Datasets for Phi Text Generation Training

## Overview
This guide identifies high-quality medical datasets that are excellent for training Microsoft's Phi-4 model on medical text generation tasks. These datasets are available through Hugging Face's datasets library and are well-suited for instruction-tuning and fine-tuning.

## 🎯 Current Project Dataset
**Gaborandi/Alzheimer_pubmed_abstracts** - This is the dataset currently being used in the project for training Phi on Alzheimer's disease research abstracts.

## Top Recommended Datasets

### 1. **Alzheimer's Research Abstracts** - Current Project Dataset ⭐
- **Dataset ID**: `Gaborandi/Alzheimer_pubmed_abstracts`
- **Size**: Curated collection of Alzheimer's disease research abstracts
- **Format**: PubMed abstracts with titles, authors, and research content
- **Use Case**: Alzheimer's research text generation, medical literature synthesis
- **Quality**: High-quality, peer-reviewed research from PubMed
- **Status**: ✅ Currently in use for Phi training

### 2. **MedQA** - Medical Question Answering
- **Dataset ID**: `bigbio/med_qa`
- **Size**: ~12K medical questions with answers
- **Format**: Question-answer pairs from medical licensing exams
- **Use Case**: Medical Q&A generation, clinical reasoning
- **Quality**: High-quality, professionally curated medical content

### 2. **PubMed Abstracts** - Medical Literature
- **Dataset ID**: `pubmed_abstracts`
- **Size**: Millions of medical research abstracts
- **Format**: Structured abstracts with titles, authors, and text
- **Use Case**: Medical literature generation, research summary writing
- **Quality**: Peer-reviewed, scientific accuracy

### 3. **MIMIC-III** - Clinical Notes
- **Dataset ID**: `mimic-iii` (requires approval)
- **Size**: ~2M clinical notes from ICU patients
- **Format**: De-identified clinical documentation
- **Use Case**: Clinical note generation, medical documentation
- **Quality**: Real-world clinical data, comprehensive coverage

### 4. **MedNLI** - Medical Natural Language Inference
- **Dataset ID**: `bigbio/mednli`
- **Size**: ~14K sentence pairs with medical context
- **Format**: Premise-hypothesis pairs with medical reasoning
- **Use Case**: Medical text understanding, clinical reasoning
- **Quality**: Expert-annotated, diverse medical scenarios

### 5. **MedMCQA** - Medical Multiple Choice Questions
- **Dataset ID**: `bigbio/medmcqa`
- **Size**: ~194K medical questions with explanations
- **Format**: Multiple choice with detailed explanations
- **Use Case**: Medical education content generation
- **Quality**: Comprehensive coverage of medical topics

## Implementation Examples

### Basic Dataset Loading
```python
from datasets import load_dataset

# Load current project dataset - Alzheimer's research abstracts
alzheimer_dataset = load_dataset("Gaborandi/Alzheimer_pubmed_abstracts")

# Load MedQA dataset
medqa_dataset = load_dataset("bigbio/med_qa", "med_qa_en_source")

# Load PubMed abstracts
pubmed_dataset = load_dataset("pubmed_abstracts")

# Load MedNLI
mednli_dataset = load_dataset("bigbio/mednli", "mednli_bigbio_pe")
```

### Data Preprocessing for Phi

#### Alzheimer's Research Abstracts (Current Project)
```python
def format_alzheimer_abstract(example):
    """Format Alzheimer's research abstracts for Phi instruction tuning"""
    instruction = f"Summarize this Alzheimer's research: {example['title']}"
    response = example['abstract'] if 'abstract' in example else example['text']
    
    full_text = f"Instruction: {instruction}\nResponse: {response}\n\n"
    
    return {
        "text": full_text,
        "instruction": instruction,
        "response": response,
        "title": example['title']
    }

# Apply formatting to current dataset
formatted_alzheimer_dataset = alzheimer_dataset.map(format_alzheimer_abstract)
```

#### Medical Q&A Data
```python
def format_medical_qa(example):
    """Format medical Q&A data for Phi instruction tuning"""
    return {
        "text": f"Question: {example['question']}\nAnswer: {example['answer']}",
        "instruction": example['question'],
        "response": example['answer']
    }

# Apply formatting
formatted_dataset = medqa_dataset.map(format_medical_qa)
```

### Training Configuration

#### For Alzheimer's Research (Current Project)
```python
training_args = TrainingArguments(
    output_dir="phi4-alzheimer-research",
    learning_rate=1e-4,
    per_device_train_batch_size=4,  # Reduced for medical text complexity
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    warmup_steps=100,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    push_to_hub=True,
)
```

#### For General Medical Q&A
```python
training_args = TrainingArguments(
    output_dir="phi4-medical-qa",
    learning_rate=1e-4,
    per_device_train_batch_size=4,  # Reduced for medical text complexity
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    warmup_steps=100,
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=500,
    push_to_hub=True,
)
```

## Dataset-Specific Considerations

### Alzheimer's Research Abstracts (Current Project) ⭐
- **Strengths**: Specialized focus on Alzheimer's disease, high-quality research content, peer-reviewed sources
- **Challenges**: Limited to one disease area, technical research language
- **Best For**: Alzheimer's research text generation, medical literature synthesis, specialized medical AI applications
- **Unique Value**: Domain expertise in neurodegenerative diseases

### MedQA
- **Strengths**: High-quality medical content, diverse topics
- **Challenges**: May need additional context for complex cases
- **Best For**: General medical knowledge generation

### PubMed Abstracts
- **Strengths**: Large volume, scientific accuracy
- **Challenges**: Technical language, may need simplification
- **Best For**: Research writing, scientific communication

### MIMIC-III
- **Strengths**: Real clinical data, comprehensive coverage
- **Challenges**: Requires approval, complex medical terminology
- **Best For**: Clinical documentation, real-world applications

## Ethical and Privacy Considerations

1. **HIPAA Compliance**: Ensure all medical data is properly de-identified
2. **Data Privacy**: Never use personally identifiable information
3. **Medical Accuracy**: Validate generated content with medical professionals
4. **Bias Awareness**: Medical datasets may contain biases that need addressing

## Recommended Starting Point

### Current Project Focus ⭐
**Gaborandi/Alzheimer_pubmed_abstracts** is the primary dataset for this project, providing:
- Specialized Alzheimer's disease research content
- High-quality, peer-reviewed medical literature
- Structured format suitable for Phi instruction tuning
- Domain expertise in neurodegenerative diseases

### Alternative Starting Points
For beginners working on general medical tasks, start with **MedQA** as it provides:
- Clear question-answer format
- Manageable dataset size
- High-quality medical content
- Easy integration with existing Phi training pipeline

## Next Steps

### For Current Alzheimer's Project ⭐
1. **Dataset Exploration**: Examine the structure and content of the Alzheimer's abstracts
2. **Data Preprocessing**: Implement the Alzheimer's-specific formatting function
3. **Training Setup**: Configure training parameters for research abstract generation
4. **Domain Validation**: Ensure generated content maintains medical accuracy for Alzheimer's research
5. **Specialized Applications**: Focus on neurodegenerative disease research applications

### General Medical AI Development
1. Choose a dataset based on your specific use case
2. Implement the data preprocessing pipeline
3. Adapt your existing training code from the notebook
4. Monitor training metrics and medical accuracy
5. Validate results with domain experts

## Resources

- [Hugging Face Medical Datasets](https://huggingface.co/datasets?search=medical)
- [Gaborandi/Alzheimer_pubmed_abstracts](https://huggingface.co/datasets/Gaborandi/Alzheimer_pubmed_abstracts) - Current project dataset
- [BigBio Project](https://github.com/bigscience-workshop/biomedical) - Curated biomedical datasets
- [MIMIC-III Access](https://mimic.mit.edu/docs/gettingstarted/) - Clinical data access
- [PubMed Database](https://pubmed.ncbi.nlm.nih.gov/) - Source of medical research abstracts
