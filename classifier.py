"""
Combined classifier module for toxicity detection.
Uses toxic-bert and zero-shot classification to detect inappropriate messages.
"""

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Classifier components (loaded conditionally)
zero_shot_classifier = None
toxic_tokenizer = None
toxic_model = None

# Define offensive categories for zero-shot
OFFENSIVE_CATEGORIES = [
    "chat message contains racism",
    "chat message contains sexism", 
    "chat message contains political opinion",
    "chat message contains insult",
    "chat message contains requests to add on social platforms"
]


def load_classifier_models(threshold: float = 0.75):
    """
    Load the combined classifier models (toxic-bert + zero-shot).
    
    Args:
        threshold: Classification threshold (default 0.75 = 75%)
    """
    global zero_shot_classifier, toxic_tokenizer, toxic_model
    
    try:
        print('Loading zero-shot classifier...')
        zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        print('Zero-shot classifier loaded!')
        
        print('Loading toxic-bert...')
        toxic_model_name = "unitary/toxic-bert"
        toxic_tokenizer = AutoTokenizer.from_pretrained(toxic_model_name)
        toxic_model = AutoModelForSequenceClassification.from_pretrained(toxic_model_name)
        toxic_model.eval()
        print(f'Toxic-bert loaded! (threshold: {threshold:.0%})')
    except Exception as e:
        print(f'Error loading classifier models: {e}')
        print('Classifier will be disabled for this session.')
        raise


def get_toxic_bert_score(text: str) -> tuple[float, str]:
    """
    Get the maximum toxicity score from toxic-bert.
    
    Args:
        text: Message text to analyze
        
    Returns:
        Tuple of (max_score, max_label)
    """
    if toxic_tokenizer is None or toxic_model is None:
        return 0.0, "model_not_loaded"
    
    inputs = toxic_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = toxic_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).numpy()[0]
    
    max_score = max(probs)
    max_label = toxic_model.config.id2label[probs.argmax()]
    
    return max_score, max_label


def get_zero_shot_score(text: str) -> tuple[float, str]:
    """
    Get the maximum confidence score from zero-shot classifier.
    
    Args:
        text: Message text to analyze
        
    Returns:
        Tuple of (max_score, max_label)
    """
    if zero_shot_classifier is None:
        return 0.0, "model_not_loaded"
    
    result = zero_shot_classifier(text, OFFENSIVE_CATEGORIES, multi_label=True)
    max_score = max(result['scores'])
    max_label = result['labels'][result['scores'].index(max_score)]
    
    return max_score, max_label


def classify_message(text: str, threshold: float = 0.75) -> dict:
    """
    Combine both classifiers and use the maximum confidence score.
    
    Args:
        text: Message text to classify
        threshold: Confidence threshold (default 0.75 = 75%)
    
    Returns:
        Dictionary with classification results including:
        - text: Original text
        - is_toxic: Boolean indicating if toxic
        - max_score: Maximum confidence score
        - max_label: Label from the model with highest score
        - model_used: Which model produced the max score
        - toxic_bert_score: Score from toxic-bert
        - toxic_bert_label: Label from toxic-bert
        - zero_shot_score: Score from zero-shot
        - zero_shot_label: Label from zero-shot
    """
    # Get scores from both models
    toxic_score, toxic_label = get_toxic_bert_score(text)
    zero_shot_score, zero_shot_label = get_zero_shot_score(text)
    
    # Use the maximum score
    if toxic_score >= zero_shot_score:
        max_score = toxic_score
        max_label = toxic_label
        model_used = "toxic-bert"
    else:
        max_score = zero_shot_score
        max_label = zero_shot_label
        model_used = "zero-shot"
    
    # Determine if toxic based on threshold
    is_toxic = max_score >= threshold
    
    return {
        'text': text,
        'is_toxic': bool(is_toxic),
        'max_score': float(max_score),
        'max_label': max_label,
        'model_used': model_used,
        'toxic_bert_score': float(toxic_score),
        'toxic_bert_label': toxic_label,
        'zero_shot_score': float(zero_shot_score),
        'zero_shot_label': zero_shot_label
    }


def is_classifier_loaded() -> bool:
    """
    Check if classifier models are loaded.
    
    Returns:
        True if models are loaded, False otherwise
    """
    return (zero_shot_classifier is not None and 
            toxic_tokenizer is not None and 
            toxic_model is not None)
