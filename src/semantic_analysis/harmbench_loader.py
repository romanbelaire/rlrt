"""
HarmBench data loader for semantic analysis data collection.
"""

import os
import pandas as pd
from typing import Optional, Dict, List, Any
from src.config import SemanticAnalysisConfig


def load_harmbench_data(config: SemanticAnalysisConfig) -> pd.DataFrame:
    """
    Load HarmBench data from CSV file.
    
    Args:
        config: Semantic analysis configuration
        
    Returns:
        DataFrame with HarmBench data including categories
    """
    print(f"Loading HarmBench data from: {config.harmbench_path}")
    
    if not os.path.exists(config.harmbench_path):
        raise FileNotFoundError(f"HarmBench data file not found: {config.harmbench_path}")
    
    # Load CSV data
    df = pd.read_csv(config.harmbench_path)
    print(f"Loaded {len(df)} entries from HarmBench")
    
    # Validate required columns
    required_columns = [config.question_column, config.category_column]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Required columns not found in HarmBench data: {missing_columns}")
    
    print(f"Using question column: '{config.question_column}'")
    print(f"Using category column: '{config.category_column}'")
    
    # Clean and preprocess data
    df = _preprocess_harmbench_data(df, config)
    
    return df


def _preprocess_harmbench_data(df: pd.DataFrame, config: SemanticAnalysisConfig) -> pd.DataFrame:
    """
    Preprocess HarmBench data for semantic analysis.
    
    Args:
        df: Raw HarmBench DataFrame
        config: Semantic analysis configuration
        
    Returns:
        Preprocessed DataFrame
    """
    print("Preprocessing HarmBench data...")
    
    # Remove rows with missing question data
    initial_count = len(df)
    df = df.dropna(subset=[config.question_column])
    print(f"Removed {initial_count - len(df)} rows with missing questions")
    
    # Clean question text
    df[config.question_column] = df[config.question_column].astype(str).str.strip()
    
    # Remove empty questions
    df = df[df[config.question_column].str.len() > 0]
    print(f"Removed {initial_count - len(df)} rows with empty questions")
    
    # Handle categories
    if config.category_column in df.columns:
        # Fill missing categories
        df[config.category_column] = df[config.category_column].fillna("unknown")
        
        # Clean category names
        df[config.category_column] = df[config.category_column].astype(str).str.strip()
        
        # Handle FunctionalCategory and SemanticCategory types
        # Check if categories contain these specific types
        category_values = df[config.category_column].unique()
        has_functional = any('FunctionalCategory' in str(cat) for cat in category_values)
        has_semantic = any('SemanticCategory' in str(cat) for cat in category_values)
        
        if has_functional or has_semantic:
            print("Detected FunctionalCategory and/or SemanticCategory types in data")
            
            # Create separate columns for category type and category value
            df['category_type'] = df[config.category_column].apply(lambda x: 
                'FunctionalCategory' if 'FunctionalCategory' in str(x) else 
                'SemanticCategory' if 'SemanticCategory' in str(x) else 
                'Unknown'
            )
            
            # Extract the actual category value (remove the type prefix)
            df['category_value'] = df[config.category_column].apply(lambda x:
                str(x).replace('FunctionalCategory:', '').replace('SemanticCategory:', '').strip()
                if 'FunctionalCategory:' in str(x) or 'SemanticCategory:' in str(x)
                else str(x)
            )
            
            print(f"Category type distribution:")
            type_counts = df['category_type'].value_counts()
            for cat_type, count in type_counts.items():
                print(f"  {cat_type}: {count}")
        
        # Show category distribution
        category_counts = df[config.category_column].value_counts()
        print(f"Category distribution:")
        for category, count in category_counts.head(10).items():
            print(f"  {category}: {count}")
        if len(category_counts) > 10:
            print(f"  ... and {len(category_counts) - 10} more categories")
    else:
        # Add default category column
        df[config.category_column] = "unknown"
        print("No category column found, using 'unknown' for all entries")
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=[config.question_column])
    print(f"Removed {initial_count - len(df)} duplicate questions")
    
    # Sort by category for better organization
    df = df.sort_values([config.category_column, config.question_column])
    
    print(f"Final dataset: {len(df)} entries")
    return df


def get_harmbench_statistics(df: pd.DataFrame, config: SemanticAnalysisConfig) -> Dict[str, Any]:
    """
    Get statistics about the HarmBench dataset.
    
    Args:
        df: HarmBench DataFrame
        config: Semantic analysis configuration
        
    Returns:
        Dictionary with dataset statistics
    """
    stats = {
        'total_entries': len(df),
        'unique_questions': df[config.question_column].nunique(),
        'duplicate_questions': len(df) - df[config.question_column].nunique(),
    }
    
    if config.category_column in df.columns:
        category_stats = df[config.category_column].value_counts()
        stats['total_categories'] = len(category_stats)
        stats['category_distribution'] = category_stats.to_dict()
        stats['most_common_category'] = category_stats.index[0]
        stats['least_common_category'] = category_stats.index[-1]
        
        # Add category type statistics if available
        if 'category_type' in df.columns:
            type_stats = df['category_type'].value_counts()
            stats['category_type_distribution'] = type_stats.to_dict()
            stats['total_category_types'] = len(type_stats)
            
            # Add statistics for each category type
            for cat_type in df['category_type'].unique():
                type_df = df[df['category_type'] == cat_type]
                if 'category_value' in type_df.columns:
                    value_stats = type_df['category_value'].value_counts()
                    stats[f'{cat_type.lower()}_distribution'] = value_stats.to_dict()
                    stats[f'{cat_type.lower()}_count'] = len(type_df)
    
    # Question length statistics
    question_lengths = df[config.question_column].str.len()
    stats['avg_question_length'] = question_lengths.mean()
    stats['min_question_length'] = question_lengths.min()
    stats['max_question_length'] = question_lengths.max()
    
    return stats


def filter_harmbench_data(df: pd.DataFrame, 
                         categories: Optional[List[str]] = None,
                         category_types: Optional[List[str]] = None,
                         max_questions: Optional[int] = None,
                         min_question_length: Optional[int] = None,
                         max_question_length: Optional[int] = None) -> pd.DataFrame:
    """
    Filter HarmBench data based on various criteria.
    
    Args:
        df: HarmBench DataFrame
        categories: List of categories to include (None for all)
        category_types: List of category types to include (e.g., ['FunctionalCategory', 'SemanticCategory'])
        max_questions: Maximum number of questions to include
        min_question_length: Minimum question length
        max_question_length: Maximum question length
        
    Returns:
        Filtered DataFrame
    """
    filtered_df = df.copy()
    initial_count = len(filtered_df)
    
    # Filter by category types (if category_type column exists)
    if category_types is not None and 'category_type' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['category_type'].isin(category_types)]
        print(f"Filtered by category types: {initial_count - len(filtered_df)} entries removed")
        initial_count = len(filtered_df)
    
    # Filter by categories
    if categories is not None:
        # Check if we have the new category structure
        if 'category_value' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['category_value'].isin(categories)]
        else:
            # Fall back to original category column
            filtered_df = filtered_df[filtered_df[config.category_column].isin(categories)]
        print(f"Filtered by categories: {initial_count - len(filtered_df)} entries removed")
        initial_count = len(filtered_df)
    
    # Filter by question length
    if min_question_length is not None:
        filtered_df = filtered_df[filtered_df['Behavior'].str.len() >= min_question_length]
        print(f"Filtered by min length: {initial_count - len(filtered_df)} entries removed")
        initial_count = len(filtered_df)
    
    if max_question_length is not None:
        filtered_df = filtered_df[filtered_df['Behavior'].str.len() <= max_question_length]
        print(f"Filtered by max length: {initial_count - len(filtered_df)} entries removed")
        initial_count = len(filtered_df)
    
    # Limit number of questions
    if max_questions is not None and len(filtered_df) > max_questions:
        filtered_df = filtered_df.head(max_questions)
        print(f"Limited to {max_questions} questions")
    
    print(f"Final filtered dataset: {len(filtered_df)} entries")
    return filtered_df


def sample_harmbench_data(df: pd.DataFrame, 
                         sample_size: int,
                         config: SemanticAnalysisConfig,
                         stratify_by_category: bool = True,
                         stratify_by_category_type: bool = False) -> pd.DataFrame:
    """
    Sample HarmBench data for analysis.
    
    Args:
        df: HarmBench DataFrame
        sample_size: Number of samples to take
        config: Semantic analysis configuration
        stratify_by_category: Whether to stratify sampling by category
        stratify_by_category_type: Whether to stratify sampling by category type (FunctionalCategory/SemanticCategory)
        
    Returns:
        Sampled DataFrame
    """
    if len(df) <= sample_size:
        print(f"Dataset size ({len(df)}) <= sample size ({sample_size}), returning full dataset")
        return df
    
    # Determine stratification column
    stratify_column = None
    if stratify_by_category_type and 'category_type' in df.columns:
        stratify_column = 'category_type'
        print("Stratifying by category type (FunctionalCategory/SemanticCategory)")
    elif stratify_by_category and config.category_column in df.columns:
        stratify_column = config.category_column
        print("Stratifying by category")
    
    if stratify_column is not None:
        # Stratified sampling
        def sample_group(group):
            return group.sample(n=min(len(group), max(1, int(sample_size * len(group) / len(df)))))
        
        sampled_df = df.groupby(stratify_column, group_keys=False).apply(sample_group)
        
        # If we got too many samples, take a random subset
        if len(sampled_df) > sample_size:
            sampled_df = sampled_df.sample(n=sample_size, random_state=42)
        elif len(sampled_df) < sample_size:
            # If we got too few samples, add more randomly
            remaining = df[~df.index.isin(sampled_df.index)]
            additional_needed = sample_size - len(sampled_df)
            if len(remaining) >= additional_needed:
                additional = remaining.sample(n=additional_needed, random_state=42)
                sampled_df = pd.concat([sampled_df, additional])
        
        print(f"Stratified sampling: {len(sampled_df)} samples from {len(df)} total")
    else:
        # Simple random sampling
        sampled_df = df.sample(n=sample_size, random_state=42)
        print(f"Random sampling: {len(sampled_df)} samples from {len(df)} total")
    
    return sampled_df 