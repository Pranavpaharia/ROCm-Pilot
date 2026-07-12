#!/usr/bin/env python3
"""
ROCm-Pilot Evaluation Harness - Evaluation Script
This script evaluates the performance of ROCm-Pilot system responses
against golden Q&A pairs.
"""

import json
import sys
import os

class EvaluationHarness:
    """Main evaluation harness class for ROCm-Pilot"""
    
    def __init__(self, dataset_path: str = "golden_qa_dataset.json"):
        self.dataset_path = dataset_path
        self.dataset = self._load_dataset()
        
    def _load_dataset(self):
        """Load the golden Q&A dataset"""
        if not os.path.exists(self.dataset_path):
            print(f"Error: Dataset file {self.dataset_path} not found.")
            return []
        try:
            with open(self.dataset_path, 'r') as f:
                data = json.load(f)
                return data.get('dataset', [])
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {self.dataset_path}")
            return []

    def mock_system_query(self, question: str) -> str:
        """
        Mock function to represent the ROCm-Pilot answering a question.
        In the real system, this would call the actual ROCm-Pilot API/CLI.
        """
        # For testing, we just return a placeholder or slightly modified expected answer.
        return f"Mock answer for: {question}"

    def calculate_fact_presence(self, actual_answer: str, facts: list) -> float:
        """Calculate what percentage of ground truth facts appear in the answer."""
        if not facts:
            return 1.0
        
        actual_lower = actual_answer.lower()
        found_facts = 0
        
        for fact in facts:
            # Simple keyword matching (in a real system, you'd use an LLM-as-a-judge here)
            keywords = [word for word in fact.lower().split() if len(word) > 3]
            if any(kw in actual_lower for kw in keywords):
                found_facts += 1
                
        return found_facts / len(facts)

    def evaluate_all(self):
        print(f"Starting evaluation of {len(self.dataset)} questions...\n")
        
        total_score = 0
        
        for i, qa in enumerate(self.dataset):
            question = qa["question"]
            facts = qa.get("ground_truth_facts", [])
            
            print(f"Q{i+1}: {question}")
            actual_answer = self.mock_system_query(question)
            
            fact_score = self.calculate_fact_presence(actual_answer, facts)
            
            print(f"  Fact Presence Score: {fact_score * 100:.1f}%")
            total_score += fact_score
            
        avg_score = (total_score / len(self.dataset)) * 100 if self.dataset else 0
        print(f"\n--- Final Results ---")
        print(f"Average Fact Presence: {avg_score:.1f}%")

if __name__ == '__main__':
    harness = EvaluationHarness()
    if harness.dataset:
        harness.evaluate_all()
    else:
        print("No data to evaluate. Please generate the dataset first.")