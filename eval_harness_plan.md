# ROCm-Pilot Evaluation Harness - Implementation Plan

## Overview
This document outlines the implementation plan for creating the evaluation harness for ROCm-Pilot, focusing on Days 1-2 of the roadmap. The evaluation harness will consist of a dataset of 50 golden Q&A pairs and an evaluation script to measure answer quality, retrieval hit-rate, and fact presence.

## Objectives
- Create 50 golden Q&A pairs covering key ROCm setup scenarios
- Develop evaluation script to score system responses
- Implement metrics for answer quality assessment
- Establish testing framework for validation

## Implementation Components

### 1. Golden Q&A Dataset Creation

#### Categories of Questions
- ROCm installation procedures (step-by-step guides)
- Matrix operations and GPU compatibility (ROCm vs CUDA)
- vLLM setup and configuration (LLM inference on AMD GPUs)
- Radeon vs Instinct GPU comparisons (performance characteristics)
- CUDA to ROCm migration guidance (code adaptation)

#### Dataset Structure
Each Q&A entry will include:
- Question text (clear and specific)
- Expected answer content (comprehensive and accurate)
- Ground truth facts (verifiable information)
- Category tags (for categorization and analysis)

### 2. Evaluation Script Development

#### Core Functionality
- Process Q&A dataset
- Query ROCm-Pilot system with each question
- Score answers based on:
  - Retrieval hit-rate (relevant document finding)
  - Fact presence (correctness of information)
  - Answer relevance (completeness and accuracy)

#### Scoring Metrics
- **Retrieval Hit-Rate**: Percentage of relevant documents found
- **Fact Presence**: Percentage of correct facts included in response
- **Answer Relevance**: How well the answer addresses the question
- **Completeness**: Coverage of all key aspects of the question

### 3. Testing Framework

#### Local Testing Setup
- Mock environment for testing without external dependencies
- Hardware-agnostic test cases
- CI-friendly unit tests

#### Remote Testing Considerations
- SSH connection testing capability
- Remote system validation methods
- Environment configuration for remote testing

## Remote Testing Strategy

### Challenge
The repository is currently on a MacBook, but the evaluation needs to be tested on actual ROCm systems with AMD GPUs.

### Proposed Solutions

#### 1. SSH Remote Testing Framework
- Create SSH connection manager for remote system access
- Implement remote command execution capabilities
- Set up secure credential handling for remote connections

#### 2. Mock Testing Environment
- Develop comprehensive mock system that simulates ROCm environment
- Create mock GPU detection and system information
- Implement mock responses for ROCm-specific commands

#### 3. Remote System Integration
- Configure remote testing infrastructure (e.g., AMD GPU servers)
- Set up CI/CD pipeline that can run tests on remote systems
- Create test scripts that can be executed remotely

### Implementation Approach
1. **Initial Development**: Create evaluation harness locally with mock capabilities
2. **Remote Testing Setup**: Configure SSH connection and remote execution
3. **Validation**: Run tests on actual ROCm systems when available

## Deliverables

### Phase 1 (Days 1-2)
- Complete 50 golden Q&A dataset (50 questions with expected answers)
- Evaluation script with scoring capabilities
- Basic testing framework with mock environment
- Documentation for running evaluations

### Phase 2 (Post-Week 1)
- Remote testing integration
- Actual ROCm system evaluation
- Performance metrics and analysis

## Technical Requirements

### Dependencies
- Python 3.8+
- Testing frameworks (pytest, unittest)
- Mock libraries for testing
- SSH libraries for remote connections (paramiko)

### File Structure
```
eval_harness/
├── golden_qa_dataset.json
├── evaluation_script.py
├── test_framework/
│   ├── mock_system.py
│   └── remote_connection.py
├── test_results/
└── documentation/
    └── eval_harness_documentation.md
```

## Risk Mitigation

### Remote Testing Limitations
- Develop comprehensive local mock testing
- Create detailed documentation for remote setup
- Implement fallback testing methods

### Data Accuracy
- Validate all Q&A content with official ROCm documentation
- Include multiple sources for verification
- Establish review process for content accuracy

## Timeline
- **Day 1**: Dataset creation and basic evaluation script
- **Day 2**: Complete evaluation framework and testing
- **Post-Day 2**: Remote testing integration and validation

## Success Criteria
- 50 high-quality Q&A pairs covering key ROCm topics
- Evaluation script that provides meaningful metrics
- Testable framework that can be run locally and remotely
- Documentation for both local and remote usage