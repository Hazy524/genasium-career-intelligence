# Genasium – AI-Powered Career Intelligence System

## Overview
Genasium is an AI-driven job recommendation and career intelligence system that matches user resumes against job postings using a hybrid scoring approach:

- LLM-based matching
- Semantic similarity scoring
- Lexical TF-IDF scoring
- Market demand skill heatmap analysis

## Architecture
- Streamlit frontend
- Modular core logic (job source detection, trusted filtering)
- Chroma vector database
- Unit-tested source attribution logic

## Features
- Trusted-source job filtering (LinkedIn, Indeed, JobStreet, etc.)
- Smart hybrid match scoring
- Resume memory bank generation
- Global skill gap heatmap
- Modular architecture with unit tests

## Setup

1. Create virtual environment:
   ```bash
   python -m venv .venv
