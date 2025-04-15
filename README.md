# Multimodal Search System

A production-ready multimodal search system that combines image and text understanding for e-commerce applications.

## Features

- Text and image-based search
- State-of-the-art deep learning models (ViT & BERT)
- Late fusion architecture
- AWS SageMaker deployment
- FastAPI backend
- Streamlit frontend
- FAISS vector store

## Quick Start

1. Clone the repository
```bash
git clone https://github.com/YingzuoLiu/multimodal-search.git
cd multimodal-search
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the application
```bash
# Start API server
python src/api/main.py

# Start frontend (in another terminal)
streamlit run src/frontend/app.py
```

## Project Structure

```
multimodal-search/
├── src/               # Source code
│   ├── api/          # FastAPI backend
│   ├── models/       # ML models
│   ├── utils/        # Utilities
│   └── frontend/     # Streamlit frontend
├── scripts/          # Deployment scripts
├── tests/            # Test files
├── docs/             # Documentation
└── requirements.txt  # Dependencies
```

## Documentation

See [Technical Documentation](docs/technical_documentation.md) for detailed information about the system architecture and deployment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
