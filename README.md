# QDA Local – Fully-automated, local QDA

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A fully automated Qualitative Data Analysis (QDA) pipeline that runs locally with Docker. The system processes text data through a comprehensive pipeline: Ingestion → Segmentation → Embeddings/TF-IDF → Clustering → Rule-based Auto-Codes → NER/Sentiment → LLM-free summaries → Reports & REFI-QDA export.

## 🚀 Quick Start

### Prerequisites
- Docker Engine 20.10.0+
- Docker Compose 2.0.0+
- (Optional) NVIDIA Container Toolkit for GPU acceleration

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/qda_local.git
cd qda_local
```

### 2. Set up environment variables
```bash
cp .env.example .env
# Edit .env file if you need to customize paths or resource limits
```

### 3. Start the services
```bash
docker compose up --build -d
```

### 4. Run an analysis
```bash
# Place your input CSV in the data directory
# Then trigger the analysis
curl -X POST "http://localhost:8009/analyze?input_path=/app/data/input.csv&out_dir=/app/out"
```

## 📂 Project Structure

```
qda_local/
├── backend/            # FastAPI backend service
├── worker/             # Background worker for processing
├── data/               # Input data directory (mounted volume)
│   └── input.csv       # Example input file
├── out/                # Output directory (mounted volume)
├── config/             # Configuration files
├── docker-compose.yml  # Docker Compose configuration
└── .env.example       # Example environment variables
```

## 🛠️ Configuration

### Environment Variables
Edit the `.env` file to customize:
- Cache locations
- Resource limits
- Logging levels
- Service ports

### Input Format
Input should be a CSV file with the following structure:
```csv
id,text,question_id,question_text
1,"Sample response text...",1,"What is your opinion?"
```

## 📊 Outputs

Analysis results are saved in the `out/` directory with the following structure:

```
out/
├── reports/                   # Generated reports
│   ├── overall_summary.md     # Summary of all data
│   ├── question_reports/      # Per-question analysis
│   └── code_reports/          # Detailed code analysis
├── visualizations/            # Charts and graphs
├── exports/                   # Standard QDA exports
│   ├── maxqda_import.csv      # MAXQDA import format
│   ├── atlasti_import.csv     # ATLAS.ti import format
│   └── qda_export.qdpx       # REFI-QDA project
└── intermediate/              # Intermediate processing files
```

## 🔧 Advanced Usage

### Monitoring Services
- **API Docs**: http://localhost:8009/docs
- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Ollama API**: http://localhost:11434

### Resource Management
To adjust resource limits, uncomment and modify the relevant sections in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 16G
      # For GPU support
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Persistent Caching
Model files are cached in the `.cache/` directory by default. To clear the cache:

```bash
docker compose down -v
rm -rf .cache/*
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ for qualitative researchers
- Uses [Qdrant](https://qdrant.tech/) for vector search
- [Ollama](https://ollama.ai/) for local LLM support
- [spaCy](https://spacy.io/) for NLP processing
