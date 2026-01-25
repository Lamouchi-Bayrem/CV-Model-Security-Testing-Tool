# 🔒 CV Model Security Testing Tool

![Security Testing](https://img.shields.io/badge/Security-Testing-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

A comprehensive web application for defensive security testing of Computer Vision models. Perform adversarial attacks, robustness testing, input fuzzing, and integrity checks on your ML models.

## 📋 Table of Contents
- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Supported Models](#-supported-models)
- [Security Tests](#-security-tests)
- [Report Generation](#-report-generation)
- [Project Structure](#-project-structure)
- [Screenshots](#-screenshots)
- [Ethical Disclaimer](#-ethical-disclaimer)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

### 🔍 **Model Analysis**
- **Multi-format Support**: Load PyTorch, TensorFlow, ONNX, Keras models
- **Model Information**: Extract input/output shapes, parameters, file hashes
- **Integrity Checks**: SHA256 verification, backdoor detection scanning

### 🧪 **Security Testing**
- **Adversarial Attacks**: FGSM & PGD attack simulations
- **Robustness Testing**: Gaussian noise, salt & pepper, contrast changes
- **Input Fuzzing**: Mutation-based fuzzing with crash detection
- **Comprehensive Metrics**: Accuracy drop, perturbation analysis, stability scores

### 📊 **Visualization & Reporting**
- **Interactive Visualizations**: Attack comparisons, robustness charts
- **Multiple Report Formats**: PDF, HTML, Text, and JSON
- **Professional Reports**: Executive summaries, detailed results, recommendations
- **Image Inclusion**: Visualizations embedded in all reports

### 🎯 **User Experience**
- **Modern UI**: Clean Streamlit interface with dark/light themes
- **Authentication**: Simple login system for access control
- **Real-time Progress**: Live test progress with visual feedback
- **Export Capabilities**: Download results and comprehensive reports

## 🎥 Demo

### Quick Demo Video
[![Demo Video](demo/demo_thumbnail.png)](demo/demo_video.mp4)

*Click the image above to watch the full demo video*

### Live Demo Steps:
1. **Upload Model** → Load your CV model (PyTorch, TensorFlow, ONNX, etc.)
2. **Configure Tests** → Select attack types and parameters
3. **Run Analysis** → Execute security tests with live progress
4. **View Results** → Interactive visualizations and metrics
5. **Export Report** → Generate professional PDF/HTML reports

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended

### Quick Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/cv-model-security-tester.git
cd cv-model-security-tester

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Manual Installation
```bash
# Core dependencies
pip install streamlit==1.28.0 numpy==1.24.3 pandas==2.1.3 pillow==10.1.0 matplotlib==3.7.3

# Report generation
pip install reportlab==3.6.12

# Visualization
pip install seaborn==0.12.2

# Optional ML frameworks (install based on your needs)
pip install torch==2.1.0
pip install tensorflow==2.13.0
pip install onnxruntime==1.16.1
```

## 📖 Usage

### Starting the Application
```bash
# Navigate to project directory
cd cv-model-security-tester

# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Run the application
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Default Credentials
- **Username**: `admin`
- **Password**: `admin`

*Note: In production, replace with proper authentication*

### Step-by-Step Guide

#### 1. **Authentication**
![Authentication](demo/screenshots/auth.png)
- Enter credentials to access the tool
- Demo credentials provided for testing

#### 2. **Upload Model**
![Upload Model](demo/screenshots/upload.png)
- Click "Browse files" or drag & drop model file
- Supported formats: `.pt`, `.pth`, `.onnx`, `.h5`, `.pb`
- View model information after successful load

#### 3. **Configure Tests**
![Test Configuration](demo/screenshots/tests.png)
- Select security tests to run:
  - ✅ FGSM Adversarial Attack
  - ✅ PGD Adversarial Attack  
  - ✅ Robustness Testing
  - ✅ Input Fuzzing
  - ✅ Model Integrity Check
- Adjust parameters:
  - Attack strength (ε): 0.01-0.5
  - Test samples: 1-100
  - Fuzzing samples: 10-1000

#### 4. **Run Tests**
![Running Tests](demo/screenshots/running.png)
- Click "🚀 Run Selected Tests"
- Watch real-time progress
- View immediate results for each test

#### 5. **Analyze Results**
![Results](demo/screenshots/results.png)
- **Summary Dashboard**: Pass/fail statistics
- **Detailed Analysis**: Per-test metrics and visualizations
- **Visual Comparisons**: Original vs adversarial images
- **Robustness Charts**: Noise test performance

#### 6. **Export Reports**
![Report Generation](demo/screenshots/report.png)
- Choose format: PDF, HTML, Text, or JSON
- Include visualizations in reports
- Download comprehensive security assessment

## 📁 Supported Models

### Model Formats
| Format | Extension | Framework | Status |
|--------|-----------|-----------|---------|
| PyTorch | `.pt`, `.pth` | PyTorch | ✅ Full Support |
| ONNX | `.onnx` | ONNX Runtime | ✅ Full Support |
| Keras/TF | `.h5` | TensorFlow/Keras | ✅ Full Support |
| TensorFlow | `.pb` | TensorFlow | ⚠️ Limited Support |
| SavedModel | Directory | TensorFlow | ⚠️ Limited Support |

### Model Requirements
- **Input Format**: Images (any size, normalized 0-1)
- **Output Format**: Classification probabilities
- **Memory**: < 2GB recommended for smooth operation
- **Compatibility**: CPU inference supported, GPU optional

## 🧪 Security Tests

### 1. **Adversarial Attacks**
#### FGSM (Fast Gradient Sign Method)
- **Purpose**: Test vulnerability to gradient-based attacks
- **Parameters**: Attack strength (ε), number of samples
- **Metrics**: Accuracy drop, perturbation magnitude

#### PGD (Projected Gradient Descent)
- **Purpose**: Test against iterative adversarial attacks
- **Parameters**: ε, iterations, step size
- **Metrics**: Iterative accuracy degradation

### 2. **Robustness Testing**
#### Noise Robustness
- Gaussian noise (0.01-0.2 variance)
- Salt & pepper noise (0.01-0.1 probability)
- Contrast changes (0.5-1.5 factor)

#### Performance Metrics
- Baseline accuracy
- Noise degradation curves
- Stability scores

### 3. **Input Fuzzing**
#### Mutation Strategies
- Random noise injection
- Brightness/contrast adjustments
- Scale transformations
- Edge case generation

#### Crash Detection
- Exception handling
- NaN/Inf value detection
- Memory usage monitoring

### 4. **Integrity Checks**
#### File Analysis
- SHA256 hash verification
- File size validation
- Format consistency checks

#### Security Scanning
- Basic backdoor pattern detection
- Suspicious layer identification
- Integrity scoring (0-100)

## 📄 Report Generation

### Report Types
| Format | Features | Best For |
|--------|----------|----------|
| **PDF** | Professional layout, embedded images | Formal reports, printing |
| **HTML** | Interactive, modern design, best image support | Web viewing, sharing |
| **Text** | Simple format, no images | Quick analysis, logs |
| **JSON** | Raw data, machine-readable | Further analysis, automation |

### Report Contents
1. **Cover Page**: Title, date, model information
2. **Executive Summary**: Key findings, pass/fail statistics
3. **Model Information**: Technical specifications
4. **Test Results**: Detailed metrics for each test
5. **Visualizations**: Charts and comparison images
6. **Recommendations**: Security improvement suggestions
7. **Appendices**: Raw data, configuration details

### Sample Reports
Check the `samples/` directory for example reports:
- `sample_report.pdf` - Professional PDF format
- `sample_report.html` - Interactive HTML report
- `sample_results.json` - Raw JSON data

## 📁 Project Structure

```
cv-model-security-tester/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── demo/                          # Demo assets
│   ├── demo_video.mp4             # Demo video
│   ├── demo_thumbnail.png         # Video thumbnail
│   └── screenshots/               # Application screenshots
│       ├── auth.png
│       ├── upload.png
│       ├── tests.png
│       ├── running.png
│       ├── results.png
│       └── report.png
├── samples/                       # Sample reports
│   ├── sample_report.pdf
│   ├── sample_report.html
│   └── sample_results.json
├── src/                           # Source code
│   ├── __init__.py
│   ├── model_loader.py           # Model loading and inference
│   ├── security_tester.py        # Security test implementations
│   ├── visualizer.py             # Plotting and visualization
│   └── report_generator.py       # PDF/HTML report generation
├── test_models/                   # Test model files (optional)
│   ├── sample_model.pt
│   ├── sample_model.onnx
│   └── sample_model.h5
├── uploads/                       # User upload directory
└── .gitignore
```

## 📸 Screenshots

### Application Interface
| Dashboard | Test Configuration | Results View |
|-----------|-------------------|--------------|
| ![Dashboard](demo/screenshots/dashboard.png) | ![Config](demo/screenshots/config.png) | ![Results](demo/screenshots/results_view.png) |

### Visualizations
| Attack Comparison | Robustness Analysis | Report Preview |
|-------------------|---------------------|----------------|
| ![Attack](demo/screenshots/attack_viz.png) | ![Robustness](demo/screenshots/robustness_viz.png) | ![Report](demo/screenshots/report_preview.png) |

### Report Examples
| PDF Report | HTML Report |
|------------|-------------|
| ![PDF](demo/screenshots/pdf_report.png) | ![HTML](demo/screenshots/html_report.png) |

## ⚠️ Ethical Disclaimer

**IMPORTANT**: This tool is designed for **DEFENSIVE SECURITY TESTING ONLY**.

### Authorized Use
- ✅ Testing your own models
- ✅ Testing models with explicit permission
- ✅ Educational and research purposes
- ✅ Security hardening and vulnerability assessment

### Prohibited Use
- ❌ Testing models without permission
- ❌ Malicious attacks on production systems
- ❌ Circumventing security measures
- ❌ Any illegal or unethical activities

### Security First
- All tests are simulations
- No actual exploitation performed
- Designed for vulnerability identification
- Intended for remediation and improvement

By using this tool, you agree to use it responsibly and ethically.

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Reporting Issues
1. Check existing issues
2. Use the issue template
3. Provide detailed reproduction steps
4. Include error messages and screenshots

### Feature Requests
1. Describe the feature clearly
2. Explain the use case
3. Suggest implementation approach
4. Consider if it aligns with project goals

### Code Contributions
1. Fork the repository
2. Create a feature branch
3. Write clear, documented code
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Fork and clone
git clone https://github.com/yourusername/cv-model-security-tester.git

# Create development branch
git checkout -b feature/your-feature-name

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Start development server
streamlit run app.py
```

### Areas for Contribution
- Additional attack implementations
- New model format support
- Enhanced visualizations
- Performance optimizations
- Documentation improvements
- Test coverage expansion

## 📊 Performance Considerations

### System Requirements
- **Minimum**: 4GB RAM, 2-core CPU
- **Recommended**: 8GB RAM, 4-core CPU
- **Storage**: 1GB free space

### Optimization Tips
1. **Use smaller test samples** (10-20) for quick testing
2. **Disable visualizations** if not needed
3. **Use HTML reports** for faster generation
4. **Close other applications** during large tests

### Known Limitations
- Large models (>500MB) may have slower loading
- Complex attacks (PGD with many iterations) can be CPU-intensive
- PDF generation with many images may be slow

## 🔧 Troubleshooting

### Common Issues

#### 1. **Model Loading Failures**
```
Error: Could not load model
```
**Solution:**
- Verify model format compatibility
- Check file integrity
- Ensure sufficient memory

#### 2. **ReportLab PDF Issues**
```
'usedforsecurity' is an invalid keyword argument
```
**Solution:**
```bash
pip uninstall reportlab -y
pip install reportlab==3.6.12
```

#### 3. **Missing Dependencies**
```
ModuleNotFoundError: No module named 'art'
```
**Solution:**
- The tool works without ART library
- Attacks are simulated if ART not available
- For real attacks: `pip install adversarial-robustness-toolbox`

#### 4. **Memory Errors**
```
MemoryError: Unable to allocate array
```
**Solution:**
- Reduce test sample size
- Use smaller models
- Close other applications
- Restart the tool

### Getting Help
1. Check the [Issues](https://github.com/yourusername/cv-model-security-tester/issues) page
2. Search for similar problems
3. Create a new issue with details
4. Include error messages and system info

## 📈 Future Development

### Planned Features
- [ ] **More Attack Types**: CW, DeepFool, AutoAttack
- [ ] **Defense Testing**: Adversarial training, defensive distillation
- [ ] **Batch Processing**: Test multiple models simultaneously
- [ ] **API Integration**: REST API for automation
- [ ] **Cloud Deployment**: Docker containers, cloud hosting
- [ ] **Advanced Analytics**: ML-based vulnerability prediction
- [ ] **Team Collaboration**: Multi-user, project management
- [ ] **Custom Test Creation**: User-defined test scripts

### Research Integration
- Latest adversarial attack papers
- State-of-the-art defense mechanisms
- Benchmark datasets and comparisons
- Academic paper reproducibility

## 📚 Learning Resources

### Related Papers
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) - FGSM
- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) - PGD
- [Practical Black-Box Attacks against Machine Learning](https://arxiv.org/abs/1602.02697)

### Tutorials and Courses
- [Adversarial Robustness - Theory and Practice](https://adversarial-ml-tutorial.org/)
- [MIT 6.S191: Introduction to Deep Learning](http://introtodeeplearning.com/)
- [Fast.ai: Practical Deep Learning for Coders](https://course.fast.ai/)

### Useful Tools
- [ART - Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
- [CleverHans - Adversarial Examples Library](https://github.com/cleverhans-lab/cleverhans)
- [Foolbox - Python toolbox to create adversarial examples](https://github.com/bethgelab/foolbox)

## 🏆 Acknowledgments

### Contributors
- [Your Name](https://github.com/yourusername) - Project Lead
- [Contributor Name](https://github.com/contributor) - Feature Contributor

### Technologies Used
- **Streamlit**: Web application framework
- **ReportLab**: PDF generation
- **Matplotlib/Seaborn**: Visualization
- **NumPy/Pandas**: Data processing
- **PyTorch/TensorFlow**: ML framework support

### Inspiration
- Academic research in adversarial machine learning
- Industry needs for ML security testing
- Open source security tools community

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 CV Model Security Testing Tool

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
```

## 🌐 Connect

- **GitHub**: [yourusername/cv-model-security-tester](https://github.com/yourusername/cv-model-security-tester)
- **Issues**: [Report Bugs & Features](https://github.com/yourusername/cv-model-security-tester/issues)
- **Email**: your.email@example.com
- **Twitter**: [@yourhandle](https://twitter.com/yourhandle)

---

**⭐ If you find this tool useful, please give it a star on GitHub!**

---

*Last updated: January 2024*  
*Version: 1.0.0*  
*Maintainer: Your Name*