# CV Model Security Testing Tool

A secure web-based tool for testing and pentesting computer vision models for vulnerabilities. Supports multiple model formats and performs defensive security testing including adversarial attacks, robustness testing, input fuzzing, and integrity checks.

## Features

- ✅ **Multi-Format Support**: PyTorch (.pt, .pth), ONNX (.onnx), Keras/TensorFlow (.h5), TensorFlow (.pb)
- ✅ **Adversarial Attack Testing**: FGSM and PGD attacks
- ✅ **Robustness Testing**: Noise resistance, perturbation analysis
- ✅ **Input Fuzzing**: Crash detection, anomaly detection
- ✅ **Model Integrity Checks**: File hash verification, backdoor scanning
- ✅ **Visualizations**: Before/after comparisons, metrics plots
- ✅ **PDF Reports**: Comprehensive security test reports
- ✅ **Ethical & Safe**: Defensive testing only, no actual exploitation
- ✅ **User Authentication**: Simulated authentication system
- ✅ **Logging**: Test execution logging

## Security Tests

### 1. Adversarial Attacks
- **FGSM (Fast Gradient Sign Method)**: Fast adversarial example generation
- **PGD (Projected Gradient Descent)**: Iterative adversarial attack
- Measures accuracy drop and perturbation magnitude

### 2. Robustness Testing
- Gaussian noise resistance
- Salt and pepper noise resistance
- Accuracy under various noise levels

### 3. Input Fuzzing
- Random input mutations
- Crash detection
- Anomaly detection (NaN, Inf values)

### 4. Model Integrity
- File hash verification (SHA256)
- Backdoor/trojan scanning
- Suspicious pattern detection

## Requirements

- Python 3.8+
- Modern web browser
- CPU-only execution (GPU not required)

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```
   
   Or use the run script:
   ```bash
   python run.py
   ```

3. **Open in browser**: The app will automatically open at `http://localhost:8501`

## Usage

### 1. Upload Model
- Navigate to "Upload Model" page
- Select your model file (.pt, .onnx, .h5, .pb)
- View model information and file hash

### 2. Run Tests
- Go to "Run Tests" page
- Select tests to run:
  - FGSM Adversarial Attack
  - PGD Adversarial Attack
  - Robustness Testing
  - Input Fuzzing
  - Model Integrity Check
- Adjust parameters (attack strength, sample count)
- Click "Run Selected Tests"

### 3. View Results
- Navigate to "View Results" page
- See visualizations:
  - Original vs adversarial image comparisons
  - Accuracy metrics
  - Robustness charts
  - Fuzzing statistics

### 4. Export Report
- Go to "Export Report" page
- Generate PDF report with all test results
- Download comprehensive security report

## Project Structure

```
cv_model_security/
├── src/
│   ├── __init__.py
│   ├── model_loader.py      # Model loading (PyTorch, ONNX, TF, Keras)
│   ├── security_tester.py    # Security test implementations
│   ├── report_generator.py  # PDF report generation
│   └── visualizer.py         # Plot generation
├── app.py                    # Streamlit main app
├── run.py                    # Entry point
├── requirements.txt
└── README.md
```

## Technical Details

### Model Loading
- **PyTorch**: State dict and checkpoint loading
- **ONNX**: ONNX Runtime inference
- **Keras/TensorFlow**: H5 model loading
- **TensorFlow**: SavedModel and frozen graph support

### Security Testing
- **ART Integration**: Adversarial Robustness Toolbox for attacks
- **CPU-Only**: All operations run on CPU (no GPU required)
- **Safe Execution**: No actual exploitation, only simulations

### Visualizations
- Matplotlib-based plots
- Before/after image comparisons
- Metrics visualization
- Robustness charts

## Ethical Considerations

⚠️ **Important**: 
- This tool is for **defensive security testing only**
- Use only on models you own or have permission to test
- No actual exploitation or malicious attacks are performed
- All tests are simulations for vulnerability identification
- Results are for remediation purposes only

## Limitations

- Model architecture knowledge required for some formats
- Some tests require model-specific implementations
- Adversarial attacks work best with supported frameworks (ONNX, Keras)
- Fuzzing is simplified (real fuzzing requires more sophisticated techniques)

## Troubleshooting

### Model Loading Fails
- Check model format compatibility
- Ensure model file is not corrupted
- Verify framework version compatibility
- Some PyTorch models require architecture definition

### Tests Fail
- Ensure model is properly loaded
- Check that model format is supported for specific test
- Some tests require specific model types (classification models)

### Performance Issues
- Tests run on CPU (may be slow for large models)
- Reduce number of test samples if needed
- Some operations are computationally intensive

## Future Enhancements

- [ ] Support for more model formats
- [ ] Additional attack types (C&W, DeepFool)
- [ ] Model architecture auto-detection
- [ ] API endpoint testing
- [ ] Database for test history
- [ ] Custom test script support
- [ ] Real-time monitoring
- [ ] Integration with CI/CD pipelines

## Security Notes

🔒 **Security Features**:
- User authentication simulation
- File integrity verification
- Safe test execution environment
- No network access during testing
- Isolated execution context

## License

This project is provided as-is for educational and defensive security testing purposes.

## Acknowledgments

- Adversarial Robustness Toolbox (ART)
- Streamlit for web framework
- OpenCV, PyTorch, TensorFlow communities

## Disclaimer

This tool is designed for legitimate security testing of computer vision models. Users are responsible for ensuring they have proper authorization before testing any models. The authors are not responsible for misuse of this tool.





