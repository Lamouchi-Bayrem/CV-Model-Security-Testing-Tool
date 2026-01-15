"""
Streamlit Web App for CV Model Security Testing
"""

import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import time
import os
from pathlib import Path
import tempfile


from src.model_loader import ModelLoader
from src.security_tester import SecurityTester
from src.report_generator import ReportGenerator
from src.visualizer import Visualizer

# Page configuration
st.set_page_config(
    page_title="CV Model Security Tester",
    page_icon="🔒",
    layout="wide"
)

# Initialize session state
if 'model_loader' not in st.session_state:
    st.session_state.model_loader = ModelLoader()
if 'security_tester' not in st.session_state:
    st.session_state.security_tester = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'test_results' not in st.session_state:
    st.session_state.test_results = {}
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False


def authenticate():
    """Simulate user authentication"""
    st.session_state.authenticated = True


def main():
    """Main application"""
    # Header with disclaimer
    st.title("🔒 CV Model Security Testing Tool")
    st.markdown("**Defensive Security Testing for Computer Vision Models**")
    
    # Ethical disclaimer
    st.warning(
        "⚠️ **ETHICAL DISCLAIMER**: This tool is for defensive security testing only. "
        "No actual exploitation or malicious attacks are performed. All tests are "
        "simulations designed to identify vulnerabilities for remediation purposes. "
        "Use responsibly and only on models you own or have permission to test."
    )
    
    # Authentication simulation
    if not st.session_state.authenticated:
        st.sidebar.header("🔐 Authentication")
        username = st.sidebar.text_input("Username", type="default")
        password = st.sidebar.text_input("Password", type="password")
        
        if st.sidebar.button("Login", type="primary"):
            if username and password:  # Simple check
                authenticate()
                st.sidebar.success("Authenticated")
                st.rerun()
            else:
                st.sidebar.error("Please enter credentials")
        
        st.info("Please authenticate to continue")
        return
    
    # Main content after authentication
    sidebar = st.sidebar
    
    sidebar.header("📋 Navigation")
    page = sidebar.radio("Select Page", ["Upload Model", "Run Tests", "View Results", "Export Report"])
    
    if page == "Upload Model":
        upload_model_page()
    elif page == "Run Tests":
        run_tests_page()
    elif page == "View Results":
        view_results_page()
    elif page == "Export Report":
        export_report_page()


def upload_model_page():
    """Model upload page"""
    st.header("📤 Upload Model")
    
    st.markdown("""
    ### Supported Formats
    - **PyTorch**: `.pt`, `.pth`
    - **ONNX**: `.onnx`
    - **Keras/TensorFlow**: `.h5`
    - **TensorFlow**: `.pb`, SavedModel directories
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a model file",
        type=['pt', 'pth', 'onnx', 'h5', 'pb'],
        help="Upload your CV model file for security testing"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Detect format
        file_ext = Path(uploaded_file.name).suffix.lower().lstrip('.')
        
        with st.spinner("Loading model..."):
            result = st.session_state.model_loader.load_model(tmp_path, file_ext)
        
        if result['success']:
            st.session_state.model_loaded = True
            st.session_state.security_tester = SecurityTester(st.session_state.model_loader)
            st.session_state.model_info = result['info']
            
            st.success("✅ Model loaded successfully!")
            
            # Display model information
            st.subheader("Model Information")
            info = result['info']
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Format", info.get('format', 'Unknown'))
                st.metric("Framework", info.get('framework', 'Unknown'))
                st.metric("File Size", f"{info.get('file_size', 0) / 1024 / 1024:.2f} MB")
            
            with col2:
                st.metric("Input Shape", str(info.get('input_shape', 'Unknown')))
                st.metric("Output Shape", str(info.get('output_shape', 'Unknown')))
                st.metric("Parameters", str(info.get('num_parameters', 'Unknown')))
            
            # File hash
            st.text(f"File Hash (SHA256): {info.get('file_hash', 'Unknown')}")
            
            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except:
                pass
        else:
            st.error(f"❌ Failed to load model: {result.get('error', 'Unknown error')}")
    else:
        st.info("👆 Please upload a model file to begin")


def run_tests_page():
    """Run security tests page"""
    st.header("🧪 Run Security Tests")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please upload a model first")
        return
    
    st.markdown("### Select Tests to Run")
    
    # Test selection
    col1, col2 = st.columns(2)
    
    with col1:
        test_fgsm = st.checkbox("FGSM Adversarial Attack", value=False)
        test_pgd = st.checkbox("PGD Adversarial Attack", value=False)
        test_robustness = st.checkbox("Robustness Testing", value=False)
    
    with col2:
        test_fuzzing = st.checkbox("Input Fuzzing", value=False)
        test_integrity = st.checkbox("Model Integrity Check", value=True)
    
    # Test parameters
    st.markdown("### Test Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        eps = st.slider("Attack Strength (ε)", 0.01, 0.5, 0.1, 0.01)
    with col2:
        num_test_samples = st.number_input("Test Samples", min_value=1, max_value=100, value=10)
    with col3:
        fuzz_samples = st.number_input("Fuzzing Samples", min_value=10, max_value=1000, value=100)
    
    # Run tests button
    if st.button("🚀 Run Selected Tests", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        test_results = {}
        total_tests = sum([test_fgsm, test_pgd, test_robustness, test_fuzzing, test_integrity])
        current_test = 0
        
        # Generate dummy test data (in real app, user would provide)
        test_images = np.random.rand(num_test_samples, 28, 28, 1).astype(np.float32)
        test_labels = np.random.randint(0, 10, num_test_samples)
        
        # FGSM Attack
        if test_fgsm:
            current_test += 1
            status_text.text(f"Running FGSM attack ({current_test}/{total_tests})...")
            progress_bar.progress(current_test / total_tests)
            result = st.session_state.security_tester.run_adversarial_attack(
                'fgsm', test_images, test_labels, eps=eps
            )
            test_results['fgsm_attack'] = result
            time.sleep(0.5)
        
        # PGD Attack
        if test_pgd:
            current_test += 1
            status_text.text(f"Running PGD attack ({current_test}/{total_tests})...")
            progress_bar.progress(current_test / total_tests)
            result = st.session_state.security_tester.run_adversarial_attack(
                'pgd', test_images, test_labels, eps=eps
            )
            test_results['pgd_attack'] = result
            time.sleep(0.5)
        
        # Robustness Testing
        if test_robustness:
            current_test += 1
            status_text.text(f"Testing robustness ({current_test}/{total_tests})...")
            progress_bar.progress(current_test / total_tests)
            result = st.session_state.security_tester.test_robustness(test_images, test_labels)
            test_results['robustness'] = result
            time.sleep(0.5)
        
        # Input Fuzzing
        if test_fuzzing:
            current_test += 1
            status_text.text(f"Fuzzing inputs ({current_test}/{total_tests})...")
            progress_bar.progress(current_test / total_tests)
            base_image = test_images[0]
            result = st.session_state.security_tester.fuzz_inputs(base_image, num_samples=fuzz_samples)
            test_results['fuzzing'] = result
            time.sleep(0.5)
        
        # Integrity Check
        if test_integrity:
            current_test += 1
            status_text.text(f"Checking integrity ({current_test}/{total_tests})...")
            progress_bar.progress(current_test / total_tests)
            result = st.session_state.security_tester.check_model_integrity()
            test_results['integrity'] = result
            time.sleep(0.5)
        if deepfool:
            with st.spinner("Running DeepFool attack..."):
                results = security_tester.run_adversarial_attack('deepfool', test_images, test_labels, eps=eps)
                st.session_state.test_results['DeepFool'] = results

        if cwl2:
            with st.spinner("Running Carlini & Wagner L2 attack..."):
                results = security_tester.run_adversarial_attack('cwl2', test_images, test_labels, eps=eps)
                st.session_state.test_results['CarliniWagnerL2'] = results
        
        # Store results
        st.session_state.test_results = test_results
        
        progress_bar.progress(1.0)
        status_text.text("✅ All tests completed!")
        
        st.success("Tests completed successfully!")
        st.balloons()

        # Adversarial attacks
        fgsm = st.checkbox("FGSM Adversarial Attack", value=True)
        pgd = st.checkbox("PGD Adversarial Attack", value=True)
        deepfool = st.checkbox("DeepFool Attack", value=False)
        cwl2 = st.checkbox("Carlini & Wagner L2 Attack", value=False)


def view_results_page():
    """View test results page"""
    st.header("📊 Test Results")
    
    if not st.session_state.test_results:
        st.info("No test results available. Please run tests first.")
        return
    
    visualizer = Visualizer()
    
    # Display results for each test
    for test_name, result in st.session_state.test_results.items():
        st.markdown(f"### {test_name.replace('_', ' ').title()}")
        
        if not result.get('success', False):
            st.error(f"Test failed: {result.get('error', 'Unknown error')}")
            continue
        
        # Adversarial attack results
        if 'adversarial_accuracy' in result:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Visualization
                if 'adversarial_images' in result:
                    plot = visualizer.plot_adversarial_comparison(
                        np.random.rand(5, 28, 28, 1),  # Placeholder
                        result['adversarial_images'][:5] if len(result['adversarial_images']) >= 5 else result['adversarial_images'],
                        result['perturbations'][:5] if len(result['perturbations']) >= 5 else result['perturbations']
                    )
                    if plot:
                        st.image(plot, use_container_width=True)
            
            with col2:
                st.metric("Original Accuracy", f"{result['original_accuracy']:.2%}")
                st.metric("Adversarial Accuracy", f"{result['adversarial_accuracy']:.2%}")
                st.metric("Accuracy Drop", f"{result['accuracy_drop']:.2%}", 
                         delta=f"-{result['accuracy_drop']:.2%}")
                st.metric("Attack Time", f"{result['attack_time']:.2f}s")
            
            # Metrics plot
            metrics_plot = visualizer.plot_attack_metrics(result)
            if metrics_plot:
                st.image(metrics_plot, use_container_width=True)
        
        # Robustness results
        elif 'baseline_accuracy' in result:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                robustness_plot = visualizer.plot_robustness_results(result)
                if robustness_plot:
                    st.image(robustness_plot, use_container_width=True)
            
            with col2:
                st.metric("Baseline Accuracy", f"{result['baseline_accuracy']:.2%}")
                st.json(result.get('noise_tests', {}))
        
        # Fuzzing results
        elif 'crash_rate' in result:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", result['num_samples'])
            with col2:
                st.metric("Crashes", result['crashes'], delta=f"{result['crash_rate']:.2%}")
            with col3:
                st.metric("Anomalies", result['anomalies'], delta=f"{result['anomaly_rate']:.2%}")
        
        # Integrity check
        elif 'file_hash' in result:
            st.json(result)
        
        st.divider()


def export_report_page():
    """Export report page"""
    st.header("📄 Export Report")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please upload a model first")
        return
    
    if not st.session_state.test_results:
        st.warning("⚠️ Please run tests first")
        return
    
    st.markdown("### Generate PDF Report")
    
    report_generator = ReportGenerator()
    
    if st.button("📥 Generate and Download Report", type="primary"):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            report_path = tmp_file.name
        
        with st.spinner("Generating report..."):
            report_path = report_generator.generate_report(
                st.session_state.model_info,
                st.session_state.test_results,
                report_path
            )
        
        # Read PDF and provide download
        with open(report_path, 'rb') as f:
            pdf_data = f.read()
        
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_data,
            file_name=f"security_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
        
        # Cleanup
        try:
            os.unlink(report_path)
        except:
            pass


if __name__ == "__main__":
    main()










