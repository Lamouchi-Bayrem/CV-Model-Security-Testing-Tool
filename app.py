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
import json
import traceback

from src.model_loader import ModelLoader
from src.security_tester import SecurityTester
from src.report_generator import PDFReportGenerator, HTMLReportGenerator
from src.visualizer import Visualizer


# Page configuration
st.set_page_config(
    page_title="CV Model Security Tester",
    page_icon="🔒",
    layout="wide"
)

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
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
    if 'test_images' not in st.session_state:
        st.session_state.test_images = None
    if 'test_labels' not in st.session_state:
        st.session_state.test_labels = None
    if 'model_info' not in st.session_state:
        st.session_state.model_info = {}
    if 'visualizer' not in st.session_state:
        st.session_state.visualizer = Visualizer()

init_session_state()


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
    
    # Authentication
    if not st.session_state.authenticated:
        show_authentication()
        return
    
    # Main content
    show_main_content()


def show_authentication():
    """Show authentication page"""
    st.sidebar.header("🔐 Authentication")
    
    username = st.sidebar.text_input("Username", value="admin")
    password = st.sidebar.text_input("Password", type="password", value="admin")
    
    if st.sidebar.button("Login", type="primary"):
        if username and password:
            st.session_state.authenticated = True
            st.sidebar.success("Authenticated")
            st.rerun()
        else:
            st.sidebar.error("Please enter credentials")
    
    st.info("👆 Please authenticate to continue")


def show_main_content():
    """Show main application content"""
    # Sidebar navigation
    st.sidebar.header("📋 Navigation")
    
    page = st.sidebar.radio(
        "Select Page",
        ["📤 Upload Model", "🧪 Run Tests", "📊 View Results", "📄 Export Report"]
    )
    
    # Logout button
    st.sidebar.divider()
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()
    
    # Show selected page
    if page == "📤 Upload Model":
        upload_model_page()
    elif page == "🧪 Run Tests":
        run_tests_page()
    elif page == "📊 View Results":
        view_results_page()
    elif page == "📄 Export Report":
        export_report_page()


def upload_model_page():
    """Model upload page"""
    st.header("📤 Upload Model")
    
    st.markdown("""
    ### Supported Formats
    - **PyTorch**: `.pt`, `.pth`
    - **ONNX**: `.onnx`
    - **Keras/TensorFlow**: `.h5`
    - **TensorFlow**: `.pb`
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a model file",
        type=['pt', 'pth', 'onnx', 'h5', 'pb'],
        help="Upload your CV model file for security testing"
    )
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        file_ext = Path(uploaded_file.name).suffix.lower().lstrip('.')
        
        with st.spinner("Loading model..."):
            try:
                result = st.session_state.model_loader.load_model(tmp_path, file_ext)
                
                if result['success']:
                    st.session_state.model_loaded = True
                    st.session_state.security_tester = SecurityTester(st.session_state.model_loader)
                    st.session_state.model_info = result['info']
                    
                    st.success("✅ Model loaded successfully!")
                    show_model_info(result['info'])
                    
                else:
                    st.error(f"❌ Failed to load model: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"❌ Error loading model: {str(e)}")
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    else:
        st.info("👆 Please upload a model file to begin")


def show_model_info(model_info):
    """Display model information"""
    st.subheader("Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Format", model_info.get('format', 'Unknown'))
        st.metric("Framework", model_info.get('framework', 'Unknown'))
        st.metric("File Size", f"{model_info.get('file_size', 0) / 1024 / 1024:.2f} MB")
    
    with col2:
        st.metric("Input Shape", str(model_info.get('input_shape', 'Unknown')))
        st.metric("Output Shape", str(model_info.get('output_shape', 'Unknown')))
        st.metric("Parameters", str(model_info.get('num_parameters', 'Unknown')))
    
    file_hash = model_info.get('file_hash', 'Unknown')
    st.text_input("File Hash (SHA256)", value=file_hash, disabled=True)


def run_tests_page():
    """Run security tests page"""
    st.header("🧪 Run Security Tests")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please upload a model first")
        return
    
    # Test selection
    st.markdown("### Select Tests to Run")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_fgsm = st.checkbox("FGSM Adversarial Attack", value=True)
        test_pgd = st.checkbox("PGD Adversarial Attack", value=True)
        test_robustness = st.checkbox("Robustness Testing", value=True)
    
    with col2:
        test_fuzzing = st.checkbox("Input Fuzzing", value=True)
        test_integrity = st.checkbox("Model Integrity Check", value=True)
    
    # Test parameters
    st.markdown("### Test Parameters")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        eps = st.slider("Attack Strength (ε)", 0.01, 0.5, 0.1, 0.01)
    with col2:
        num_test_samples = st.number_input("Test Samples", min_value=1, max_value=50, value=10)
    with col3:
        fuzz_samples = st.number_input("Fuzzing Samples", min_value=10, max_value=500, value=100)
    
    # Visualization options
    with st.expander("🎨 Visualization Options"):
        generate_plots = st.checkbox("Generate visualizations for reports", value=True)
        plot_quality = st.slider("Plot quality", 1, 10, 5)
    
    # Run tests button
    if st.button("🚀 Run Selected Tests", type="primary"):
        run_security_tests({
            'fgsm': test_fgsm,
            'pgd': test_pgd,
            'robustness': test_robustness,
            'fuzzing': test_fuzzing,
            'integrity': test_integrity
        }, {
            'eps': eps,
            'num_samples': num_test_samples,
            'fuzz_samples': fuzz_samples,
            'generate_plots': generate_plots,
            'plot_quality': plot_quality
        })


def run_security_tests(selected_tests, params):
    """Run selected security tests"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_tests = sum(selected_tests.values())
    current_test = 0
    test_results = {}
    
    # Generate test data
    test_images, test_labels = generate_test_data(params['num_samples'])
    st.session_state.test_images = test_images
    st.session_state.test_labels = test_labels
    
    # Run tests
    tests_to_run = [
        ('FGSM Attack', 'fgsm', selected_tests['fgsm']),
        ('PGD Attack', 'pgd', selected_tests['pgd']),
        ('Robustness Test', 'robustness', selected_tests['robustness']),
        ('Fuzzing Test', 'fuzzing', selected_tests['fuzzing']),
        ('Integrity Check', 'integrity', selected_tests['integrity'])
    ]
    
    for test_name, test_key, should_run in tests_to_run:
        if not should_run:
            continue
            
        current_test += 1
        status_text.text(f"Running {test_name} ({current_test}/{total_tests})...")
        progress_bar.progress(current_test / total_tests)
        
        try:
            if test_key == 'fgsm':
                result = st.session_state.security_tester.run_adversarial_attack(
                    'fgsm', test_images, test_labels, eps=params['eps']
                )
            elif test_key == 'pgd':
                result = st.session_state.security_tester.run_adversarial_attack(
                    'pgd', test_images, test_labels, eps=params['eps']
                )
            elif test_key == 'robustness':
                result = st.session_state.security_tester.test_robustness(test_images, test_labels)
            elif test_key == 'fuzzing':
                base_image = test_images[0]
                result = st.session_state.security_tester.fuzz_inputs(
                    base_image, num_samples=params['fuzz_samples']
                )
            elif test_key == 'integrity':
                result = st.session_state.security_tester.check_model_integrity()
            else:
                result = {'success': False, 'error': 'Unknown test type'}
            
            test_results[test_key] = result
            
        except Exception as e:
            error_result = {'success': False, 'error': str(e)}
            test_results[test_key] = error_result
        
        time.sleep(0.3)
    
    # Update session state
    st.session_state.test_results = test_results
    
    # Final status
    progress_bar.progress(1.0)
    status_text.text("✅ All tests completed!")
    
    # Show summary
    passed = sum(1 for r in test_results.values() if r.get('success', False))
    failed = len(test_results) - passed
    
    if failed == 0:
        st.success(f"🎉 All {passed} tests passed successfully!")
        st.balloons()
    else:
        st.warning(f"⚠️ {passed} tests passed, {failed} tests failed")
    
    # Show quick results
    show_results_summary(test_results)


def generate_test_data(num_samples):
    """Generate test data"""
    test_images = np.random.rand(num_samples, 28, 28, 1).astype(np.float32)
    
    # Add patterns for more realistic results
    for i in range(num_samples):
        pattern_type = i % 4
        if pattern_type == 0:  # Horizontal bars
            test_images[i, 10:18, :, 0] = 0.8
        elif pattern_type == 1:  # Vertical bars
            test_images[i, :, 10:18, 0] = 0.8
        elif pattern_type == 2:  # Cross
            test_images[i, 10:18, :, 0] = 0.8
            test_images[i, :, 10:18, 0] = 0.8
        elif pattern_type == 3:  # Border
            test_images[i, 5:23, 5:23, 0] = 0.3
    
    test_labels = np.array([i % 10 for i in range(num_samples)])
    
    return test_images, test_labels


def show_results_summary(test_results):
    """Show quick results summary"""
    st.markdown("### 📋 Quick Results")
    
    for test_name, result in test_results.items():
        test_display = test_name.replace('_', ' ').title()
        
        if result.get('success', False):
            st.success(f"✅ {test_display}: PASSED")
        else:
            st.error(f"❌ {test_display}: FAILED - {result.get('error', 'Unknown')}")


def view_results_page():
    """View test results page"""
    st.header("📊 Test Results")
    
    if not st.session_state.test_results:
        st.info("No test results available. Please run tests first.")
        return
    
    # Summary statistics
    total_tests = len(st.session_state.test_results)
    passed_tests = sum(1 for r in st.session_state.test_results.values() if r.get('success', False))
    failed_tests = total_tests - passed_tests
    
    # Summary cards
    st.markdown("### 📋 Test Summary")
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Total Tests", total_tests)
    with cols[1]:
        st.metric("Passed", passed_tests)
    with cols[2]:
        st.metric("Failed", failed_tests, delta_color="inverse")
    with cols[3]:
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    st.divider()
    
    # Detailed results
    st.markdown("### 🔍 Detailed Results")
    
    visualizer = st.session_state.visualizer
    
    for test_name, result in st.session_state.test_results.items():
        # Display test header
        status_emoji = "✅" if result.get('success', False) else "❌"
        status_text = "PASS" if result.get('success', False) else "FAIL"
        st.markdown(f"#### {status_emoji} {test_name.replace('_', ' ').title()} - {status_text}")
        
        if not result.get('success', False):
            st.error(f"**Test failed:** {result.get('error', 'Unknown error')}")
            st.divider()
            continue
        
        # Display based on test type
        if 'original_accuracy' in result:
            show_adversarial_results_with_images(test_name, result, visualizer)
        elif 'baseline_accuracy' in result:
            show_robustness_results_with_images(test_name, result, visualizer)
        elif 'crash_rate' in result:
            show_fuzzing_results_simple(test_name, result)
        elif 'integrity_score' in result:
            show_integrity_results_simple(test_name, result)
        
        st.divider()
    
    # Overall assessment
    st.markdown("### 🏆 Overall Assessment")
    
    if failed_tests == 0:
        st.success("""
        ✅ **Excellent!** Your model passed all security tests. 
        It shows good robustness against the tested attacks.
        """)
    elif failed_tests <= total_tests / 2:
        st.warning(f"""
        ⚠️ **Needs Improvement.** Your model failed {failed_tests} out of {total_tests} tests.
        Consider implementing additional defenses like adversarial training or input sanitization.
        """)
    else:
        st.error(f"""
        ❌ **Critical Issues.** Your model failed {failed_tests} out of {total_tests} tests.
        Immediate remediation is required before deployment.
        """)
    
    # Raw data view
    if st.button("📋 View Raw Data"):
        st.json(st.session_state.test_results)


def show_adversarial_results_with_images(test_name, result, visualizer):
    """Show adversarial attack results with images"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Generate and display visualization
        if 'adversarial_images' in result and result['adversarial_images']:
            try:
                if st.session_state.test_images is not None:
                    num_to_show = min(3, len(st.session_state.test_images))
                    
                    # Convert lists to numpy arrays if needed
                    if isinstance(result['adversarial_images'], list):
                        adv_images = np.array(result['adversarial_images'][:num_to_show])
                    else:
                        adv_images = result['adversarial_images'][:num_to_show]
                    
                    perts = None
                    if 'perturbations' in result and result['perturbations']:
                        if isinstance(result['perturbations'], list):
                            perts = np.array(result['perturbations'][:num_to_show])
                        else:
                            perts = result['perturbations'][:num_to_show]
                    
                    plot = visualizer.plot_adversarial_comparison(
                        st.session_state.test_images[:num_to_show],
                        adv_images,
                        perts
                    )
                    if plot:
                        st.image(plot, use_container_width=True, 
                                caption=f"{test_name.replace('_', ' ')} - Adversarial Comparison")
            except Exception as e:
                st.warning(f"Could not generate visualization: {str(e)}")
    
    with col2:
        st.metric("Original Accuracy", f"{result['original_accuracy']:.2%}")
        st.metric("Adversarial Accuracy", f"{result['adversarial_accuracy']:.2%}")
        st.metric("Accuracy Drop", f"{result['accuracy_drop']:.2%}")
        if 'attack_time' in result:
            st.caption(f"Attack time: {result['attack_time']:.2f}s")
        
        # Store the visualization for reports
        if 'plot' in locals() and plot:
            st.session_state[f"{test_name}_plot"] = plot


def show_robustness_results_with_images(test_name, result, visualizer):
    """Show robustness test results with images"""
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Generate and display visualization
        try:
            plot = visualizer.plot_robustness_results(result)
            if plot:
                st.image(plot, use_container_width=True, 
                        caption=f"{test_name.replace('_', ' ')} - Robustness Analysis")
        except Exception as e:
            st.warning(f"Could not generate visualization: {str(e)}")
    
    with col2:
        st.metric("Baseline Accuracy", f"{result['baseline_accuracy']:.2%}")
        
        # Show top noise test results
        if 'noise_tests' in result and result['noise_tests']:
            st.markdown("**Top Noise Tests:**")
            for i, (noise_type, accuracy) in enumerate(list(result['noise_tests'].items())[:3]):
                noise_name = noise_type.replace('_', ' ').title()
                st.text(f"{noise_name}: {accuracy:.2%}")
        
        # Store the visualization for reports
        if 'plot' in locals() and plot:
            st.session_state[f"{test_name}_plot"] = plot


def show_fuzzing_results_simple(test_name, result):
    """Show fuzzing test results"""
    cols = st.columns(4)
    with cols[0]:
        st.metric("Samples", result['num_samples'])
    with cols[1]:
        st.metric("Crashes", result['crashes'])
    with cols[2]:
        st.metric("Anomalies", result['anomalies'])
    with cols[3]:
        stability = 1 - result['crash_rate'] - result['anomaly_rate']
        st.metric("Stability", f"{stability:.2%}")


def show_integrity_results_simple(test_name, result):
    """Show integrity check results"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Integrity Score", f"{result.get('integrity_score', 0)}/100")
        st.metric("File Hash", result.get('file_hash', 'Unknown')[:16] + '...')
    
    with col2:
        if result.get('checksum_valid', False):
            st.success("✅ Checksum Valid")
        else:
            st.error("❌ Checksum Invalid")


def export_report_page():
    """Export report page with images"""
    st.header("📄 Export Report")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please upload a model first")
        return
    
    if not st.session_state.test_results:
        st.warning("⚠️ Please run tests first")
        return
    
    st.markdown("### Generate Security Report with Images")
    
    # Report options
    col1, col2 = st.columns(2)
    
    with col1:
        report_format = st.selectbox("Report Format", ["HTML", "PDF", "Text", "JSON"])
        
    with col2:
        include_images = st.checkbox("Include visualizations", value=True)
        include_details = st.checkbox("Include detailed results", value=True)
        include_recommendations = st.checkbox("Include recommendations", value=True)
    
    # Generate report
    if st.button("🚀 Generate Report with Images", type="primary"):
        generate_and_download_report_with_images(
            report_format, 
            include_images, 
            include_details, 
            include_recommendations
        )


def generate_and_download_report_with_images(report_format, include_images, include_details, include_recommendations):
    """Generate and download report with images"""
    with st.spinner(f"Generating {report_format} report with images..."):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Generate visualizations for the report
            plots = []
            if include_images:
                plots = generate_report_visualizations()
            
            if report_format == "PDF":
                # PDF report with images
                try:
                    pdf_generator = PDFReportGenerator()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        report_path = tmp_file.name
                    
                    # Generate PDF with plots
                    report_path = pdf_generator.generate_pdf_report(
                        st.session_state.model_info,
                        st.session_state.test_results,
                        report_path,
                        plots=plots
                    )
                    
                    # Read and offer download
                    with open(report_path, 'rb') as f:
                        report_data = f.read()
                    
                    st.download_button(
                        label="📥 Download PDF Report with Images",
                        data=report_data,
                        file_name=f"security_report_{timestamp}.pdf",
                        mime="application/pdf"
                    )
                    
                    # Show preview of first plot
                    if plots and len(plots) > 0:
                        st.image(plots[0]['image'], caption="Sample visualization included in report")
                    
                    st.success("✅ PDF report with images generated successfully!")
                    
                except ImportError as e:
                    st.error(f"❌ PDF generation not available: {str(e)}")
                    st.info("**Try this fix:** `pip uninstall reportlab && pip install reportlab==3.6.12`")
                    return
                except Exception as e:
                    st.error(f"❌ PDF generation failed: {str(e)}")
                    st.info("Try HTML format instead")
                    return
                
            elif report_format == "HTML":
                # HTML report with images
                try:
                    html_generator = HTMLReportGenerator()
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
                        report_path = tmp_file.name
                    
                    # Generate HTML with plots
                    report_path = html_generator.generate_html_report(
                        st.session_state.model_info,
                        st.session_state.test_results,
                        report_path,
                        plots=plots
                    )
                    
                    with open(report_path, 'r', encoding='utf-8') as f:
                        report_data = f.read()
                    
                    st.download_button(
                        label="📥 Download HTML Report with Images",
                        data=report_data,
                        file_name=f"security_report_{timestamp}.html",
                        mime="text/html"
                    )
                    
                    # Show preview
                    with st.expander("📄 Report Preview"):
                        st.components.v1.html(report_data, height=600, scrolling=True)
                    
                    st.success("✅ HTML report with images generated successfully!")
                    
                except Exception as e:
                    st.error(f"❌ HTML generation failed: {str(e)}")
                
            elif report_format == "Text":
                # Text report (no images)
                report_lines = [
                    "=" * 60,
                    "CV MODEL SECURITY TEST REPORT",
                    "=" * 60,
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    "SUMMARY",
                    "-" * 40,
                ]
                
                total = len(st.session_state.test_results)
                passed = sum(1 for r in st.session_state.test_results.values() if r.get('success', False))
                report_lines.extend([
                    f"Total Tests: {total}",
                    f"Passed: {passed}",
                    f"Failed: {total - passed}",
                    "",
                    "MODEL INFORMATION",
                    "-" * 40,
                    f"Format: {st.session_state.model_info.get('format', 'Unknown')}",
                    f"Framework: {st.session_state.model_info.get('framework', 'Unknown')}",
                    f"File Size: {st.session_state.model_info.get('file_size', 0) / 1024 / 1024:.2f} MB",
                    "",
                    "TEST RESULTS",
                    "-" * 40,
                ])
                
                for test_name, result in st.session_state.test_results.items():
                    status = "PASS" if result.get('success', False) else "FAIL"
                    report_lines.append(f"{test_name.replace('_', ' ').title()}: {status}")
                    if not result.get('success', False):
                        report_lines.append(f"  Error: {result.get('error', 'Unknown')}")
                
                if include_recommendations:
                    report_lines.extend([
                        "",
                        "RECOMMENDATIONS",
                        "-" * 40,
                    ])
                    
                    if passed == total:
                        report_lines.append("• All tests passed - maintain current security practices")
                    else:
                        report_lines.append(f"• Address {total - passed} failed test(s)")
                        report_lines.append("• Implement adversarial training")
                        report_lines.append("• Add input validation")
                
                report_text = "\n".join(report_lines)
                
                st.download_button(
                    label="📥 Download Text Report",
                    data=report_text,
                    file_name=f"security_report_{timestamp}.txt",
                    mime="text/plain"
                )
                
                st.success("✅ Text report generated successfully!")
                
            else:  # JSON
                # JSON report with image metadata
                report_data = {
                    'model_info': st.session_state.model_info,
                    'test_results': st.session_state.test_results,
                    'timestamp': timestamp,
                    'summary': {
                        'total_tests': len(st.session_state.test_results),
                        'passed_tests': sum(1 for r in st.session_state.test_results.values() if r.get('success', False))
                    },
                    'metadata': {
                        'includes_images': include_images,
                        'image_count': len(plots) if plots else 0
                    }
                }
                
                json_str = json.dumps(report_data, indent=2, default=str)
                
                st.download_button(
                    label="📥 Download JSON Report",
                    data=json_str,
                    file_name=f"security_report_{timestamp}.json",
                    mime="application/json"
                )
                
                st.success("✅ JSON report generated successfully!")
            
        except Exception as e:
            st.error(f"❌ Failed to generate report: {str(e)}")
            st.info("Try a different report format or check the error details.")
        
        finally:
            # Cleanup
            try:
                if 'report_path' in locals():
                    os.unlink(report_path)
            except:
                pass


def generate_report_visualizations():
    """Generate visualizations for the report"""
    plots = []
    visualizer = st.session_state.visualizer
    
    # Generate visualizations for each successful test
    for test_name, result in st.session_state.test_results.items():
        if not result.get('success', False):
            continue
        
        try:
            if 'original_accuracy' in result:
                # Generate adversarial attack visualization
                if st.session_state.test_images is not None:
                    num_to_show = min(2, len(st.session_state.test_images))
                    
                    adv_images = None
                    if 'adversarial_images' in result and result['adversarial_images']:
                        if isinstance(result['adversarial_images'], list):
                            adv_images = np.array(result['adversarial_images'][:num_to_show])
                        else:
                            adv_images = result['adversarial_images'][:num_to_show]
                    
                    perts = None
                    if 'perturbations' in result and result['perturbations']:
                        if isinstance(result['perturbations'], list):
                            perts = np.array(result['perturbations'][:num_to_show])
                        else:
                            perts = result['perturbations'][:num_to_show]
                    
                    plot = visualizer.plot_adversarial_comparison(
                        st.session_state.test_images[:num_to_show],
                        adv_images,
                        perts
                    )
                    
                    if plot:
                        plots.append({
                            'name': test_name,
                            'image': plot,
                            'title': f"{test_name.replace('_', ' ').title()} - Adversarial Attack"
                        })
            
            elif 'baseline_accuracy' in result:
                # Generate robustness visualization
                plot = visualizer.plot_robustness_results(result)
                
                if plot:
                    plots.append({
                        'name': test_name,
                        'image': plot,
                        'title': f"{test_name.replace('_', ' ').title()} - Robustness Analysis"
                    })
        
        except Exception as e:
            # Skip if visualization fails
            continue
    
    return plots


if __name__ == "__main__":
    main()