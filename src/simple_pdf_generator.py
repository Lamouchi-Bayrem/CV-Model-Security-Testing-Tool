"""
Simple PDF Generator using fpdf (fallback option)
"""

from fpdf import FPDF
from datetime import datetime
from typing import Dict


class SimplePDFGenerator:
    """Simple PDF generator using fpdf library"""
    
    def generate_report(self, model_info: Dict, test_results: Dict, output_path: str) -> str:
        """Generate a simple PDF report"""
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Add cover page
        self._add_cover_page(pdf, model_info)
        
        # Add table of contents
        self._add_table_of_contents(pdf)
        
        # Add executive summary
        self._add_executive_summary(pdf, test_results)
        
        # Add model information
        self._add_model_info(pdf, model_info)
        
        # Add test results
        self._add_test_results(pdf, test_results)
        
        # Add recommendations
        self._add_recommendations(pdf, test_results)
        
        # Save PDF
        pdf.output(output_path)
        return output_path
    
    def _add_cover_page(self, pdf: FPDF, model_info: Dict):
        """Add cover page to PDF"""
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 40, 'CV Model Security Test Report', 0, 1, 'C')
        pdf.ln(20)
        
        # Subtitle
        pdf.set_font('Arial', 'I', 14)
        pdf.cell(0, 10, 'Comprehensive Security Assessment', 0, 1, 'C')
        pdf.ln(30)
        
        # Model information
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Model Information:', 0, 1, 'L')
        pdf.set_font('Arial', '', 11)
        
        info_items = [
            ('Format', model_info.get('format', 'Unknown')),
            ('Framework', model_info.get('framework', 'Unknown')),
            ('File Size', f"{model_info.get('file_size', 0) / 1024 / 1024:.2f} MB"),
            ('Report Date', datetime.now().strftime('%B %d, %Y'))
        ]
        
        for label, value in info_items:
            pdf.cell(40, 8, f"{label}:", 0, 0)
            pdf.cell(0, 8, value, 0, 1)
        
        pdf.ln(20)
        
        # Disclaimer
        pdf.set_font('Arial', 'I', 9)
        pdf.multi_cell(0, 5, 
            "ETHICAL DISCLAIMER: This report is for defensive security testing only. "
            "No actual exploitation or malicious attacks were performed. All tests are "
            "simulations designed to identify vulnerabilities for remediation purposes.")
    
    def _add_table_of_contents(self, pdf: FPDF):
        """Add table of contents"""
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Table of Contents', 0, 1)
        pdf.ln(10)
        
        contents = [
            ('Executive Summary', 1),
            ('Model Information', 2),
            ('Test Results', 3),
            ('Security Recommendations', 4)
        ]
        
        pdf.set_font('Arial', '', 12)
        for title, page in contents:
            pdf.cell(0, 8, f"{title} ................................................ {page}", 0, 1)
    
    def _add_executive_summary(self, pdf: FPDF, test_results: Dict):
        """Add executive summary section"""
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Executive Summary', 0, 1)
        pdf.ln(10)
        
        # Calculate statistics
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results.values() if r.get('success', False))
        failed_tests = total_tests - passed_tests
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Assessment
        if failed_tests == 0:
            assessment = "EXCELLENT - All security tests passed"
        elif failed_tests <= total_tests / 3:
            assessment = "GOOD - Minor issues detected"
        else:
            assessment = "NEEDS IMPROVEMENT - Security issues detected"
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Overall Assessment:', 0, 1)
        pdf.set_font('Arial', '', 11)
        pdf.multi_cell(0, 6, assessment)
        pdf.ln(5)
        
        # Statistics table
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(60, 8, 'Metric', 'B', 0)
        pdf.cell(40, 8, 'Value', 'B', 1)
        pdf.set_font('Arial', '', 11)
        
        stats = [
            ('Total Tests', str(total_tests)),
            ('Tests Passed', str(passed_tests)),
            ('Tests Failed', str(failed_tests)),
            ('Success Rate', f'{success_rate:.1f}%')
        ]
        
        for label, value in stats:
            pdf.cell(60, 8, label, 0, 0)
            pdf.cell(40, 8, value, 0, 1)
    
    def _add_model_info(self, pdf: FPDF, model_info: Dict):
        """Add model information section"""
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Model Information', 0, 1)
        pdf.ln(10)
        
        info_items = [
            ('Format', model_info.get('format', 'Unknown')),
            ('Framework', model_info.get('framework', 'Unknown')),
            ('File Size', f"{model_info.get('file_size', 0) / 1024 / 1024:.2f} MB"),
            ('File Hash', model_info.get('file_hash', 'Unknown')[:32] + '...'),
            ('Input Shape', str(model_info.get('input_shape', 'Unknown'))),
            ('Output Shape', str(model_info.get('output_shape', 'Unknown'))),
            ('Parameters', str(model_info.get('num_parameters', 'Unknown')))
        ]
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(60, 8, 'Property', 'B', 0)
        pdf.cell(0, 8, 'Value', 'B', 1)
        pdf.set_font('Arial', '', 11)
        
        for label, value in info_items:
            pdf.cell(60, 8, label, 0, 0)
            pdf.multi_cell(0, 8, value)
    
    def _add_test_results(self, pdf: FPDF, test_results: Dict):
        """Add test results section"""
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Test Results', 0, 1)
        pdf.ln(10)
        
        pdf.set_font('Arial', 'B', 11)
        pdf.cell(50, 8, 'Test', 'B', 0)
        pdf.cell(30, 8, 'Status', 'B', 0)
        pdf.cell(0, 8, 'Details', 'B', 1)
        pdf.set_font('Arial', '', 11)
        
        for test_name, result in test_results.items():
            test_display = test_name.replace('_', ' ').title()
            status = "PASS" if result.get('success', False) else "FAIL"
            
            pdf.cell(50, 8, test_display, 0, 0)
            pdf.cell(30, 8, status, 0, 0)
            
            if result.get('success', False):
                if 'accuracy_drop' in result:
                    pdf.cell(0, 8, f"Accuracy Drop: {result['accuracy_drop']:.2%}", 0, 1)
                elif 'baseline_accuracy' in result:
                    pdf.cell(0, 8, f"Baseline: {result['baseline_accuracy']:.2%}", 0, 1)
                else:
                    pdf.cell(0, 8, "Completed", 0, 1)
            else:
                error = result.get('error', 'Unknown error')
                if len(error) > 60:
                    error = error[:57] + "..."
                pdf.cell(0, 8, f"Error: {error}", 0, 1)
    
    def _add_recommendations(self, pdf: FPDF, test_results: Dict):
        """Add recommendations section"""
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Security Recommendations', 0, 1)
        pdf.ln(10)
        
        # Analyze results for recommendations
        failed_tests = sum(1 for r in test_results.values() if not r.get('success', False))
        
        if failed_tests == 0:
            recommendations = [
                "All security tests passed successfully.",
                "Continue regular security monitoring and audits.",
                "Implement defense in depth strategy for production deployment."
            ]
        else:
            recommendations = [
                f"Address {failed_tests} failed test(s) before production deployment.",
                "Implement adversarial training if accuracy drop is significant.",
                "Add robust input validation and error handling.",
                "Schedule regular security audits (quarterly recommended).",
                "Establish continuous monitoring for model drift."
            ]
        
        pdf.set_font('Arial', '', 11)
        for i, rec in enumerate(recommendations, 1):
            pdf.multi_cell(0, 6, f"{i}. {rec}")
            pdf.ln(3)
            