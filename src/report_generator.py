"""
Report Generator Module
Generates PDF reports with test results
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import io


class ReportGenerator:
    """Generates PDF security test reports"""
    
    def __init__(self):
        """Initialize report generator"""
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2c3e50'),
            spaceAfter=12
        )
        
        self.warning_style = ParagraphStyle(
            'Warning',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=colors.red,
            backColor=colors.yellow,
            borderPadding=5
        )
    
    def generate_report(self, model_info: Dict, test_results: Dict, 
                       output_path: str, plots: List = None) -> str:
        """
        Generate PDF security test report
        
        Args:
            model_info: Model information
            test_results: Test results dictionary
            output_path: Output PDF file path
            plots: List of plot file paths (optional)
            
        Returns:
            str: Path to generated report
        """
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Title
        story.append(Paragraph("CV Model Security Test Report", self.title_style))
        story.append(Spacer(1, 0.2 * inch))
        
        # Date and disclaimer
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"Generated: {date_str}", self.styles['Normal']))
        story.append(Spacer(1, 0.1 * inch))
        
        # Ethical disclaimer
        disclaimer = (
            "<b>ETHICAL DISCLAIMER:</b> This report is for defensive security testing only. "
            "No actual exploitation or malicious attacks were performed. All tests are "
            "simulations designed to identify vulnerabilities for remediation purposes."
        )
        story.append(Paragraph(disclaimer, self.warning_style))
        story.append(Spacer(1, 0.3 * inch))
        
        # Model Information
        story.append(Paragraph("Model Information", self.heading_style))
        model_data = [
            ['Property', 'Value'],
            ['Format', model_info.get('format', 'Unknown')],
            ['Framework', model_info.get('framework', 'Unknown')],
            ['File Size', f"{model_info.get('file_size', 0) / 1024 / 1024:.2f} MB"],
            ['File Hash', model_info.get('file_hash', 'Unknown')[:16] + '...'],
            ['Input Shape', str(model_info.get('input_shape', 'Unknown'))],
            ['Output Shape', str(model_info.get('output_shape', 'Unknown'))]
        ]
        
        model_table = Table(model_data, colWidths=[2 * inch, 4 * inch])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(model_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # Test Results Summary
        story.append(Paragraph("Test Results Summary", self.heading_style))
        
        summary_data = [['Test Type', 'Status', 'Details']]
        
        for test_name, result in test_results.items():
            if isinstance(result, dict):
                status = "PASS" if result.get('success', False) else "FAIL"
                details = result.get('error', 'Completed')[:50]
                summary_data.append([test_name, status, details])
        
        summary_table = Table(summary_data, colWidths=[2 * inch, 1 * inch, 3 * inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.3 * inch))
        
        # Detailed Results
        story.append(Paragraph("Detailed Test Results", self.heading_style))
        
        for test_name, result in test_results.items():
            if isinstance(result, dict) and result.get('success'):
                story.append(Paragraph(f"<b>{test_name}</b>", self.styles['Heading3']))
                
                # Add test-specific details
                if 'adversarial_accuracy' in result:
                    story.append(Paragraph(
                        f"Original Accuracy: {result['original_accuracy']:.2%}<br/>"
                        f"Adversarial Accuracy: {result['adversarial_accuracy']:.2%}<br/>"
                        f"Accuracy Drop: {result['accuracy_drop']:.2%}",
                        self.styles['Normal']
                    ))
                
                if 'baseline_accuracy' in result:
                    story.append(Paragraph(
                        f"Baseline Accuracy: {result['baseline_accuracy']:.2%}",
                        self.styles['Normal']
                    ))
                
                story.append(Spacer(1, 0.2 * inch))
        
        # Build PDF
        doc.build(story)
        
        return output_path





