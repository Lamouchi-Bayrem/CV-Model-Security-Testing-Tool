"""
Report Generator with Images - Fixed temporary file path issue
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import tempfile
import os
import json
import base64
from io import BytesIO
import numpy as np
from PIL import Image as PILImage


class PDFReportGenerator:
    """Generates PDF reports with images - FIXED TEMP FILE ISSUE"""
    
    def generate_pdf_report(self, model_info: Dict, test_results: Dict, 
                           output_path: str, plots: List[Dict] = None) -> str:
        """
        Generate PDF security test report with images
        
        Args:
            model_info: Model information dictionary
            test_results: Test results dictionary
            output_path: Output PDF file path
            plots: List of plot dictionaries with 'name', 'image', 'title'
            
        Returns:
            str: Path to generated report
        """
        try:
            # Import reportlab components
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib import colors
            from reportlab.lib.units import inch, cm
            from reportlab.pdfgen import canvas
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.platypus import Image as RLImage
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
            from reportlab.lib.utils import ImageReader
            
            # Create document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            styles = getSampleStyleSheet()
            story = []
            
            # Title
            title_style = styles['Title']
            title_style.alignment = TA_CENTER
            story.append(Paragraph("CV Model Security Test Report", title_style))
            story.append(Spacer(1, 0.5*inch))
            
            # Date and disclaimer
            normal_style = styles['Normal']
            story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
            story.append(Paragraph("ETHICAL DISCLAIMER: For defensive security testing only.", 
                                 styles['Italic']))
            story.append(Spacer(1, 0.3*inch))
            
            # Summary
            total_tests = len(test_results)
            passed_tests = sum(1 for r in test_results.values() if r.get('success', False))
            
            summary_data = [
                ['Total Tests', str(total_tests)],
                ['Tests Passed', str(passed_tests)],
                ['Tests Failed', str(total_tests - passed_tests)],
                ['Success Rate', f"{(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%"]
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ]))
            
            story.append(Paragraph("Executive Summary", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            story.append(summary_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Model Information
            story.append(Paragraph("Model Information", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            
            model_data = [
                ['Property', 'Value'],
                ['Format', model_info.get('format', 'Unknown')],
                ['Framework', model_info.get('framework', 'Unknown')],
                ['File Size', f"{model_info.get('file_size', 0) / 1024 / 1024:.2f} MB"],
                ['Input Shape', str(model_info.get('input_shape', 'Unknown'))],
                ['Output Shape', str(model_info.get('output_shape', 'Unknown'))],
            ]
            
            model_table = Table(model_data, colWidths=[1.5*inch, 3*inch])
            model_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ]))
            
            story.append(model_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Test Results
            story.append(Paragraph("Test Results", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            
            # Add test results table
            test_data = [['Test', 'Status', 'Details']]
            
            for test_name, result in test_results.items():
                test_display = test_name.replace('_', ' ').title()
                status = "PASS" if result.get('success', False) else "FAIL"
                
                details = ""
                if not result.get('success', False):
                    details = result.get('error', 'Failed')[:50]
                elif 'accuracy_drop' in result:
                    details = f"Accuracy Drop: {result['accuracy_drop']:.2%}"
                elif 'baseline_accuracy' in result:
                    details = f"Baseline: {result['baseline_accuracy']:.2%}"
                elif 'crash_rate' in result:
                    details = f"Crash Rate: {result['crash_rate']:.2%}"
                else:
                    details = "Completed"
                
                test_data.append([test_display, status, details])
            
            test_table = Table(test_data, colWidths=[1.5*inch, 0.8*inch, 2.7*inch])
            test_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ]))
            
            story.append(test_table)
            story.append(Spacer(1, 0.3*inch))
            
            # Add images/plots if available - FIXED: Use BytesIO instead of temp files
            if plots and len(plots) > 0:
                story.append(Paragraph("Visualizations", styles['Heading2']))
                story.append(Spacer(1, 0.1*inch))
                
                for plot_info in plots:
                    if 'title' in plot_info:
                        story.append(Paragraph(plot_info['title'], styles['Heading3']))
                        story.append(Spacer(1, 0.1*inch))
                    
                    if 'image' in plot_info and plot_info['image'] is not None:
                        try:
                            # Convert PIL Image to BytesIO to avoid temp file issues
                            img_buffer = BytesIO()
                            plot_info['image'].save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            
                            # Create ImageReader directly from BytesIO
                            img_reader = ImageReader(img_buffer)
                            
                            # Create a simple canvas-based approach for images
                            # We'll add images using a different method
                            img_width = 5 * inch
                            img_height = 3 * inch
                            
                            # Create a simple table with the image
                            img_table_data = [[RLImage(img_buffer, width=img_width, height=img_height)]]
                            img_table = Table(img_table_data)
                            story.append(img_table)
                            story.append(Spacer(1, 0.2*inch))
                            
                        except Exception as e:
                            # Fallback: just add text description
                            story.append(Paragraph(f"Visualization available for {plot_info.get('title', 'test')}", 
                                                 styles['Italic']))
                            story.append(Spacer(1, 0.1*inch))
            
            # Recommendations
            story.append(Paragraph("Security Recommendations", styles['Heading2']))
            story.append(Spacer(1, 0.1*inch))
            
            recommendations = []
            if passed_tests == total_tests:
                recommendations.append("✅ All tests passed successfully")
                recommendations.append("• Continue regular security monitoring")
                recommendations.append("• Schedule quarterly security audits")
            else:
                recommendations.append(f"⚠️ {total_tests - passed_tests} tests failed")
                recommendations.append("• Address failed tests immediately")
                recommendations.append("• Implement adversarial training")
                recommendations.append("• Add input validation layers")
                recommendations.append("• Perform regular security audits")
            
            for rec in recommendations:
                story.append(Paragraph(rec, normal_style))
                story.append(Spacer(1, 0.05*inch))
            
            # Build PDF
            doc.build(story)
            
            return output_path
            
        except Exception as e:
            if 'usedforsecurity' in str(e):
                raise ImportError("ReportLab version incompatible. Try: pip install reportlab==3.6.12")
            else:
                raise e


class SimplePDFReportGenerator:
    """Alternative simple PDF generator without complex table layouts"""
    
    def generate_pdf_report(self, model_info: Dict, test_results: Dict, 
                           output_path: str, plots: List[Dict] = None) -> str:
        """
        Generate simple PDF report using canvas (more reliable)
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            from reportlab.lib.units import inch
            from reportlab.lib.utils import ImageReader
            from io import BytesIO
            
            # Create canvas
            c = canvas.Canvas(output_path, pagesize=A4)
            width, height = A4
            
            # Title
            c.setFont("Helvetica-Bold", 20)
            c.drawCentredString(width/2, height-50, "CV Model Security Test Report")
            
            # Date
            c.setFont("Helvetica", 12)
            c.drawString(50, height-80, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Summary
            total_tests = len(test_results)
            passed_tests = sum(1 for r in test_results.values() if r.get('success', False))
            
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, height-120, "Summary")
            
            c.setFont("Helvetica", 12)
            c.drawString(50, height-150, f"Total Tests: {total_tests}")
            c.drawString(50, height-170, f"Tests Passed: {passed_tests}")
            c.drawString(50, height-190, f"Tests Failed: {total_tests - passed_tests}")
            c.drawString(50, height-210, f"Success Rate: {(passed_tests/total_tests*100):.1f}%" if total_tests > 0 else "0%")
            
            # Model Information
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, height-250, "Model Information")
            
            c.setFont("Helvetica", 12)
            y_pos = height-280
            c.drawString(50, y_pos, f"Format: {model_info.get('format', 'Unknown')}")
            y_pos -= 20
            c.drawString(50, y_pos, f"Framework: {model_info.get('framework', 'Unknown')}")
            y_pos -= 20
            c.drawString(50, y_pos, f"File Size: {model_info.get('file_size', 0) / 1024 / 1024:.2f} MB")
            y_pos -= 20
            c.drawString(50, y_pos, f"Input Shape: {str(model_info.get('input_shape', 'Unknown'))}")
            
            # Test Results
            y_pos -= 40
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_pos, "Test Results")
            
            c.setFont("Helvetica", 12)
            y_pos -= 30
            
            for test_name, result in test_results.items():
                test_display = test_name.replace('_', ' ').title()
                status = "PASS" if result.get('success', False) else "FAIL"
                
                if y_pos < 100:  # New page if needed
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y_pos = height - 50
                
                c.drawString(50, y_pos, f"{test_display}: {status}")
                y_pos -= 20
                
                if not result.get('success', False):
                    error_msg = result.get('error', 'Unknown')[:60]
                    c.drawString(70, y_pos, f"Error: {error_msg}")
                    y_pos -= 20
            
            # Add images if available
            if plots and len(plots) > 0:
                y_pos -= 40
                c.setFont("Helvetica-Bold", 14)
                c.drawString(50, y_pos, "Visualizations")
                y_pos -= 30
                
                for plot_info in plots:
                    if 'image' in plot_info and plot_info['image'] is not None:
                        try:
                            # Check if we need a new page
                            if y_pos < 200:
                                c.showPage()
                                y_pos = height - 50
                                c.setFont("Helvetica", 12)
                            
                            # Add plot title
                            if 'title' in plot_info:
                                c.drawString(50, y_pos, plot_info['title'])
                                y_pos -= 20
                            
                            # Convert PIL image to BytesIO
                            img_buffer = BytesIO()
                            plot_info['image'].save(img_buffer, format='PNG')
                            img_buffer.seek(0)
                            
                            # Draw image
                            img_width = 400
                            img_height = 250
                            c.drawImage(ImageReader(img_buffer), 50, y_pos-img_height, 
                                       width=img_width, height=img_height)
                            
                            y_pos -= (img_height + 40)
                            
                        except Exception as e:
                            c.drawString(50, y_pos, f"Could not load image: {str(e)[:50]}")
                            y_pos -= 20
            
            # Recommendations
            if y_pos < 150:
                c.showPage()
                y_pos = height - 50
            
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y_pos, "Security Recommendations")
            y_pos -= 30
            
            c.setFont("Helvetica", 12)
            if passed_tests == total_tests:
                recommendations = [
                    "✅ All tests passed successfully",
                    "• Continue regular security monitoring",
                    "• Schedule quarterly security audits",
                    "• Keep security frameworks updated"
                ]
            else:
                recommendations = [
                    f"⚠️ {total_tests - passed_tests} tests failed",
                    "• Address failed tests immediately",
                    "• Implement adversarial training",
                    "• Add input validation layers",
                    "• Perform regular security audits"
                ]
            
            for rec in recommendations:
                if y_pos < 50:
                    c.showPage()
                    y_pos = height - 50
                    c.setFont("Helvetica", 12)
                
                c.drawString(50, y_pos, rec)
                y_pos -= 20
            
            # Disclaimer
            if y_pos < 100:
                c.showPage()
                y_pos = height - 50
            
            c.setFont("Helvetica-Oblique", 10)
            disclaimer = "ETHICAL DISCLAIMER: This report is for defensive security testing only."
            c.drawString(50, y_pos, disclaimer)
            
            # Save PDF
            c.save()
            
            return output_path
            
        except Exception as e:
            raise e


class HTMLReportGenerator:
    """HTML report generator with images"""
    
    def generate_html_report(self, model_info: Dict, test_results: Dict, 
                            output_path: str, plots: List[Dict] = None) -> str:
        """Generate HTML report with embedded images"""
        total_tests = len(test_results)
        passed_tests = sum(1 for r in test_results.values() if r.get('success', False))
        
        # Convert images to base64 for HTML embedding
        plot_images_html = ""
        if plots:
            for i, plot_info in enumerate(plots):
                if 'image' in plot_info and plot_info['image'] is not None:
                    try:
                        # Convert PIL image to base64
                        buffered = BytesIO()
                        plot_info['image'].save(buffered, format="PNG", optimize=True)
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        
                        title = plot_info.get('title', f'Visualization {i+1}')
                        description = plot_info.get('description', 'Security test visualization')
                        
                        plot_images_html += f"""
                        <div class="plot-card">
                            <div class="plot-header">
                                <h3>{title}</h3>
                                <p class="plot-description">{description}</p>
                            </div>
                            <div class="plot-image">
                                <img src="data:image/png;base64,{img_str}" 
                                     alt="{title}" 
                                     class="plot-img">
                            </div>
                        </div>
                        """
                    except Exception as e:
                        plot_images_html += f"""
                        <div class="plot-error">
                            <p>⚠️ Could not load image: {str(e)[:100]}</p>
                        </div>
                        """
        
        # Create metrics HTML
        metrics_html = ""
        for test_name, result in test_results.items():
            if result.get('success', False) and 'accuracy_drop' in result:
                test_display = test_name.replace('_', ' ').title()
                metrics_html += f"""
                <div class="metric-detail">
                    <span class="metric-name">{test_display}:</span>
                    <span class="metric-value">{result['accuracy_drop']:.2%} accuracy drop</span>
                </div>
                """
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>CV Model Security Test Report</title>
            <style>
                /* Reset and Base Styles */
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
                    overflow: hidden;
                }}
                
                /* Header */
                .header {{
                    background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                    position: relative;
                    overflow: hidden;
                }}
                
                .header::before {{
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
                    background-size: 50px 50px;
                    opacity: 0.3;
                    animation: moveBackground 20s linear infinite;
                }}
                
                @keyframes moveBackground {{
                    0% {{ transform: rotate(0deg); }}
                    100% {{ transform: rotate(360deg); }}
                }}
                
                .header h1 {{
                    font-size: 2.8em;
                    font-weight: 800;
                    margin-bottom: 10px;
                    letter-spacing: -0.5px;
                    position: relative;
                    z-index: 1;
                }}
                
                .header p {{
                    font-size: 1.2em;
                    opacity: 0.9;
                    position: relative;
                    z-index: 1;
                }}
                
                /* Content Sections */
                .content {{
                    padding: 40px;
                }}
                
                .section {{
                    margin-bottom: 50px;
                    padding: 30px;
                    background: #f8fafc;
                    border-radius: 15px;
                    border-left: 5px solid #3b82f6;
                    transition: all 0.3s ease;
                }}
                
                .section:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.1);
                }}
                
                .section h2 {{
                    color: #1e3a8a;
                    font-size: 1.8em;
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #3b82f6;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                
                .section h2::before {{
                    font-size: 1.2em;
                }}
                
                /* Summary Cards */
                .summary-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 25px 0;
                }}
                
                .summary-card {{
                    background: white;
                    padding: 25px;
                    border-radius: 12px;
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                }}
                
                .summary-card::before {{
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 4px;
                    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
                }}
                
                .summary-card:hover {{
                    transform: translateY(-8px);
                    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.15);
                }}
                
                .summary-card.total {{ border-top-color: #3b82f6; }}
                .summary-card.passed {{ border-top-color: #10b981; }}
                .summary-card.failed {{ border-top-color: #ef4444; }}
                .summary-card.rate {{ border-top-color: #f59e0b; }}
                
                .card-value {{
                    font-size: 3em;
                    font-weight: 800;
                    margin: 15px 0;
                    background: linear-gradient(135deg, #1a2980, #26d0ce);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }}
                
                .summary-card.passed .card-value {{
                    background: linear-gradient(135deg, #10b981, #34d399);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }}
                
                .summary-card.failed .card-value {{
                    background: linear-gradient(135deg, #ef4444, #f87171);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }}
                
                .card-label {{
                    font-size: 0.95em;
                    color: #6b7280;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                
                /* Tables */
                .results-table {{
                    width: 100%;
                    border-collapse: separate;
                    border-spacing: 0;
                    margin: 25px 0;
                    background: white;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                }}
                
                .results-table thead {{
                    background: linear-gradient(135deg, #1e3a8a, #3b82f6);
                }}
                
                .results-table th {{
                    padding: 18px 20px;
                    text-align: left;
                    color: white;
                    font-weight: 600;
                    font-size: 1.1em;
                }}
                
                .results-table tbody tr {{
                    transition: all 0.3s ease;
                }}
                
                .results-table tbody tr:nth-child(even) {{
                    background: #f8fafc;
                }}
                
                .results-table tbody tr:hover {{
                    background: #e0f2fe;
                    transform: scale(1.005);
                }}
                
                .results-table td {{
                    padding: 16px 20px;
                    border-bottom: 1px solid #e5e7eb;
                }}
                
                .status-badge {{
                    display: inline-block;
                    padding: 6px 16px;
                    border-radius: 20px;
                    font-size: 0.85em;
                    font-weight: 600;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                
                .status-pass {{
                    background: linear-gradient(135deg, #d1fae5, #10b981);
                    color: #065f46;
                }}
                
                .status-fail {{
                    background: linear-gradient(135deg, #fee2e2, #ef4444);
                    color: #7f1d1d;
                }}
                
                /* Plots/Visualizations */
                .plots-container {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                    gap: 30px;
                    margin: 30px 0;
                }}
                
                .plot-card {{
                    background: white;
                    border-radius: 15px;
                    overflow: hidden;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                    transition: all 0.3s ease;
                }}
                
                .plot-card:hover {{
                    transform: translateY(-8px);
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
                }}
                
                .plot-header {{
                    padding: 20px;
                    background: linear-gradient(135deg, #f8fafc, #e2e8f0);
                    border-bottom: 1px solid #e5e7eb;
                }}
                
                .plot-header h3 {{
                    color: #1e3a8a;
                    font-size: 1.3em;
                    margin-bottom: 8px;
                }}
                
                .plot-description {{
                    color: #6b7280;
                    font-size: 0.95em;
                    line-height: 1.5;
                }}
                
                .plot-image {{
                    padding: 20px;
                    text-align: center;
                }}
                
                .plot-img {{
                    max-width: 100%;
                    height: auto;
                    border-radius: 8px;
                    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
                }}
                
                /* Recommendations */
                .recommendations {{
                    background: linear-gradient(135deg, #fef3c7, #fbbf24);
                    padding: 35px;
                    border-radius: 15px;
                    margin-top: 40px;
                }}
                
                .recommendations h2 {{
                    color: #92400e;
                    border-bottom-color: #f59e0b;
                }}
                
                .recommendations-list {{
                    list-style: none;
                    padding: 0;
                }}
                
                .recommendations-list li {{
                    padding: 12px 0 12px 40px;
                    position: relative;
                    font-size: 1.1em;
                    color: #78350f;
                }}
                
                .recommendations-list li::before {{
                    content: '✓';
                    position: absolute;
                    left: 0;
                    top: 10px;
                    width: 28px;
                    height: 28px;
                    background: #10b981;
                    color: white;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    font-size: 0.9em;
                }}
                
                /* Disclaimer */
                .disclaimer {{
                    background: linear-gradient(135deg, #fee2e2, #fecaca);
                    padding: 25px;
                    border-radius: 12px;
                    margin-top: 40px;
                    border-left: 5px solid #dc2626;
                }}
                
                .disclaimer strong {{
                    color: #7f1d1d;
                    display: block;
                    margin-bottom: 10px;
                    font-size: 1.1em;
                }}
                
                .disclaimer p {{
                    color: #991b1b;
                    line-height: 1.6;
                }}
                
                /* Print Styles */
                @media print {{
                    body {{
                        background: white !important;
                        padding: 0 !important;
                    }}
                    
                    .container {{
                        box-shadow: none !important;
                        border-radius: 0 !important;
                        margin: 0 !important;
                    }}
                    
                    .section:hover {{
                        transform: none !important;
                        box-shadow: none !important;
                    }}
                    
                    .summary-card:hover {{
                        transform: none !important;
                        box-shadow: none !important;
                    }}
                    
                    .results-table tbody tr:hover {{
                        transform: none !important;
                        background: inherit !important;
                    }}
                    
                    .plot-card:hover {{
                        transform: none !important;
                        box-shadow: none !important;
                    }}
                    
                    .no-print {{
                        display: none !important;
                    }}
                }}
                
                /* Responsive Design */
                @media (max-width: 768px) {{
                    .header h1 {{
                        font-size: 2em;
                    }}
                    
                    .content {{
                        padding: 20px;
                    }}
                    
                    .section {{
                        padding: 20px;
                    }}
                    
                    .summary-grid {{
                        grid-template-columns: 1fr;
                    }}
                    
                    .plots-container {{
                        grid-template-columns: 1fr;
                    }}
                    
                    .results-table {{
                        font-size: 0.9em;
                    }}
                    
                    .results-table th,
                    .results-table td {{
                        padding: 12px 15px;
                    }}
                }}
                
                /* Animations */
                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(20px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
                
                .section {{
                    animation: fadeIn 0.6s ease-out;
                }}
                
                .summary-card {{
                    animation: fadeIn 0.8s ease-out;
                }}
                
                /* Additional Styles */
                .metric-detail {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px 15px;
                    background: white;
                    margin: 8px 0;
                    border-radius: 8px;
                    border-left: 4px solid #3b82f6;
                }}
                
                .metric-name {{
                    font-weight: 600;
                    color: #1e3a8a;
                }}
                
                .metric-value {{
                    color: #10b981;
                    font-weight: 600;
                }}
                
                .print-button {{
                    position: fixed;
                    bottom: 30px;
                    right: 30px;
                    background: linear-gradient(135deg, #3b82f6, #1d4ed8);
                    color: white;
                    border: none;
                    padding: 15px 25px;
                    border-radius: 50px;
                    font-weight: 600;
                    cursor: pointer;
                    box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
                    z-index: 1000;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }}
                
                .print-button:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 12px 30px rgba(59, 130, 246, 0.6);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <!-- Header -->
                <div class="header">
                    <h1>🔒 CV Model Security Test Report</h1>
                    <p>Comprehensive Security Assessment • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <!-- Main Content -->
                <div class="content">
                    <!-- Executive Summary -->
                    <div class="section">
                        <h2>📋 Executive Summary</h2>
                        <div class="summary-grid">
                            <div class="summary-card total">
                                <div class="card-value">{total_tests}</div>
                                <div class="card-label">Total Tests</div>
                            </div>
                            <div class="summary-card passed">
                                <div class="card-value">{passed_tests}</div>
                                <div class="card-label">Tests Passed</div>
                            </div>
                            <div class="summary-card failed">
                                <div class="card-value">{total_tests - passed_tests}</div>
                                <div class="card-label">Tests Failed</div>
                            </div>
                            <div class="summary-card rate">
                                <div class="card-value">{(passed_tests/total_tests*100):.1f}%</div>
                                <div class="card-label">Success Rate</div>
                            </div>
                        </div>
                        
                        {metrics_html}
                    </div>
                    
                    <!-- Model Information -->
                    <div class="section">
                        <h2>📊 Model Information</h2>
                        <table class="results-table">
                            <thead>
                                <tr>
                                    <th>Property</th>
                                    <th>Value</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Format</strong></td>
                                    <td>{model_info.get('format', 'Unknown')}</td>
                                </tr>
                                <tr>
                                    <td><strong>Framework</strong></td>
                                    <td>{model_info.get('framework', 'Unknown')}</td>
                                </tr>
                                <tr>
                                    <td><strong>File Size</strong></td>
                                    <td>{model_info.get('file_size', 0) / 1024 / 1024:.2f} MB</td>
                                </tr>
                                <tr>
                                    <td><strong>Input Shape</strong></td>
                                    <td>{str(model_info.get('input_shape', 'Unknown'))}</td>
                                </tr>
                                <tr>
                                    <td><strong>Output Shape</strong></td>
                                    <td>{str(model_info.get('output_shape', 'Unknown'))}</td>
                                </tr>
                                <tr>
                                    <td><strong>File Hash</strong></td>
                                    <td>{model_info.get('file_hash', 'Unknown')[:16]}...</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Test Results -->
                    <div class="section">
                        <h2>🧪 Test Results</h2>
                        <table class="results-table">
                            <thead>
                                <tr>
                                    <th>Test</th>
                                    <th>Status</th>
                                    <th>Details</th>
                                    <th>Metrics</th>
                                </tr>
                            </thead>
                            <tbody>
        """
        
        # Add test results rows
        for test_name, result in test_results.items():
            test_display = test_name.replace('_', ' ').title()
            status_class = "status-pass" if result.get('success', False) else "status-fail"
            status_text = "PASS" if result.get('success', False) else "FAIL"
            
            details = result.get('error', 'N/A')
            metrics = ""
            
            if result.get('success', False):
                if 'accuracy_drop' in result:
                    metrics = f"Drop: {result['accuracy_drop']:.2%}"
                    details = f"Original: {result['original_accuracy']:.2%}, Adv: {result['adversarial_accuracy']:.2%}"
                elif 'baseline_accuracy' in result:
                    metrics = f"Baseline: {result['baseline_accuracy']:.2%}"
                    details = f"{len(result.get('noise_tests', {}))} noise tests"
                elif 'crash_rate' in result:
                    metrics = f"Crash Rate: {result['crash_rate']:.2%}"
                    details = f"Samples: {result['num_samples']}"
                elif 'integrity_score' in result:
                    metrics = f"Score: {result['integrity_score']}/100"
                    details = "Integrity check completed"
            
            html += f"""
                                <tr>
                                    <td><strong>{test_display}</strong></td>
                                    <td><span class="status-badge {status_class}">{status_text}</span></td>
                                    <td>{details}</td>
                                    <td>{metrics}</td>
                                </tr>
            """
        
        html += """
                            </tbody>
                        </table>
                    </div>
                    
                    <!-- Visualizations -->
        """
        
        if plot_images_html:
            html += f"""
                    <div class="section">
                        <h2>📈 Security Visualizations</h2>
                        <div class="plots-container">
                            {plot_images_html}
                        </div>
                    </div>
            """
        
        html += f"""
                    <!-- Recommendations -->
                    <div class="recommendations">
                        <h2>💡 Security Recommendations</h2>
                        <ul class="recommendations-list">
        """
        
        if passed_tests == total_tests:
            html += """
                            <li>All security tests passed successfully - excellent model security posture</li>
                            <li>Continue regular security monitoring and vulnerability assessments</li>
                            <li>Schedule quarterly security audits to maintain robust defenses</li>
                            <li>Keep all security frameworks and dependencies up to date</li>
                            <li>Implement continuous security testing in your deployment pipeline</li>
            """
        else:
            html += f"""
                            <li>Address {total_tests - passed_tests} failed security tests immediately</li>
                            <li>Implement adversarial training to improve model robustness</li>
                            <li>Add input validation and sanitization layers to prevent attacks</li>
                            <li>Perform thorough penetration testing before production deployment</li>
                            <li>Schedule monthly security reviews and threat modeling sessions</li>
                            <li>Consider implementing ensemble defenses for enhanced security</li>
            """
        
        html += """
                        </ul>
                    </div>
                    
                    <!-- Disclaimer -->
                    <div class="disclaimer">
                        <strong>⚠️ ETHICAL DISCLAIMER</strong>
                        <p>
                            This report is generated for defensive security testing purposes only. 
                            No actual exploitation or malicious attacks were performed. All tests 
                            are simulations designed to identify vulnerabilities for remediation 
                            purposes. Use responsibly and only on models you own or have explicit 
                            permission to test.
                        </p>
                    </div>
                </div>
            </div>
            
            <!-- Print Button -->
            <button class="print-button no-print" onclick="window.print()">
                🖨️ Print Report
            </button>
            
            <script>
                // Add interactivity
                document.addEventListener('DOMContentLoaded', function() {{
                    // Add animation delay to cards
                    const cards = document.querySelectorAll('.summary-card');
                    cards.forEach((card, index) => {{
                        card.style.animationDelay = `${{index * 0.1}}s`;
                    }});
                    
                    // Add hover effect to visualization cards
                    const plotCards = document.querySelectorAll('.plot-card');
                    plotCards.forEach(card => {{
                        card.addEventListener('mouseenter', function() {{
                            this.style.transform = 'translateY(-10px) scale(1.02)';
                        }});
                        
                        card.addEventListener('mouseleave', function() {{
                            this.style.transform = 'translateY(0) scale(1)';
                        }});
                    }});
                    
                    // Smooth scroll for anchor links
                    document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
                        anchor.addEventListener('click', function (e) {{
                            e.preventDefault();
                            const target = document.querySelector(this.getAttribute('href'));
                            if (target) {{
                                target.scrollIntoView({{
                                    behavior: 'smooth',
                                    block: 'start'
                                }});
                            }}
                        }});
                    }});
                    
                    // Add copy-to-clipboard for important data
                    const tableCells = document.querySelectorAll('.results-table td');
                    tableCells.forEach(cell => {{
                        cell.addEventListener('click', function() {{
                            const text = this.textContent.trim();
                            if (text && text.length > 0) {{
                                navigator.clipboard.writeText(text).then(() => {{
                                    const originalText = this.textContent;
                                    this.textContent = 'Copied!';
                                    this.style.backgroundColor = '#d1fae5';
                                    setTimeout(() => {{
                                        this.textContent = originalText;
                                        this.style.backgroundColor = '';
                                    }}, 1500);
                                }});
                            }}
                        }});
                    }});
                }});
                
                // Print optimization
                window.onbeforeprint = function() {{
                    document.body.classList.add('printing');
                }};
                
                window.onafterprint = function() {{
                    document.body.classList.remove('printing');
                }};
            </script>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path