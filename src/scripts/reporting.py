# D:\Project_Trace_Finder\src\scripts\reporting.py

from fpdf import FPDF
import pandas as pd
from datetime import datetime
import os
import cv2
import tempfile

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'AI TraceFinder - Forensic Report', 0, 1, 'C')
        self.set_font('Arial', '', 8)
        self.cell(0, 5, f'Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(report_data):
    pdf = PDF()
    pdf.add_page()
    
    # --- 1. Summary Section ---
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '1. Prediction Summary', 0, 1)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(50, 10, 'Scanner Prediction:', 0, 0)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, str(report_data.get('scanner_prediction', 'N/A')), 0, 1)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(50, 10, 'Confidence:', 0, 0)
    pdf.cell(0, 10, f"{report_data.get('scanner_confidence', 0.0):.2f}%", 0, 1)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(50, 10, 'Tamper Status:', 0, 0)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, str(report_data.get('tamper_label', 'N/A')), 0, 1)
    
    pdf.set_font('Arial', '', 12)
    pdf.cell(50, 10, 'Tamper Confidence:', 0, 0)
    pdf.cell(0, 10, f"{report_data.get('tamper_confidence', 0.0):.2f}%", 0, 1)
    
    pdf.ln(10)

    # --- 2. Image Evidence ---
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '2. Image Evidence', 0, 1)
    
    # Save residual to a temp file for embedding
    residual_display = cv2.normalize(report_data['residual_image'], None, 0, 255, cv2.NORM_MINMAX)
    
    # Use tempfile for robust path handling
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_res:
        cv2.imwrite(tmp_res.name, residual_display)
        res_path = tmp_res.name
    
    pdf.image(report_data['image_path'], w=80, h=80)
    pdf.image(res_path, x=pdf.get_x() + 90, y=pdf.get_y(), w=80, h=80)
    pdf.ln(85)
    
    # Clean up temp residual file
    if os.path.exists(res_path):
        os.remove(res_path)

    # --- 3. Full Probability Table ---
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, '3. Full Probability Distribution', 0, 1)
    
    pdf.set_font('Arial', 'B', 10)
    pdf.cell(120, 8, 'Scanner Class', 1)
    pdf.cell(0, 8, 'Confidence (%)', 1, 1, 'C')
    
    pdf.set_font('Arial', '', 10)
    prob_df = pd.DataFrame(list(report_data['probabilities'].items()), columns=['Class', 'Probability'])
    prob_df['Confidence'] = prob_df['Probability'] * 100
    prob_df = prob_df.sort_values(by='Confidence', ascending=False)
    
    for _, row in prob_df.iterrows():
        pdf.cell(120, 8, row['Class'], 1)
        pdf.cell(0, 8, f"{row['Confidence']:.3f}%", 1, 1, 'C')
        
    # Return PDF as bytes
    return bytes(pdf.output(dest='S'))