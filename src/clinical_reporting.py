import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import json
import traceback

class ClinicalReport:
    """Generate comprehensive clinical reports"""
    
    def __init__(self, output_folder):
        self.output_folder = output_folder
        os.makedirs(os.path.join(output_folder, 'clinical_reports'), exist_ok=True)
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
    def _setup_custom_styles(self):
        """Setup custom styles for the report"""
        self.styles.add(ParagraphStyle(
            name='Warning',
            parent=self.styles['Normal'],
            textColor=colors.red,
            spaceAfter=10
        ))
        self.styles.add(ParagraphStyle(
            name='Clinical',
            parent=self.styles['Normal'],
            textColor=colors.HexColor('#2E5C8A'),
            spaceAfter=10
        ))
    
    def generate_clinical_report(self, prediction_results, clinical_results, validation_results, patient_data=None):
        """Generate comprehensive clinical report"""
        try:
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'prediction_results': prediction_results,
                'clinical_metrics': clinical_results,
                'validation_results': validation_results
            }
            
            # Create report folder
            report_folder = os.path.join(self.output_folder, 'clinical_reports')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = os.path.join(report_folder, f'clinical_report_{timestamp}')
            os.makedirs(report_path, exist_ok=True)
            
            # Save report data
            with open(os.path.join(report_path, 'report_data.json'), 'w') as f:
                json.dump(report_data, f, indent=4)
            
            # Generate visualizations
            self._generate_prediction_visualizations(
                prediction_results['prediction'],
                prediction_results.get('uncertainties', []),
                report_path
            )
            
            self._generate_clinical_metrics_visualizations(
                clinical_results,
                report_path
            )
            
            # Add patient data analysis if available
            if patient_data is not None and not patient_data.empty:
                self._generate_patient_analysis(patient_data, report_path)
            
            # Generate summary report
            self._generate_summary_report(report_data, report_path)
            
            print(f"Clinical report generated at: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"Error generating clinical report: {str(e)}")
            traceback.print_exc()
            return None
            
    def _generate_prediction_visualizations(self, predictions, uncertainties, report_path):
        """Generate visualizations for predictions"""
        plt.figure(figsize=(10, 6))
        
        # Handle single value predictions
        if isinstance(predictions, (float, int)):
            predictions = [predictions]
        
        # Prediction distribution
        try:
            sns.kdeplot(predictions, fill=True, warn_singular=False)
            plt.title('Prediction Distribution')
            plt.xlabel('Prediction Value')
            plt.ylabel('Density')
        except Exception as e:
            print(f"Warning: Could not generate prediction distribution plot: {str(e)}")
            
        plt.savefig(os.path.join(report_path, 'prediction_distribution.png'))
        plt.close()
        
        # Uncertainty visualization if available
        if uncertainties:
            plt.figure(figsize=(10, 6))
            try:
                sns.kdeplot(uncertainties, fill=True, warn_singular=False)
                plt.title('Uncertainty Distribution')
                plt.xlabel('Uncertainty Value')
                plt.ylabel('Density')
            except Exception as e:
                print(f"Warning: Could not generate uncertainty distribution plot: {str(e)}")
                
            plt.savefig(os.path.join(report_path, 'uncertainty_distribution.png'))
            plt.close()
    
    def _generate_clinical_metrics_visualizations(self, clinical_results, report_path):
        """Generate visualizations for clinical metrics"""
        # Create metrics summary plot
        plt.figure(figsize=(12, 6))
        metrics = clinical_results.get('diagnostic_metrics', {})
        
        if metrics:
            metrics_values = [v for v in metrics.values() if isinstance(v, (int, float))]
            metrics_names = [k for k, v in metrics.items() if isinstance(v, (int, float))]
            
            if metrics_values and metrics_names:
                plt.bar(metrics_names, metrics_values)
                plt.xticks(rotation=45)
                plt.title('Clinical Metrics Summary')
                plt.tight_layout()
                plt.savefig(os.path.join(report_path, 'clinical_metrics.png'))
        
        plt.close()
    
    def _generate_patient_analysis(self, patient_data, report_path):
        """Generate patient-specific analysis"""
        try:
            # Basic statistics
            stats = patient_data.describe()
            stats.to_csv(os.path.join(report_path, 'patient_statistics.csv'))
            
            # Correlation heatmap
            plt.figure(figsize=(12, 8))
            numeric_data = patient_data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', center=0)
                plt.title('Patient Features Correlation')
                plt.tight_layout()
                plt.savefig(os.path.join(report_path, 'patient_correlations.png'))
            plt.close()
            
        except Exception as e:
            print(f"Warning: Could not generate patient analysis: {str(e)}")
    
    def _generate_summary_report(self, report_data, report_path):
        """Generate text summary report"""
        summary = []
        summary.append("Clinical Report Summary")
        summary.append("=" * 50)
        summary.append(f"Generated on: {report_data['timestamp']}")
        summary.append("\nPrediction Results:")
        summary.append(f"Mean Prediction: {report_data['prediction_results']['prediction']:.3f}")
        
        if 'clinical_metrics' in report_data:
            summary.append("\nClinical Metrics:")
            for metric, value in report_data['clinical_metrics'].get('diagnostic_metrics', {}).items():
                if isinstance(value, (int, float)):
                    summary.append(f"{metric}: {value:.3f}")
        
        if 'validation_results' in report_data:
            summary.append("\nValidation Status:")
            status = report_data['validation_results'].get('validation_status', {})
            summary.append(f"Status: {status.get('status', 'Unknown')}")
        
        with open(os.path.join(report_path, 'summary.txt'), 'w') as f:
            f.write('\n'.join(summary))
    
    def _create_header_section(self):
        """Create report header section"""
        elements = []
        elements.append(Paragraph("Clinical Analysis Report", self.styles['Title']))
        elements.append(Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 20))
        return elements
    
    def _create_prediction_section(self, prediction_results):
        """Create prediction results section"""
        elements = []
        elements.append(Paragraph("Prediction Results", self.styles['Heading1']))
        
        # Add prediction details
        if prediction_results['prediction'] > 0.7:
            risk_level = "High Risk"
            style = self.styles['Warning']
        else:
            risk_level = "Low Risk"
            style = self.styles['Clinical']
            
        elements.append(Paragraph(f"Risk Level: {risk_level}", style))
        elements.append(Paragraph(
            f"Prediction Score: {prediction_results['prediction']:.3f}",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 10))
        return elements
    
    def _create_metrics_section(self, metrics):
        """Create metrics section"""
        elements = []
        elements.append(Paragraph("Clinical Metrics", self.styles['Heading1']))
        
        # Add diagnostic metrics table
        diagnostic = metrics['diagnostic_metrics']
        data = [
            ['Metric', 'Value'],
            ['Sensitivity', f"{diagnostic['sensitivity']:.3f}"],
            ['Specificity', f"{diagnostic['specificity']:.3f}"],
            ['PPV', f"{diagnostic['ppv']:.3f}"],
            ['NPV', f"{diagnostic['npv']:.3f}"]
        ]
        
        table = Table(data)
        elements.append(table)
        elements.append(Spacer(1, 10))
        return elements
    
    def _create_validation_section(self, validation):
        """Create validation section"""
        elements = []
        elements.append(Paragraph("Validation Results", self.styles['Heading1']))
        
        status = validation['validation_status']
        if status['status'] == 'PASSED':
            elements.append(Paragraph(
                "✓ Model passed clinical validation",
                self.styles['Clinical']
            ))
        else:
            elements.append(Paragraph(
                "⚠ Model failed clinical validation",
                self.styles['Warning']
            ))
            
        elements.append(Spacer(1, 10))
        return elements
    
    def _create_visualization_section(self):
        """Add visualizations to report"""
        elements = []
        elements.append(Paragraph("Visual Analysis", self.styles['Heading1']))
        
        # Add generated plots
        for plot_name in ['risk_assessment.png', 'performance_metrics.png',
                         'validation_results.png', 'uncertainty_analysis.png']:
            img_path = os.path.join(self.viz_folder, plot_name)
            if os.path.exists(img_path):
                img = Image(img_path, width=400, height=300)
                elements.append(img)
                elements.append(Spacer(1, 10))
                
        return elements 