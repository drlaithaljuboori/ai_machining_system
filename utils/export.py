# utils/export.py
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import datetime
from dataclasses import dataclass, asdict
import xml.etree.ElementTree as ET
from xml.dom import minidom
import yaml
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import base64
from io import BytesIO

@dataclass
class ExportConfig:
    """Configuration for export operations"""
    output_format: str = "all"  # "all", "pdf", "json", "xml", "excel"
    include_visualizations: bool = True
    include_raw_data: bool = False
    compression: bool = False
    quality: str = "high"  # "high", "medium", "low"

class ReportGenerator:
    """
    Comprehensive report generator for machining analysis results
    Supports multiple formats: PDF, JSON, XML, Excel, and CAM-ready formats
    """
    
    def __init__(self, config: ExportConfig = None):
        self.config = config or ExportConfig()
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_pdf_report(self, results: Dict, output_path: str = None) -> str:
        """
        Generate comprehensive PDF report with analysis results
        
        Args:
            results: Complete analysis results
            output_path: Path to save the PDF report
            
        Returns:
            Path to the generated PDF file
        """
        if output_path is None:
            output_path = f"machining_analysis_report_{self.timestamp}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build story (content elements)
        story = []
        styles = getSampleStyleSheet()
        
        # Add custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title page
        story.append(Paragraph("AI-Powered Machining Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Add metadata table
        metadata = self._extract_metadata(results)
        story.append(self._create_metadata_table(metadata, styles))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        summary_text = self._generate_executive_summary(results)
        story.append(Paragraph(summary_text, styles["Normal"]))
        story.append(Spacer(1, 20))
        
        # Geometry Analysis Section
        story.append(Paragraph("Geometry Analysis", heading_style))
        geometry_data = self._extract_geometry_analysis(results)
        story.append(self._create_geometry_table(geometry_data, styles))
        story.append(Spacer(1, 20))
        
        # Tool Recommendations Section
        story.append(Paragraph("Tool Recommendations", heading_style))
        tool_data = self._extract_tool_recommendations(results)
        story.append(self._create_tool_recommendations_table(tool_data, styles))
        story.append(Spacer(1, 20))
        
        # Machining Parameters Section
        story.append(Paragraph("Optimized Machining Parameters", heading_style))
        param_data = self._extract_machining_parameters(results)
        
        for operation, tools in param_data.items():
            story.append(Paragraph(f"Operation: {operation.title()}", styles["Heading3"]))
            story.append(self._create_parameters_table(tools, styles))
            story.append(Spacer(1, 12))
        
        # Performance Metrics Section
        story.append(Paragraph("Performance Metrics", heading_style))
        metrics_data = self._extract_performance_metrics(results)
        story.append(self._create_metrics_table(metrics_data, styles))
        story.append(Spacer(1, 20))
        
        # Recommendations Section
        story.append(Paragraph("Machining Recommendations", heading_style))
        recommendations = self._generate_recommendations(results)
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles["Normal"]))
        
        # Build PDF
        try:
            doc.build(story)
            print(f"PDF report generated: {output_path}")
            return output_path
        except Exception as e:
            print(f"Error generating PDF report: {str(e)}")
            return None
    
    def generate_json_export(self, results: Dict, output_path: str = None) -> str:
        """
        Generate JSON export of analysis results
        
        Args:
            results: Complete analysis results
            output_path: Path to save the JSON file
            
        Returns:
            Path to the generated JSON file
        """
        if output_path is None:
            output_path = f"machining_analysis_{self.timestamp}.json"
        
        try:
            # Prepare data for JSON serialization
            export_data = self._prepare_json_data(results)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=self._json_serializer)
            
            print(f"JSON export generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating JSON export: {str(e)}")
            return None
    
    def generate_xml_export(self, results: Dict, output_path: str = None) -> str:
        """
        Generate XML export of analysis results
        
        Args:
            results: Complete analysis results
            output_path: Path to save the XML file
            
        Returns:
            Path to the generated XML file
        """
        if output_path is None:
            output_path = f"machining_analysis_{self.timestamp}.xml"
        
        try:
            root = ET.Element("MachiningAnalysis")
            root.set("version", "1.0")
            root.set("timestamp", self.timestamp)
            
            # Add metadata
            metadata_elem = ET.SubElement(root, "Metadata")
            metadata = self._extract_metadata(results)
            for key, value in metadata.items():
                ET.SubElement(metadata_elem, key).text = str(value)
            
            # Add geometry analysis
            geometry_elem = ET.SubElement(root, "GeometryAnalysis")
            geometry_data = self._extract_geometry_analysis(results)
            for key, value in geometry_data.items():
                ET.SubElement(geometry_elem, key).text = str(value)
            
            # Add tool recommendations
            tools_elem = ET.SubElement(root, "ToolRecommendations")
            tool_data = self._extract_tool_recommendations(results)
            for operation in tool_data:
                op_elem = ET.SubElement(tools_elem, "Operation")
                op_elem.set("type", operation["type"])
                for tool in operation["tools"]:
                    tool_elem = ET.SubElement(op_elem, "Tool")
                    for key, value in tool.items():
                        ET.SubElement(tool_elem, key).text = str(value)
            
            # Add machining parameters
            params_elem = ET.SubElement(root, "MachiningParameters")
            param_data = self._extract_machining_parameters(results)
            for operation, tools in param_data.items():
                op_elem = ET.SubElement(params_elem, "Operation")
                op_elem.set("name", operation)
                for tool_params in tools:
                    tool_elem = ET.SubElement(op_elem, "ToolParameters")
                    for key, value in tool_params.items():
                        ET.SubElement(tool_elem, key).text = str(value)
            
            # Format and save XML
            xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(xml_str)
            
            print(f"XML export generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating XML export: {str(e)}")
            return None
    
    def generate_excel_export(self, results: Dict, output_path: str = None) -> str:
        """
        Generate Excel export with multiple sheets
        
        Args:
            results: Complete analysis results
            output_path: Path to save the Excel file
            
        Returns:
            Path to the generated Excel file
        """
        if output_path is None:
            output_path = f"machining_analysis_{self.timestamp}.xlsx"
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                
                # Sheet 1: Summary
                summary_data = self._prepare_summary_sheet(results)
                summary_df = pd.DataFrame([summary_data])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Sheet 2: Tool Recommendations
                tool_data = self._prepare_tool_sheet(results)
                if tool_data:
                    tool_df = pd.DataFrame(tool_data)
                    tool_df.to_excel(writer, sheet_name='Tool_Recommendations', index=False)
                
                # Sheet 3: Machining Parameters
                param_data = self._prepare_parameters_sheet(results)
                if param_data:
                    param_df = pd.DataFrame(param_data)
                    param_df.to_excel(writer, sheet_name='Machining_Parameters', index=False)
                
                # Sheet 4: Performance Metrics
                metrics_data = self._prepare_metrics_sheet(results)
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_df.to_excel(writer, sheet_name='Performance_Metrics', index=False)
                
                # Sheet 5: Geometry Analysis
                geometry_data = self._prepare_geometry_sheet(results)
                if geometry_data:
                    geometry_df = pd.DataFrame([geometry_data])
                    geometry_df.to_excel(writer, sheet_name='Geometry_Analysis', index=False)
            
            print(f"Excel export generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating Excel export: {str(e)}")
            return None
    
    def generate_cam_ready_export(self, results: Dict, output_path: str = None) -> str:
        """
        Generate CAM-ready export in standard formats
        
        Args:
            results: Complete analysis results
            output_path: Path to save the CAM file
            
        Returns:
            Path to the generated CAM file
        """
        if output_path is None:
            output_path = f"cam_ready_export_{self.timestamp}.json"
        
        try:
            cam_data = self._prepare_cam_data(results)
            
            # Export as JSON (can be extended to other CAM formats)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(cam_data, f, indent=2, default=self._json_serializer)
            
            print(f"CAM-ready export generated: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error generating CAM-ready export: {str(e)}")
            return None
    
    def generate_comprehensive_export(self, results: Dict, output_dir: str = "./exports") -> Dict[str, str]:
        """
        Generate all export formats in a comprehensive package
        
        Args:
            results: Complete analysis results
            output_dir: Directory to save all exports
            
        Returns:
            Dictionary of generated file paths
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        export_files = {}
        
        try:
            # Generate all export formats
            export_files['pdf'] = self.generate_pdf_report(
                results, f"{output_dir}/machining_analysis_report_{self.timestamp}.pdf"
            )
            
            export_files['json'] = self.generate_json_export(
                results, f"{output_dir}/machining_analysis_{self.timestamp}.json"
            )
            
            export_files['xml'] = self.generate_xml_export(
                results, f"{output_dir}/machining_analysis_{self.timestamp}.xml"
            )
            
            export_files['excel'] = self.generate_excel_export(
                results, f"{output_dir}/machining_analysis_{self.timestamp}.xlsx"
            )
            
            export_files['cam'] = self.generate_cam_ready_export(
                results, f"{output_dir}/cam_ready_{self.timestamp}.json"
            )
            
            # Generate YAML config if needed
            config_file = f"{output_dir}/export_config_{self.timestamp}.yaml"
            with open(config_file, 'w') as f:
                yaml.dump(asdict(self.config), f)
            export_files['config'] = config_file
            
            print(f"Comprehensive export package generated in: {output_dir}")
            return export_files
            
        except Exception as e:
            print(f"Error generating comprehensive export: {str(e)}")
            return {}
    
    def _extract_metadata(self, results: Dict) -> Dict[str, Any]:
        """Extract metadata from analysis results"""
        metadata = {
            "report_id": f"MA_{self.timestamp}",
            "generation_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "system_version": "AI Machining System 1.0.0",
            "analysis_type": "Geometric Analysis & Tool Selection"
        }
        
        # Add material and machine info if available
        optimized_params = results.get('optimized_parameters', {})
        if optimized_params:
            metadata.update({
                "material": optimized_params.get('material_used', 'Unknown'),
                "machine_type": optimized_params.get('machine_type', 'Unknown'),
                "optimization_objective": optimized_params.get('optimization_objective', 'balanced')
            })
        
        return metadata
    
    def _extract_geometry_analysis(self, results: Dict) -> Dict[str, Any]:
        """Extract geometry analysis data"""
        geometry_data = {
            "part_complexity": "Medium",
            "curvature_distribution": "Mixed",
            "feature_count": "Unknown",
            "surface_area": "Unknown",
            "volume": "Unknown"
        }
        
        mesh_data = results.get('mesh_data', {})
        if mesh_data and 'vertices' in mesh_data:
            vertices = mesh_data['vertices']
            if hasattr(vertices, 'cpu'):
                vertices = vertices.cpu().numpy()
            geometry_data['vertex_count'] = len(vertices)
        
        region_classification = results.get('region_classification', {})
        if region_classification:
            total_regions = sum(len(indices) for indices in region_classification.values() 
                              if hasattr(indices, '__len__'))
            geometry_data['analyzed_regions'] = total_regions
            
            # Calculate complexity based on curvature regions
            if 'high_curvature' in region_classification:
                hc_count = len(region_classification['high_curvature'])
                complexity = "High" if hc_count > total_regions * 0.3 else "Medium"
                geometry_data['part_complexity'] = complexity
        
        return geometry_data
    
    def _extract_tool_recommendations(self, results: Dict) -> List[Dict]:
        """Extract tool recommendation data"""
        tool_recommendations = results.get('tool_recommendations', {})
        tool_sequence = tool_recommendations.get('tool_sequence', [])
        
        extracted_data = []
        for operation in tool_sequence:
            op_data = {
                "type": operation['operation'],
                "strategy": operation.get('strategy', 'standard'),
                "tools": []
            }
            
            for tool in operation.get('tools', []):
                tool_data = {
                    "name": tool.get('name', 'Unknown'),
                    "type": tool.get('type', 'end_mill'),
                    "diameter": tool.get('diameter', 0),
                    "flutes": tool.get('flutes', 2),
                    "coating": tool.get('coating', 'Unknown')
                }
                op_data["tools"].append(tool_data)
            
            extracted_data.append(op_data)
        
        return extracted_data
    
    def _extract_machining_parameters(self, results: Dict) -> Dict[str, List]:
        """Extract machining parameters data"""
        optimized_params = results.get('optimized_parameters', {})
        operations = optimized_params.get('optimized_operations', {})
        
        extracted_data = {}
        for op_name, tools in operations.items():
            tool_params_list = []
            for tool_data in tools:
                cutting_params = tool_data.get('cutting_parameters', {})
                tool_info = tool_data.get('tool', {})
                
                param_data = {
                    "tool_name": tool_info.get('name', 'Unknown'),
                    "tool_diameter": tool_info.get('diameter', 0),
                    "rpm": cutting_params.get('rpm', 0),
                    "feed_rate": cutting_params.get('feed_rate', 0),
                    "feed_per_tooth": cutting_params.get('feed_per_tooth', 0),
                    "axial_doc": cutting_params.get('axial_depth_of_cut', 0),
                    "radial_doc": cutting_params.get('radial_depth_of_cut', 0),
                    "stepover_percent": cutting_params.get('stepover', 0)
                }
                tool_params_list.append(param_data)
            
            extracted_data[op_name] = tool_params_list
        
        return extracted_data
    
    def _extract_performance_metrics(self, results: Dict) -> Dict[str, Any]:
        """Extract performance metrics data"""
        optimized_params = results.get('optimized_parameters', {})
        optimization_summary = optimized_params.get('optimization_summary', {})
        
        metrics = {
            "total_operations": optimization_summary.get('total_operations', 0),
            "total_tools": optimization_summary.get('total_tools', 0),
            "average_mrr": optimization_summary.get('average_mrr', 0),
            "total_power": optimization_summary.get('total_power_requirement', 0),
            "estimated_time": optimization_summary.get('estimated_machining_time', 0),
            "optimization_success": optimization_summary.get('optimization_success', False)
        }
        
        return metrics
    
    def _generate_executive_summary(self, results: Dict) -> str:
        """Generate executive summary text"""
        geometry_data = self._extract_geometry_analysis(results)
        metrics_data = self._extract_performance_metrics(results)
        tool_data = self._extract_tool_recommendations(results)
        
        total_tools = sum(len(op['tools']) for op in tool_data)
        total_operations = len(tool_data)
        
        summary = f"""
        This report presents the AI-powered machining analysis for the specified part geometry. 
        The analysis identified {geometry_data.get('vertex_count', 'unknown')} vertices and 
        classified curvature regions to determine optimal machining strategies.
        
        The system recommends {total_operations} machining operations using {total_tools} different tools. 
        Estimated machining time is approximately {metrics_data.get('estimated_time', 0)} minutes with 
        an average material removal rate of {metrics_data.get('average_mrr', 0):.1f} cm³/min.
        
        All parameters have been optimized for {results.get('optimized_parameters', {}).get('optimization_objective', 'balanced')} 
        performance considering the specified material and machine constraints.
        """
        
        return summary.strip()
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """Generate machining recommendations"""
        recommendations = []
        
        # Geometry-based recommendations
        geometry_data = self._extract_geometry_analysis(results)
        if geometry_data.get('part_complexity') == 'High':
            recommendations.append("Consider using high-speed machining strategies for complex curvature regions")
            recommendations.append("Use smaller stepovers in high-curvature areas for better surface finish")
        
        # Tool-based recommendations
        tool_data = self._extract_tool_recommendations(results)
        for operation in tool_data:
            if 'finishing' in operation['type'].lower():
                recommendations.append(f"Ensure proper tool runout control for {operation['type']} operations")
            if 'roughing' in operation['type'].lower():
                recommendations.append(f"Monitor tool wear regularly during {operation['type']} operations")
        
        # Material-based recommendations
        material = results.get('optimized_parameters', {}).get('material_used', '').lower()
        if 'titanium' in material or 'inconel' in material:
            recommendations.append("Use high-pressure coolant for difficult-to-machine materials")
            recommendations.append("Consider trochoidal milling for high-strength materials")
        
        # General recommendations
        recommendations.extend([
            "Verify all parameters with machine operator experience",
            "Consider workpiece fixturing and rigidity in final setup",
            "Perform test cuts for critical dimensions",
            "Monitor cutting forces and adjust parameters as needed"
        ])
        
        return recommendations
    
    def _create_metadata_table(self, metadata: Dict, styles) -> Table:
        """Create metadata table for PDF report"""
        data = [["Field", "Value"]]
        for key, value in metadata.items():
            data.append([key.replace('_', ' ').title(), str(value)])
        
        table = Table(data, colWidths=[2*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _create_geometry_table(self, geometry_data: Dict, styles) -> Table:
        """Create geometry analysis table for PDF report"""
        data = [["Geometry Property", "Value"]]
        for key, value in geometry_data.items():
            data.append([key.replace('_', ' ').title(), str(value)])
        
        table = Table(data, colWidths=[2.5*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _create_tool_recommendations_table(self, tool_data: List[Dict], styles) -> Table:
        """Create tool recommendations table for PDF report"""
        data = [["Operation", "Strategy", "Tool Name", "Diameter (mm)", "Type", "Flutes"]]
        
        for operation in tool_data:
            for tool in operation['tools']:
                data.append([
                    operation['type'].title(),
                    operation['strategy'].replace('_', ' ').title(),
                    tool['name'],
                    f"{tool['diameter']:.1f}",
                    tool['type'].replace('_', ' ').title(),
                    str(tool['flutes'])
                ])
        
        table = Table(data, colWidths=[1*inch, 1.2*inch, 1.5*inch, 0.8*inch, 1*inch, 0.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 9)
        ]))
        
        return table
    
    def _create_parameters_table(self, tools: List[Dict], styles) -> Table:
        """Create machining parameters table for PDF report"""
        data = [["Tool", "RPM", "Feed Rate", "FPT", "Axial DOC", "Radial DOC", "Stepover %"]]
        
        for tool in tools:
            data.append([
                tool['tool_name'],
                f"{tool['rpm']:,}",
                f"{tool['feed_rate']:.0f}",
                f"{tool['feed_per_tooth']:.3f}",
                f"{tool['axial_doc']:.2f}",
                f"{tool['radial_doc']:.2f}",
                f"{tool['stepover_percent']:.1f}"
            ])
        
        table = Table(data, colWidths=[1.2*inch, 0.7*inch, 0.8*inch, 0.6*inch, 0.7*inch, 0.8*inch, 0.7*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 8)
        ]))
        
        return table
    
    def _create_metrics_table(self, metrics_data: Dict, styles) -> Table:
        """Create performance metrics table for PDF report"""
        data = [["Metric", "Value"]]
        
        for key, value in metrics_data.items():
            if key == "optimization_success":
                display_value = "Yes" if value else "No"
            elif isinstance(value, float):
                display_value = f"{value:.2f}"
            else:
                display_value = str(value)
            
            data.append([key.replace('_', ' ').title(), display_value])
        
        table = Table(data, colWidths=[2.5*inch, 2.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.purple),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        return table
    
    def _prepare_json_data(self, results: Dict) -> Dict:
        """Prepare data for JSON export"""
        export_data = {
            "metadata": self._extract_metadata(results),
            "geometry_analysis": self._extract_geometry_analysis(results),
            "tool_recommendations": self._extract_tool_recommendations(results),
            "machining_parameters": self._extract_machining_parameters(results),
            "performance_metrics": self._extract_performance_metrics(results),
            "recommendations": self._generate_recommendations(results),
            "raw_data": results if self.config.include_raw_data else None
        }
        
        return export_data
    
    def _prepare_summary_sheet(self, results: Dict) -> Dict:
        """Prepare data for Excel summary sheet"""
        metadata = self._extract_metadata(results)
        geometry = self._extract_geometry_analysis(results)
        metrics = self._extract_performance_metrics(results)
        
        summary_data = {
            "Report ID": metadata.get("report_id", ""),
            "Generation Date": metadata.get("generation_date", ""),
            "Material": metadata.get("material", "Unknown"),
            "Machine Type": metadata.get("machine_type", "Unknown"),
            "Part Complexity": geometry.get("part_complexity", "Unknown"),
            "Total Operations": metrics.get("total_operations", 0),
            "Total Tools": metrics.get("total_tools", 0),
            "Average MRR": metrics.get("average_mrr", 0),
            "Estimated Time": metrics.get("estimated_time", 0),
            "Optimization Success": "Yes" if metrics.get("optimization_success") else "No"
        }
        
        return summary_data
    
    def _prepare_tool_sheet(self, results: Dict) -> List[Dict]:
        """Prepare data for Excel tool recommendations sheet"""
        tool_data = self._extract_tool_recommendations(results)
        excel_data = []
        
        for operation in tool_data:
            for tool in operation["tools"]:
                row_data = {
                    "Operation": operation["type"].title(),
                    "Strategy": operation["strategy"].replace("_", " ").title(),
                    "Tool Name": tool["name"],
                    "Tool Type": tool["type"].replace("_", " ").title(),
                    "Diameter (mm)": tool["diameter"],
                    "Flutes": tool["flutes"],
                    "Coating": tool["coating"]
                }
                excel_data.append(row_data)
        
        return excel_data
    
    def _prepare_parameters_sheet(self, results: Dict) -> List[Dict]:
        """Prepare data for Excel machining parameters sheet"""
        param_data = self._extract_machining_parameters(results)
        excel_data = []
        
        for operation, tools in param_data.items():
            for tool in tools:
                row_data = {
                    "Operation": operation.title(),
                    "Tool Name": tool["tool_name"],
                    "Tool Diameter (mm)": tool["tool_diameter"],
                    "RPM": tool["rpm"],
                    "Feed Rate (mm/min)": tool["feed_rate"],
                    "Feed per Tooth (mm)": tool["feed_per_tooth"],
                    "Axial DOC (mm)": tool["axial_doc"],
                    "Radial DOC (mm)": tool["radial_doc"],
                    "Stepover (%)": tool["stepover_percent"]
                }
                excel_data.append(row_data)
        
        return excel_data
    
    def _prepare_metrics_sheet(self, results: Dict) -> List[Dict]:
        """Prepare data for Excel performance metrics sheet"""
        metrics_data = self._extract_performance_metrics(results)
        
        # Convert to list of dicts for Excel
        return [metrics_data]
    
    def _prepare_geometry_sheet(self, results: Dict) -> Dict:
        """Prepare data for Excel geometry analysis sheet"""
        return self._extract_geometry_analysis(results)
    
    def _prepare_cam_data(self, results: Dict) -> Dict:
        """Prepare CAM-ready data structure"""
        cam_data = {
            "setup_name": f"AI_Optimized_{self.timestamp}",
            "timestamp": self.timestamp,
            "operations": []
        }
        
        param_data = self._extract_machining_parameters(results)
        
        for operation, tools in param_data.items():
            op_data = {
                "operation_type": operation,
                "tools": []
            }
            
            for tool in tools:
                tool_data = {
                    "tool_id": tool["tool_name"],
                    "parameters": {
                        "spindle_speed": tool["rpm"],
                        "feed_rate": tool["feed_rate"],
                        "feed_per_tooth": tool["feed_per_tooth"],
                        "axial_depth_of_cut": tool["axial_doc"],
                        "radial_depth_of_cut": tool["radial_doc"],
                        "stepover": tool["stepover_percent"] / 100.0
                    }
                }
                op_data["tools"].append(tool_data)
            
            cam_data["operations"].append(op_data)
        
        return cam_data
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for non-serializable objects"""
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)

# Example usage and testing
if __name__ == "__main__":
    print("Testing Export Module")
    
    # Create test data
    test_results = {
        'mesh_data': {
            'vertices': np.random.rand(100, 3) * 100,
            'faces': np.random.randint(0, 100, (50, 3))
        },
        'curvature_map': {
            'gaussian_curvature': np.random.randn(100) * 0.1
        },
        'region_classification': {
            'flat_regions': np.array([0, 1, 2, 3]),
            'high_curvature': np.array([10, 11, 12])
        },
        'tool_recommendations': {
            'tool_sequence': [
                {
                    'operation': 'roughing',
                    'strategy': 'high_material_removal',
                    'tools': [
                        {
                            'name': 'EM_16mm_Flat',
                            'type': 'flat_end_mill',
                            'diameter': 16.0,
                            'flutes': 4,
                            'coating': 'TiAlN'
                        }
                    ]
                }
            ]
        },
        'optimized_parameters': {
            'material_used': 'aluminum',
            'machine_type': '5-axis',
            'optimization_objective': 'balanced',
            'optimized_operations': {
                'roughing': [
                    {
                        'tool': {'name': 'EM_16mm_Flat', 'diameter': 16.0},
                        'cutting_parameters': {
                            'rpm': 2000,
                            'feed_rate': 500,
                            'feed_per_tooth': 0.15,
                            'axial_depth_of_cut': 2.0,
                            'radial_depth_of_cut': 1.0,
                            'stepover': 50.0
                        },
                        'performance_metrics': {
                            'material_removal_rate': 45.5,
                            'estimated_tool_life': 120,
                            'cutting_power': 2.5
                        }
                    }
                ]
            },
            'optimization_summary': {
                'total_operations': 1,
                'total_tools': 1,
                'average_mrr': 45.5,
                'total_power_requirement': 2.5,
                'estimated_machining_time': 30,
                'optimization_success': True
            }
        }
    }
    
    # Test export functionality
    config = ExportConfig(
        output_format="all",
        include_visualizations=True,
        include_raw_data=True
    )
    
    exporter = ReportGenerator(config)
    
    # Test individual exports
    print("Testing PDF export...")
    pdf_path = exporter.generate_pdf_report(test_results, "./test_output/test_report.pdf")
    
    print("Testing JSON export...")
    json_path = exporter.generate_json_export(test_results, "./test_output/test_export.json")
    
    print("Testing Excel export...")
    excel_path = exporter.generate_excel_export(test_results, "./test_output/test_export.xlsx")
    
    print("Testing comprehensive export...")
    all_exports = exporter.generate_comprehensive_export(test_results, "./test_output")
    
    print(f"Generated exports: {list(all_exports.keys())}")
    print("All export tests completed successfully!")
